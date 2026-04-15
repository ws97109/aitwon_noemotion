"""
測試 v15 模型穩定性 - 重複運行 5 次測試集
評估模型輸出的一致性和方差
"""

import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, DebertaV2Tokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 複製 v15 的模型架構（簡化版，只需要推理）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 80):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.raw_text = split_data["raw_text"]
        self.audio = torch.FloatTensor(split_data["audio"])
        self.vision = torch.FloatTensor(split_data["vision"])
        self.audio_lengths = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]

        labels = split_data["regression_labels"]
        self.reg_labels = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        text = TASK_PROMPT + str(self.raw_text[idx])
        enc = self.tokenizer(
            text, add_special_tokens=True,
            max_length=self.max_text_len,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )

        aud_len = min(int(self.audio_lengths[idx]), self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx, :aud_len],
            "audio_mask": torch.ones(aud_len, dtype=torch.bool),
            "vision": self.vision[idx, :vis_len],
            "vision_mask": torch.ones(vis_len, dtype=torch.bool),
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label": self.reg_labels[idx],
        }


def collate_fn(batch):
    max_aud = max(b["audio"].size(0) for b in batch)
    max_vis = max(b["vision"].size(0) for b in batch)
    aud_dim = batch[0]["audio"].size(-1)
    vis_dim = batch[0]["vision"].size(-1)

    for b in batch:
        alen = b["audio"].size(0)
        vlen = b["vision"].size(0)
        if alen < max_aud:
            b["audio"] = torch.cat([b["audio"], torch.zeros(max_aud - alen, aud_dim)])
            b["audio_mask"] = torch.cat([b["audio_mask"], torch.zeros(max_aud - alen, dtype=torch.bool)])
        if vlen < max_vis:
            b["vision"] = torch.cat([b["vision"], torch.zeros(max_vis - vlen, vis_dim)])
            b["vision_mask"] = torch.cat([b["vision_mask"], torch.zeros(max_vis - vlen, dtype=torch.bool)])

    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "audio": torch.stack([b["audio"] for b in batch]),
        "audio_mask": torch.stack([b["audio_mask"] for b in batch]),
        "vision": torch.stack([b["vision"] for b in batch]),
        "vision_mask": torch.stack([b["vision_mask"] for b in batch]),
        "cls7_label": torch.stack([b["cls7_label"] for b in batch]),
        "cls2_label": torch.stack([b["cls2_label"] for b in batch]),
        "reg_label": torch.stack([b["reg_label"] for b in batch]),
    }


def compute_metrics(c7, c2, reg, l7, l2, lr):
    acc7 = (c7 == l7).mean() * 100
    acc2 = (c2 == l2).mean() * 100
    f1 = f1_score(l2, c2, average="weighted") * 100
    mae = np.abs(reg - lr).mean()
    corr, _ = pearsonr(reg, lr)
    return {
        "Acc7": acc7,
        "Acc2": acc2,
        "F1": f1,
        "MAE": mae,
        "Corr": corr,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def test_model(model, loader, device):
    """單次測試運行"""
    model.eval()

    all_c7, all_c2, all_reg = [], [], []
    all_l7, all_l2, all_lr = [], [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)

        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)

        c7 = l7.argmax(dim=-1).cpu().numpy()
        c2 = l2.argmax(dim=-1).cpu().numpy()
        reg_out = reg.cpu().numpy()

        all_c7.append(c7)
        all_c2.append(c2)
        all_reg.append(reg_out)
        all_l7.append(batch["cls7_label"].numpy())
        all_l2.append(batch["cls2_label"].numpy())
        all_lr.append(batch["reg_label"].numpy())

    all_c7 = np.concatenate(all_c7)
    all_c2 = np.concatenate(all_c2)
    all_reg = np.concatenate(all_reg)
    all_l7 = np.concatenate(all_l7)
    all_l2 = np.concatenate(all_l2)
    all_lr = np.concatenate(all_lr)

    return compute_metrics(all_c7, all_c2, all_reg, all_l7, all_l2, all_lr)


def main():
    print("=" * 70)
    print("v15 模型穩定性測試 - 重複運行 5 次")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n設備: {device}")

    # 載入數據
    data_path = PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl"
    print(f"載入資料: {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    test_ds = MOSIDataset(data["test"], tokenizer, max_text_len=80)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"測試集: {len(test_ds)} 筆\n")

    # 載入模型
    model_path = PROJECT_ROOT / "emotion_system/models/v15_best_acc7.pth"
    print(f"載入模型: {model_path}")

    if not model_path.exists():
        print(f"❌ 模型檔案不存在: {model_path}")
        print("請先訓練 v15 模型")
        return

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # 從 checkpoint 重建模型
    # 注意：這裡需要導入完整的模型定義，或者直接使用 checkpoint 中保存的架構
    # 為簡化，假設我們可以從 scaf_v15 導入
    from scaf_v15 import HybridModel

    config = {
        "lang_model": "microsoft/deberta-v3-large",
        "audio_dim": 5,
        "vision_dim": 20,
        "modal_hidden": 256,
        "fusion_dim": 512,
        "num_classes": 7,
        "dropout": 0.2,
    }

    model = HybridModel(**config).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"✅ 模型載入成功 (Epoch {ckpt['epoch']})\n")

    # 重複測試 5 次
    num_runs = 5
    all_results = []

    print("開始重複測試...\n")
    for run in range(1, num_runs + 1):
        print(f"【運行 {run}/{num_runs}】")

        # 設置不同的隨機種子（雖然推理時應該是確定性的）
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        metrics = test_model(model, test_loader, device)
        all_results.append(metrics)

        print(f"  Acc7 : {metrics['Acc7']:.2f}%")
        print(f"  Acc2 : {metrics['Acc2']:.2f}%")
        print(f"  F1   : {metrics['F1']:.2f}%")
        print(f"  MAE  : {metrics['MAE']:.4f}")
        print(f"  Corr : {metrics['Corr']:.4f}")
        print()

    # 計算統計數據
    print("=" * 70)
    print("統計結果")
    print("=" * 70)

    metrics_keys = ["Acc7", "Acc2", "F1", "MAE", "Corr"]
    stats = {}

    for key in metrics_keys:
        values = [r[key] for r in all_results]
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        stats[key] = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "values": values,
        }

        print(f"\n{key}:")
        print(f"  平均值: {mean:.4f}")
        print(f"  標準差: {std:.4f}")
        print(f"  最小值: {min_val:.4f}")
        print(f"  最大值: {max_val:.4f}")
        print(f"  範圍: {max_val - min_val:.4f}")

    # 保存結果
    output_path = PROJECT_ROOT / "emotion_system/models/v15_stability_test.json"
    with open(output_path, "w") as f:
        json.dump({
            "num_runs": num_runs,
            "all_results": [
                {k: float(v) for k, v in r.items()} for r in all_results
            ],
            "statistics": {
                k: {sk: float(sv) if sk != "values" else [float(x) for x in sv]
                    for sk, sv in v.items()}
                for k, v in stats.items()
            },
        }, f, indent=2)

    print(f"\n✅ 結果已保存至: {output_path}")

    # 判斷穩定性
    print("\n" + "=" * 70)
    print("穩定性評估")
    print("=" * 70)

    acc7_std = stats["Acc7"]["std"]
    acc7_range = stats["Acc7"]["max"] - stats["Acc7"]["min"]

    print(f"Acc7 標準差: {acc7_std:.4f}%")
    print(f"Acc7 範圍: {acc7_range:.4f}%")

    if acc7_std < 0.5:
        print("✅ 模型非常穩定（標準差 < 0.5%）")
    elif acc7_std < 1.0:
        print("✅ 模型穩定（標準差 < 1.0%）")
    elif acc7_std < 2.0:
        print("⚠️ 模型中等穩定（標準差 < 2.0%）")
    else:
        print("❌ 模型不穩定（標準差 >= 2.0%）")


if __name__ == "__main__":
    main()
