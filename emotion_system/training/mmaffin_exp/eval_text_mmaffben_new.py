"""MMAFFBen 純文字評估 — 以 scaf_final.py 模型骨幹為基礎。

依論文 (MMAFFBen, Liu et al. 2025) 的訓練/測試協議：
  • 在 MMAFFIn 訓練集上做 instruction-tuning（每任務一個 head）
  • 在對應 val 集評估
  • SP/EC 任務：macro-F1（ma-F1）
  • 多標籤任務 (XED-EC multi)：macro-F1 + Jaccard
  • SI/EI 任務：Pearson 相關（pcc）— 若資料中含實值標籤

【純文字改造】
SACFModel 原為「文字 + 音訊 + 視覺」三模態。本腳本只取其中：
  lang_backbone(DeBERTa-v3-large) + PolarityEnhancedAttention(text-only pooling)
  + shared MLP + per-task head
完全不接 audio / vision，因此可在純文字資料上推論。
backbone 權重可從使用者訓練好的 SACF checkpoint (sacf_final.pt) 初始化，
保留其在 MOSI 上學到的情感先驗。

【執行方式】
  python eval_text_mmaffben.py                       # 全任務、預設超參
  python eval_text_mmaffben.py --tasks MMS,XED       # 只跑指定任務
  python eval_text_mmaffben.py --max_train 2000      # 限制訓練樣本（除錯用）
  python eval_text_mmaffben.py --no_pretrain         # 不載入 SACF 權重，純 HF init

輸出：
  emotion_system/training/mmaffin_exp/mmaffben_results.json
  emotion_system/training/mmaffin_exp/MMAFFBEN_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, jaccard_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

# 重用 scaf_final.py 中的 PolarityEnhancedAttention，確保架構等價
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "emotion_system" / "training"))
from scaf_final import PolarityEnhancedAttention  # noqa: E402

MMAFFIN_DIR  = PROJECT_ROOT / "emotion_system" / "data" / "mmaffin_text"  / "texts"   # train + val
MMAFFBEN_DIR = PROJECT_ROOT / "emotion_system" / "data" / "mmaffben_text" / "texts"   # test
MODEL_DIR    = PROJECT_ROOT / "emotion_system" / "models"
SACF_CKPT    = MODEL_DIR / "sacf_final.pt"


# ───────────────────────── 任務定義 ─────────────────────────
# 對齊論文 Table 2 / Table 9 的 prompt-and-label 格式。
# label_map：將 "{number}. {label}" 字串映射到整數 class id（多標籤分號分隔）
TASKS: Dict[str, dict] = {
    "Onlineshopping": {
        "train": "onlineshopping_train.json", "val": "onlineshopping_val.json",
        "test":  "onlineshopping_test.json",
        "type":  "single",                  # SP-2
        "labels": ["-1. negative", "1. positive"],
        "task_tag": "SP", "lang": "zh",
    },
    "MMS": {
        "train": "MMS_train.json", "val": "MMS_val.json",
        "test":  "MMS_test.json",
        "type":  "single",                  # SP-3
        "labels": ["-1. negative", "0. neutral", "1. positive"],
        "task_tag": "SP", "lang": "multi",
    },
    "EWECT-usual": {
        "train": "EWECT_usual_train.json", "val": "EWECT_usual_val.json",
        "test":  "EWECT_usual_test.json",
        "type":  "single",                  # EC-single, 5+1
        "labels": ["0. neutral", "1. happiness", "2. sadness",
                   "3. anger",   "4. fear",      "5. surprise"],
        "task_tag": "EC", "lang": "zh",
    },
    "EWECT-virus": {
        "train": "EWECT_virus_train.json", "val": "EWECT_virus_val.json",
        "test":  "EWECT_virus_test.json",
        "type":  "single",
        "labels": ["0. neutral", "1. happiness", "2. sadness",
                   "3. anger",   "4. fear",      "5. surprise"],
        "task_tag": "EC", "lang": "zh",
    },
    "XED": {
        "train": "XED_train.json", "val": "XED_val.json",
        "test":  "XED_test.json",
        "type":  "multi",                   # EC-multi, 8 emotions
        "labels": ["1. joy", "2. sadness", "3. anger", "4. fear",
                   "5. surprise", "6. disgust", "7. anticipation", "8. trust"],
        "task_tag": "EC-multi", "lang": "multi",
    },
}

# 論文 Table 5（Text 區塊）報告數值，作為對比 baseline。
# 欄位順序：EWECT-usual, EWECT-virus, MMS, XED, Onlineshopping → ma-F1
PAPER_TABLE5_TEXT_BASELINES: Dict[str, Dict[str, float]] = {
    "EmoLlama-chat-7b":      {"EWECT-usual": 45.6, "EWECT-virus": 36.5, "MMS": 44.0, "XED": 48.6, "Onlineshopping": 20.3},
    "Llama3.2-1b-instruct":  {"EWECT-usual": 24.2, "EWECT-virus": 18.0, "MMS": 49.1, "XED": 30.9, "Onlineshopping": 28.7},
    "Llama3.2-3b-instruct":  {"EWECT-usual": 51.9, "EWECT-virus": 39.9, "MMS": 52.2, "XED": 33.1, "Onlineshopping": 11.9},
    "Mistral-7b-instruct":   {"EWECT-usual": 45.1, "EWECT-virus": 36.5, "MMS": 55.4, "XED": 33.1, "Onlineshopping": 23.3},
    "Llama3.2-11b-instruct": {"EWECT-usual": 42.3, "EWECT-virus": 35.8, "MMS": 57.4, "XED": 37.5, "Onlineshopping": 20.0},
    "Qwen2.5-VL-7b":         {"EWECT-usual": 46.3, "EWECT-virus": 38.1, "MMS": 56.2, "XED": 31.3, "Onlineshopping": 12.8},
    "InternVL2.5-8B-MPO":    {"EWECT-usual": 51.1, "EWECT-virus": 35.7, "MMS": 56.2, "XED": 31.4, "Onlineshopping": 12.4},
    "GPT-4o-mini":           {"EWECT-usual": 69.5, "EWECT-virus": 57.6, "MMS": 61.9, "XED": 48.6, "Onlineshopping": 12.5},
    "MMAFFLM-3b":            {"EWECT-usual": 66.9, "EWECT-virus": 60.3, "MMS": 93.9, "XED": 43.5, "Onlineshopping": 26.5},
    "MMAFFLM-7b":            {"EWECT-usual": 67.6, "EWECT-virus": 58.2, "MMS": 93.9, "XED": 46.3, "Onlineshopping": 28.8},
}


# ───────────────────────── 工具：解析 instruction-tuning 資料 ─────────────────────────
TEXT_RE = re.compile(r"Text:\s*(.+?)\s*$", re.DOTALL)


def extract_text(instruction: str) -> str:
    """從 instruction 字串裡抽出 'Text: ...' 之後的原文。"""
    m = TEXT_RE.search(instruction)
    return m.group(1).strip() if m else instruction.strip()


def parse_output(out_str: str, label_to_id: Dict[str, int],
                 multi: bool) -> List[int]:
    """將 '3. anger' 或 '1. joy, 8. trust' 解析為類別 ID list。"""
    parts = [p.strip() for p in out_str.split(",")]
    ids = []
    for p in parts:
        if p in label_to_id:
            ids.append(label_to_id[p])
    if not multi and len(ids) > 1:
        ids = ids[:1]
    return ids


def load_task(task_name: str, mmaffin_dir: Path, mmaffben_dir: Path,
              max_train: int | None
              ) -> Tuple[List[dict], List[dict], List[dict], dict]:
    cfg = TASKS[task_name]
    train_raw = json.loads((mmaffin_dir  / cfg["train"]).read_text(encoding="utf-8"))
    val_raw   = json.loads((mmaffin_dir  / cfg["val"  ]).read_text(encoding="utf-8"))
    test_raw  = json.loads((mmaffben_dir / cfg["test" ]).read_text(encoding="utf-8"))

    if max_train is not None and len(train_raw) > max_train:
        random.Random(0).shuffle(train_raw)
        train_raw = train_raw[:max_train]

    label_to_id = {lab: i for i, lab in enumerate(cfg["labels"])}
    multi       = (cfg["type"] == "multi")

    def _proc(rows):
        out = []
        for r in rows:
            txt = extract_text(r["instruction"])
            ids = parse_output(r["output"], label_to_id, multi)
            if not ids and not multi:
                continue
            out.append({"text": txt, "labels": ids})
        return out

    return _proc(train_raw), _proc(val_raw), _proc(test_raw), cfg


# ───────────────────────── Dataset ─────────────────────────
class MMAFFBenDataset(Dataset):
    def __init__(self, rows, tokenizer, n_classes: int, multi: bool,
                 max_len: int = 128):
        self.rows  = rows
        self.tok   = tokenizer
        self.n     = n_classes
        self.multi = multi
        self.max   = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r   = self.rows[i]
        enc = self.tok(r["text"], add_special_tokens=True, max_length=self.max,
                       padding="max_length", truncation=True, return_tensors="pt")
        if self.multi:
            y = torch.zeros(self.n, dtype=torch.float32)
            for lid in r["labels"]:
                y[lid] = 1.0
        else:
            y = torch.tensor(r["labels"][0], dtype=torch.long)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          y,
        }


# ───────────────────────── 純文字 SACF 變體 ─────────────────────────
class TextOnlySACF(nn.Module):
    """SACF 的文字分支：lang_backbone + PolarityEnhancedAttention + shared MLP + head。

    完全不使用 audio/vision，與 scaf_final.SACFModel 的文字路徑等價（除省略
    SentimentAwareCrossModalAttention，因該模組需要音/視覺輸入）。"""

    def __init__(self, lang_model: str, n_classes: int, multi: bool,
                 fusion_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        h = self.lang_backbone.config.hidden_size
        self.polarity_attn = PolarityEnhancedAttention(h, dropout)
        self.shared = nn.Sequential(
            nn.Linear(h, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(fusion_dim, n_classes)
        self.multi = multi

    def encode(self, input_ids, attention_mask):
        hidden = self.lang_backbone(input_ids=input_ids,
                                    attention_mask=attention_mask).last_hidden_state
        pooled, _ = self.polarity_attn(hidden, attention_mask)
        return self.shared(pooled)

    def forward(self, input_ids, attention_mask):
        return self.head(self.encode(input_ids, attention_mask))


def maybe_load_sacf_backbone(model: TextOnlySACF, ckpt_path: Path) -> str:
    """若使用者已訓練 SACF，載入其 lang_backbone + polarity_attn + shared 權重。"""
    if not ckpt_path.exists():
        return f"未找到 {ckpt_path.name}，使用 HF 預訓練初始化。"
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    own   = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(own)
    return f"從 {ckpt_path.name} 載入 {loaded} 張量（skip {skipped}，含 cls7/cls2/reg/sacf_attn 等不適用層）"


# ───────────────────────── 訓練 / 評估 ─────────────────────────
def train_one_task(model: TextOnlySACF, train_loader, val_loader, test_loader,
                   *, epochs: int, lang_lr: float, head_lr: float,
                   weight_decay: float, multi: bool, device: str,
                   amp: bool, log_prefix: str
                   ) -> Tuple[Dict[str, float], Dict[str, float]]:
    backbone_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lang_backbone" in n]
    head_p     = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lang_backbone" not in n]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_p, "lr": lang_lr},
         {"params": head_p,     "lr": head_lr}],
        weight_decay=weight_decay,
    )
    total_steps  = max(1, len(train_loader)) * epochs
    warmup_steps = int(total_steps * 0.06)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.amp.GradScaler("cuda") if (amp and "cuda" in device) else None

    crit = nn.BCEWithLogitsLoss() if multi else nn.CrossEntropyLoss(label_smoothing=0.05)

    for ep in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"{log_prefix} train E{ep}", leave=False):
            ids = batch["input_ids"].to(device)
            mk  = batch["attention_mask"].to(device)
            y   = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(ids, mk)
                loss   = crit(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                scheduler.step(); continue
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running += loss.item() * y.size(0); n += y.size(0)
        print(f"  {log_prefix} E{ep} train loss = {running / max(n,1):.4f}")
        # 每個 epoch 結束後監看 val（不用於 early-stop，僅輸出觀察）
        if val_loader is not None and len(val_loader.dataset) > 0:
            vm = evaluate_task(model, val_loader, multi=multi, device=device)
            print(f"  {log_prefix} E{ep} val   ma-F1 = {vm.get('ma-F1', 0):.2f}")
    val_metrics  = evaluate_task(model, val_loader,  multi=multi, device=device) \
                       if val_loader is not None else {}
    test_metrics = evaluate_task(model, test_loader, multi=multi, device=device)
    return val_metrics, test_metrics


@torch.no_grad()
def evaluate_task(model: TextOnlySACF, val_loader, *, multi: bool,
                  device: str) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []
    for batch in tqdm(val_loader, desc="eval", leave=False):
        ids = batch["input_ids"].to(device)
        mk  = batch["attention_mask"].to(device)
        y   = batch["label"]
        logits = model(ids, mk)
        if multi:
            p = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(np.int32)
        else:
            p = logits.argmax(-1).cpu().numpy()
        preds.append(p); gts.append(y.numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)

    if multi:
        ma_f1 = f1_score(gts, preds, average="macro", zero_division=0) * 100
        jac   = jaccard_score(gts, preds, average="samples", zero_division=0) * 100
        acc   = (preds == gts).all(axis=1).mean() * 100
        return {"ma-F1": float(ma_f1), "jaccard": float(jac), "subset-acc": float(acc)}
    else:
        ma_f1 = f1_score(gts, preds, average="macro", zero_division=0) * 100
        mi_f1 = f1_score(gts, preds, average="micro", zero_division=0) * 100
        acc   = (preds == gts).mean() * 100
        return {"ma-F1": float(ma_f1), "mi-F1": float(mi_f1), "acc": float(acc)}


# ───────────────────────── 主流程 ─────────────────────────
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang_model",   default="microsoft/deberta-v3-large")
    ap.add_argument("--gpu",          default="0")
    ap.add_argument("--batch_size",   type=int, default=16)
    ap.add_argument("--epochs",       type=int, default=1,
                    help="論文用 1 epoch instruction-tuning")
    ap.add_argument("--lang_lr",      type=float, default=5e-6,
                    help="論文用 5e-6")
    ap.add_argument("--head_lr",      type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_len",      type=int, default=128)
    ap.add_argument("--max_train",    type=int, default=None,
                    help="限制每任務訓練樣本數（除錯用）；None 表全量")
    ap.add_argument("--tasks",        default="all",
                    help="all 或逗號分隔：Onlineshopping,MMS,EWECT-usual,EWECT-virus,XED")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--no_pretrain",  action="store_true",
                    help="不載入 SACF 預訓練權重（純 HF init 對照）")
    ap.add_argument("--no_amp",       action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print("=" * 70)
    print(" MMAFFBen 純文字評估（SACF text-branch + per-task fine-tune）")
    print("=" * 70)
    print(f"  device       = {device} (CUDA_VISIBLE_DEVICES={args.gpu})")
    print(f"  backbone     = {args.lang_model}")
    print(f"  epochs       = {args.epochs}  | bs = {args.batch_size}  | max_len = {args.max_len}")
    print(f"  lr           = lang {args.lang_lr:.1e} / head {args.head_lr:.1e}")
    print(f"  pretrained   = {'no' if args.no_pretrain else 'sacf_final.pt'}")

    task_list = list(TASKS.keys()) if args.tasks == "all" \
                else [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in task_list:
        if t not in TASKS:
            sys.exit(f"未知任務 '{t}'，可選：{list(TASKS.keys())}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(args.lang_model)

    results: Dict[str, dict] = {}
    t_global = time.time()
    for tname in task_list:
        print(f"\n{'─' * 70}\n  ▶ Task: {tname}  [{TASKS[tname]['task_tag']} / {TASKS[tname]['lang']}]\n{'─' * 70}")
        train_rows, val_rows, test_rows, cfg = load_task(
            tname, MMAFFIN_DIR, MMAFFBEN_DIR, args.max_train)
        n_cls = len(cfg["labels"])
        multi = (cfg["type"] == "multi")
        print(f"  train n={len(train_rows)} | val n={len(val_rows)} | "
              f"test n={len(test_rows)} (MMAFFBen) | classes={n_cls} | multi={multi}")

        train_ds = MMAFFBenDataset(train_rows, tokenizer, n_cls, multi, args.max_len)
        val_ds   = MMAFFBenDataset(val_rows,   tokenizer, n_cls, multi, args.max_len)
        test_ds  = MMAFFBenDataset(test_rows,  tokenizer, n_cls, multi, args.max_len)
        train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
        val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
        test_ld  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

        set_seed(args.seed)
        model = TextOnlySACF(args.lang_model, n_classes=n_cls, multi=multi).to(device)
        if not args.no_pretrain:
            print(" ", maybe_load_sacf_backbone(model, SACF_CKPT))

        t0      = time.time()
        val_metrics, test_metrics = train_one_task(
            model, train_ld, val_ld, test_ld,
            epochs=args.epochs, lang_lr=args.lang_lr, head_lr=args.head_lr,
            weight_decay=args.weight_decay, multi=multi, device=device,
            amp=(not args.no_amp), log_prefix=tname,
        )
        elapsed = time.time() - t0

        del model
        if "cuda" in device:
            torch.cuda.empty_cache()

        print(f"  ⮕ {tname} TEST:", " | ".join(f"{k}={v:.2f}" for k, v in test_metrics.items()),
              f"  ({elapsed:.0f}s)")
        results[tname] = {
            "task_tag":     cfg["task_tag"],
            "language":     cfg["lang"],
            "n_classes":    n_cls,
            "multi_label":  multi,
            "n_train":      len(train_rows),
            "n_val":        len(val_rows),
            "n_test":       len(test_rows),
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
            "elapsed_sec":  round(elapsed, 1),
        }

    total_elapsed = time.time() - t_global
    print(f"\n  總耗時：{total_elapsed:.0f}s")

    # ───── 寫出 JSON ─────
    json_out = HERE / "mmaffben_results.json"
    json_out.write_text(json.dumps({
        "config": vars(args),
        "device": device,
        "sacf_ckpt_loaded": (not args.no_pretrain) and SACF_CKPT.exists(),
        "results": results,
        "total_elapsed_sec": round(total_elapsed, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ JSON: {json_out}")

    # ───── 寫出 Markdown 報告（Table 4 風格對照表）─────
    md_out = HERE / "MMAFFBEN_REPORT.md"
    # 欄位順序對齊論文 Table 5 Text 區塊
    col_order = ["EWECT-usual", "EWECT-virus", "MMS", "XED", "Onlineshopping"]
    cols_present = [c for c in col_order if c in results]

    def fmt(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

    def avg_row(metrics_dict: Dict[str, float]) -> float:
        vals = [metrics_dict[c] for c in cols_present if c in metrics_dict]
        return float(np.mean(vals)) if vals else 0.0

    # 我們的 row（test set）
    our_text = {c: results[c]["test_metrics"]["ma-F1"] for c in cols_present}
    our_label = ("SACF-Text (ours, sacf_v60 backbone)" if not args.no_pretrain
                 else "SACF-Text (ours, HF init)")

    lines = []
    lines.append("# MMAFFBen 純文字評估報告（Table 4 風格）")
    lines.append("")
    lines.append("依論文 *MMAFFBen: A Multilingual and Multimodal Affective Analysis "
                 "Benchmark* (Liu et al. 2025) 的訓練 / 測試協議：")
    lines.append("")
    lines.append("- **訓練資料**：MMAFFIn `texts/*_train.json`（instruction-tuning 格式）")
    lines.append("- **驗證資料**：MMAFFIn `texts/*_val.json`（過程監看，不用於早停）")
    lines.append("- **測試資料**：**MMAFFBen** `texts/*_test.json`（最終指標來源）")
    lines.append("- **基底模型**：`scaf_final.SACFModel` 的純文字分支 "
                 "（DeBERTa-v3-large + PolarityEnhancedAttention + shared MLP + per-task head）")
    lines.append(f"- **backbone 初始化**：{'`sacf_final.pt`（使用者 MOSI 訓練成果）' if not args.no_pretrain else 'HF 預訓練（無 SACF init）'}")
    lines.append(f"- **訓練超參數**：每任務獨立訓練 {args.epochs} epoch，"
                 f"batch={args.batch_size}, lang_lr={args.lang_lr:.0e}, head_lr={args.head_lr:.0e}（論文用 lr=5e-6）")
    lines.append("- **指標**：SP / EC 任務皆報 macro-F1（ma-F1）；XED 多標籤額外報 Jaccard")
    lines.append("")
    lines.append("## 主結果（macro-F1，越高越好）")
    lines.append("")

    # Header（與 Table 4 樣式一致：Models | task1 | task2 | ... | Average）
    header = "| Models | " + " | ".join(cols_present) + " | Average |"
    sep    = "|--------|" + "|".join(["------:"] * len(cols_present)) + "|--------:|"
    lines.append(header)
    lines.append(sep)

    # 我們的行（粗體）
    cells = " | ".join(f"**{fmt(our_text[c])}**" for c in cols_present)
    lines.append(f"| **{our_label}** | {cells} | **{fmt(avg_row(our_text))}** |")

    # 論文 baseline 行
    for model_name, scores in PAPER_TABLE5_TEXT_BASELINES.items():
        cells = " | ".join(fmt(scores.get(c, "—")) for c in cols_present)
        lines.append(f"| {model_name} | {cells} | {fmt(avg_row(scores))} |")

    lines.append("")
    lines.append("> 上表中「Average」= 列出任務之 ma-F1 平均。論文數值取自 Table 5 Text 區塊；"
                 "MMAFFLM-3b/7b 為論文作者用 MMAFFIn 微調 Qwen2.5-VL 的成果，是同設定下的上界參照。")
    lines.append("")

    # 詳細指標（含 mi-F1, accuracy, jaccard 等）
    lines.append("## 詳細指標（test set）")
    lines.append("")
    lines.append("| Task | Tag | Lang | #cls | n_train | n_val | n_test | ma-F1 | 其他 |")
    lines.append("|------|-----|------|-----:|--------:|------:|-------:|------:|------|")
    for t, r in results.items():
        m = r["test_metrics"]
        ma_f1 = m.get("ma-F1", 0.0)
        if r["multi_label"]:
            extra = f"jac={m.get('jaccard',0):.2f}, subset-acc={m.get('subset-acc',0):.2f}"
        else:
            extra = f"mi-F1={m.get('mi-F1',0):.2f}, acc={m.get('acc',0):.2f}"
        lines.append(f"| {t} | {r['task_tag']} | {r['language']} | {r['n_classes']} | "
                     f"{r['n_train']} | {r['n_val']} | {r['n_test']} | "
                     f"**{ma_f1:.2f}** | {extra} |")
    lines.append("")

    # Val vs Test 對照（觀察 MMAFFIn val 與 MMAFFBen test 的差異 → 過擬合診斷）
    lines.append("## Val (MMAFFIn) vs Test (MMAFFBen) 對照")
    lines.append("")
    lines.append("| Task | val ma-F1 | test ma-F1 | Δ |")
    lines.append("|------|----------:|-----------:|----:|")
    for t, r in results.items():
        v = r["val_metrics"].get("ma-F1", 0.0) if r["val_metrics"] else 0.0
        te = r["test_metrics"].get("ma-F1", 0.0)
        lines.append(f"| {t} | {v:.2f} | {te:.2f} | {te-v:+.2f} |")
    lines.append("")
    lines.append(f"_耗時：{round(total_elapsed)}s，設備：{device}_")

    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Markdown: {md_out}")

    print("\n" + "=" * 70)
    print(" 平均 ma-F1 (test):", f"{avg_row(our_text):.2f}")
    for t, r in results.items():
        print(f"  {t:18s}  test ma-F1 = {r['test_metrics']['ma-F1']:6.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
