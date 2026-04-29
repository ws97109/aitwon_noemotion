"""
SCAF Final — 多版本集成訓練主腳本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本腳本依序訓練 4 個訓練協議 × 3 個種子 = 12 個獨立的 SWA 模型，
最終以 raw argmax 取得多版本集成的測試集 Acc-7（無條件、零洩漏）。

協議定義：
  P1 (v59 風格):           TrainVal + EMD + TTA×3,  種子 [42, 123, 2024]
  P2 (v60_baseline 風格):  TrainVal + EMD + TTA×5,  種子 [42, 123, 2024]
  P3 (v60_mmaffin 風格):   TrainVal + EMD + TTA×5 + MMAffBen 預訓練骨幹, 種子 [42, 123, 2024]
  P4 (v63 風格):           TrainVal + EMD + TTA×5,  種子 [101, 202, 303]

權重儲存：
  emotion_system/models/sacf_final_p{1..4}_seed{seed}.pt   (12 個檔案)
  emotion_system/models/raw_logits_final_p{1..4}.npy        (4 個 [3, 686, 7] 陣列)
  emotion_system/models/sacf_final_summary.json            (12 個模型 + 集成的指標)

具備斷點續訓：執行時會偵測已存在的權重，跳過已訓練完成的協議+種子組合。

執行：
  CUDA_VISIBLE_DEVICES=0 python emotion_system/training/scaf_final.py

預期執行時間：~6 小時（每個 SWA 模型約 30 分鐘）
預期測試集 Acc-7：~53%（無條件、零洩漏）
"""

import pickle, random, os, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import (
    AutoModel, AutoTokenizer, DebertaV2Tokenizer,
    get_cosine_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl"
MODEL_DIR = PROJECT_ROOT / "emotion_system/models"
MMAFFIN_BACKBONE = MODEL_DIR / "mmaffin_pretrain_backbone.pt"

TASK_PROMPT = "Predict the sentiment intensity (-3 to 3, negative to positive) of the following text: "


# ─────────────────────────────────────────────────────────────
# Protocols
# ─────────────────────────────────────────────────────────────
PROTOCOLS = [
    {"id": "p1", "name": "v59 style", "n_tta": 3, "use_mmaffin": False, "seeds": [42, 123, 2024]},
    {"id": "p2", "name": "v60_baseline style", "n_tta": 5, "use_mmaffin": False, "seeds": [42, 123, 2024]},
    {"id": "p3", "name": "v60_mmaffin style", "n_tta": 5, "use_mmaffin": True,  "seeds": [42, 123, 2024]},
    {"id": "p4", "name": "v63 style", "n_tta": 5, "use_mmaffin": False, "seeds": [101, 202, 303]},
]


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class MOSIDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_text_len=80):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.raw_text = split_data["raw_text"]
        self.audio  = torch.FloatTensor(split_data["audio"])
        self.vision = torch.FloatTensor(split_data["vision"])
        self.audio_lengths  = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]
        labels = split_data["regression_labels"]
        self.reg_labels  = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))

    def __len__(self): return len(self.raw_text)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            TASK_PROMPT + str(self.raw_text[idx]),
            add_special_tokens=True, max_length=self.max_text_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        aud_len = min(int(self.audio_lengths[idx]), self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]); aud_mask[:aud_len] = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx], "audio_mask": aud_mask,
            "vision": self.vision[idx], "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label":  self.reg_labels[idx],
        }


# ─────────────────────────────────────────────────────────────
# SACF model components
# ─────────────────────────────────────────────────────────────
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4), nn.Tanh(),
            nn.Linear(hidden_dim//4, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
    def forward(self, hidden, mask):
        g = self.gate(hidden); m = mask.unsqueeze(-1).float()
        pooled = ((0.75*hidden + 0.25*hidden*g)*m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.dropout(pooled), (g*m).squeeze(-1)


class SentimentAwareCrossModalAttention(nn.Module):
    def __init__(self, lang_dim, modal_dim, top_k=5, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.token_attn = nn.Linear(lang_dim, 1)
        self.ffn  = nn.Sequential(
            nn.Linear(lang_dim, lang_dim//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim//2, lang_dim))
        self.gate    = nn.Linear(lang_dim*2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(lang_dim)
    def forward(self, xl_hidden, xl_cls, gates, xa, xv):
        B, L, H = xl_hidden.shape
        topk_idx = gates.topk(min(self.top_k, L), dim=1).indices
        topk_h   = xl_hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1,-1,H))
        w = F.softmax(self.token_attn(topk_h), dim=1)
        sa_q = (topk_h*w).sum(1)
        kv   = torch.stack([self.audio_map(xa), self.vision_map(xv)], dim=1)
        attn = F.softmax(torch.bmm(sa_q.unsqueeze(1), kv.transpose(1,2))/(H**0.5), dim=-1)
        x_hat = torch.bmm(attn, kv).squeeze(1)
        x = self.ffn(xl_cls + x_hat)
        gw = torch.sigmoid(self.gate(torch.cat([xl_cls, x], dim=-1)))
        return self.norm(xl_cls + self.dropout(x*gw))


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim*2, hidden_dim)
    def forward(self, x, mask):
        lengths = mask.sum(1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[-2], h[-1]], dim=-1))


class SACFModel(nn.Module):
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=128,
                 fusion_dim=512, top_k=5, num_classes=7, dropout=0.15):
        super().__init__()
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size
        self.polarity_attn = PolarityEnhancedAttention(lang_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_dim, modal_hidden, 2, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, modal_hidden, 2, dropout)
        self.sacf_attn = SentimentAwareCrossModalAttention(lang_dim, modal_hidden, top_k, dropout)
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Linear(fusion_dim//2, 1), nn.Tanh())

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        audio = F.normalize(torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        vision = F.normalize(torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        hidden = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)
        fused = self.sacf_attn(hidden, xl_cls, gates, xa, xv)
        feat = self.shared(fused)
        return self.cls7_head(feat), self.cls2_head(feat), self.reg_head(feat).squeeze(-1) * 3.0


class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = model; self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay*self.shadow[n] + (1-self.decay)*p.data
    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data = self.shadow[n]
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup: p.data = self.backup[n]
        self.backup = {}
    def add_new_params(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n not in self.shadow:
                self.shadow[n] = p.data.clone()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.label_smoothing = label_smoothing
    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        with torch.no_grad():
            log_pt = F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_w = (1.0 - log_pt.exp()) ** self.gamma
        return (focal_w * ce).mean()


class OrdinalEMDLoss(nn.Module):
    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=-1)
        cdf_pred = probs.cumsum(dim=-1)[:, :-1]
        cdf_true = F.one_hot(labels, 7).float().cumsum(dim=-1)[:, :-1]
        return (cdf_pred - cdf_true).abs().mean()


def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.where((c := np.bincount(cl, minlength=n).astype(float)) == 0, 1.0, c)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))


# ─────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, cls7_crit, cls2_crit, reg_crit,
                optimizer, scheduler, device, scaler, ema,
                rdrop_alpha=0.05, emd_crit=None, emd_weight=0.25):
    model.train(); total_loss = 0.0; nan_count = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device); amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device); vmask = batch["vision_mask"].to(device)
        cl7   = batch["cls7_label"].to(device)
        cl2   = batch["cls2_label"].to(device)
        rl    = batch["reg_label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            cls7_loss = cls7_crit(l7, cl7)
            if emd_crit is not None and emd_weight > 0:
                cls7_loss = (1.0 - emd_weight)*cls7_loss + emd_weight*emd_crit(l7, cl7)
            loss = cls7_loss + 0.3*cls2_crit(l2, cl2) + 0.4*reg_crit(reg, rl)
            if rdrop_alpha > 0:
                l7b, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7,-1), F.softmax(l7b,-1).detach(), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b,-1), F.softmax(l7,-1).detach(), reduction='batchmean'))/2
                loss = loss + rdrop_alpha * kl

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1; scheduler.step(); continue

        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        ema.update(); scheduler.step()
        if hasattr(scheduler, '_new_group_sched'):
            scheduler._new_group_sched.step()
        total_loss += loss.item()
    if nan_count > 0:
        print(f"  [warn] {nan_count} NaN batches", end="")
    return total_loss / max(len(loader) - nan_count, 1)


def get_test_outputs_tta(model, loader, device, n_tta=5):
    """Run n_tta MC-Dropout passes on test set; return mean logits."""
    if n_tta <= 1:
        model.eval()
        all_l7, all_l2, all_reg = [], [], []
        with torch.no_grad():
            for b in loader:
                ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
                aud = b["audio"].to(device); amask = b["audio_mask"].to(device)
                vis = b["vision"].to(device); vmask = b["vision_mask"].to(device)
                l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
                all_l7.append(l7.cpu().float().numpy())
                all_l2.append(l2.cpu().float().numpy())
                all_reg.append(reg.cpu().float().numpy())
        return (np.concatenate(all_l7), np.concatenate(all_l2), np.concatenate(all_reg))
    # TTA mode
    runs_l7, runs_l2, runs_reg = [], [], []
    model.train()
    for _ in range(n_tta):
        rl7, rl2, rreg = [], [], []
        for b in loader:
            ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
            aud = b["audio"].to(device); amask = b["audio_mask"].to(device)
            vis = b["vision"].to(device); vmask = b["vision_mask"].to(device)
            with torch.no_grad():
                l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            rl7.append(l7.cpu().float().numpy())
            rl2.append(l2.cpu().float().numpy())
            rreg.append(reg.cpu().float().numpy())
        runs_l7.append(np.concatenate(rl7))
        runs_l2.append(np.concatenate(rl2))
        runs_reg.append(np.concatenate(rreg))
    model.eval()
    return (np.mean(runs_l7, axis=0), np.mean(runs_l2, axis=0), np.mean(runs_reg, axis=0))


def train_one_protocol_seed(protocol, seed, train_loader, test_loader,
                             class_weights, device, common_cfg):
    """Train one (protocol, seed) → save weight + return TTA logits."""
    set_seed(seed)
    print(f"\n{'─'*60}")
    print(f"  協議 {protocol['id'].upper()} ({protocol['name']}) | seed={seed} | TTA×{protocol['n_tta']}"
          f"{' | + MMAffBen 預訓練' if protocol['use_mmaffin'] else ''}")
    print(f"{'─'*60}")

    model = SACFModel(dropout=common_cfg["dropout"]).to(device)

    # Load MMAffBen pretrained backbone if applicable
    if protocol["use_mmaffin"] and MMAFFIN_BACKBONE.exists():
        try:
            ck = torch.load(str(MMAFFIN_BACKBONE), map_location='cpu', weights_only=False)
            sd = ck.get("model_state_dict", ck)
            backbone_sd = {k.replace("lang_backbone.", ""): v for k, v in sd.items()
                           if k.startswith("lang_backbone.")}
            if backbone_sd:
                model.lang_backbone.load_state_dict(backbone_sd, strict=False)
                print(f"  ✓ 已載入 MMAffBen 預訓練骨幹")
        except Exception as e:
            print(f"  [warn] MMAffBen 載入失敗：{e}（使用 HF 預訓練）")

    # Freeze bottom 6 layers
    for i in range(6):
        for p in model.lang_backbone.encoder.layer[i].parameters():
            p.requires_grad = False

    backbone_p = [p for n, p in model.named_parameters() if p.requires_grad and "lang_backbone" in n]
    head_p     = [p for n, p in model.named_parameters() if p.requires_grad and "lang_backbone" not in n]
    optimizer = optim.AdamW([
        {"params": backbone_p, "lr": common_cfg["lang_lr"]},
        {"params": head_p,     "lr": common_cfg["head_lr"]},
    ], weight_decay=common_cfg["weight_decay"])

    total_steps = len(train_loader) * common_cfg["num_epochs"]
    warmup_steps = int(total_steps * 0.06)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema = EMA(model, 0.9995)

    cw = class_weights.to(device)
    cls7_crit = FocalLoss(common_cfg["focal_gamma"], cw, common_cfg["label_smoothing"])
    cls2_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    reg_crit  = nn.SmoothL1Loss()

    swa_states = []
    for epoch in range(common_cfg["num_epochs"]):
        # Unfreeze bottom 6 layers at 1/3
        if epoch == common_cfg["num_epochs"] // 3 and not getattr(model, '_unfroze', False):
            new_p = []
            for i in range(6):
                for p in model.lang_backbone.encoder.layer[i].parameters():
                    p.requires_grad = True; new_p.append(p)
            optimizer.add_param_group({
                "params": new_p, "lr": common_cfg["lang_lr"]/2,
                "weight_decay": common_cfg["weight_decay"]
            })
            current_step = epoch * len(train_loader)
            new_idx = len(optimizer.param_groups) - 1

            def _cosine_lambda(step, base=current_step, w=warmup_steps, t=total_steps):
                a = base + step
                if a < w: return float(a) / float(max(1, w))
                p = float(a - w) / float(max(1, t - w))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))

            from torch.optim.lr_scheduler import LambdaLR
            new_sched = LambdaLR(optimizer, lr_lambda=[lambda s: 1.0]*new_idx + [_cosine_lambda])
            scheduler._new_group_sched = new_sched
            ema.add_new_params(); model._unfroze = True
            print(f"  [E{epoch+1}] 解凍 6 下層")

        loss = train_epoch(model, train_loader, cls7_crit, cls2_crit, reg_crit,
                           optimizer, scheduler, device, scaler, ema,
                           rdrop_alpha=common_cfg["rdrop_alpha"],
                           emd_crit=OrdinalEMDLoss() if common_cfg["emd_weight"] > 0 else None,
                           emd_weight=common_cfg["emd_weight"])
        ep1 = epoch + 1
        # SWA snapshot
        if ep1 >= common_cfg["swa_start"] and (ep1 - common_cfg["swa_start"]) % common_cfg["swa_step"] == 0:
            ema.apply_shadow()
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            ema.restore()
            print(f"  E{ep1:02d} | Loss={loss:.4f} | [SWA #{len(swa_states)} saved]")
        else:
            if ep1 % 5 == 0:
                print(f"  E{ep1:02d} | Loss={loss:.4f}")

    # Average SWA states
    print(f"  [SWA] 平均 {len(swa_states)} 個快照 →", end=" ")
    swa_state = {}
    for k in swa_states[0]:
        if swa_states[0][k].dtype.is_floating_point:
            swa_state[k] = torch.stack([s[k].float() for s in swa_states]).mean(0).to(swa_states[0][k].dtype)
        else:
            swa_state[k] = swa_states[-1][k]
    model.load_state_dict(swa_state); model.to(device)
    print("完成")

    # TTA inference on test
    test_l7, test_l2, test_reg = get_test_outputs_tta(model, test_loader, device, n_tta=protocol["n_tta"])

    # Save weight
    save_path = MODEL_DIR / f'sacf_final_{protocol["id"]}_seed{seed}.pt'
    torch.save({
        "model_config": {"lang_model": "microsoft/deberta-v3-large", "audio_dim": 5,
                         "vision_dim": 20, "modal_hidden": 128, "fusion_dim": 512,
                         "top_k": 5, "num_classes": 7},
        "model_state_dict": swa_state,
        "protocol": protocol["id"], "seed": seed,
        "n_tta": protocol["n_tta"], "use_mmaffin": protocol["use_mmaffin"],
    }, save_path)
    print(f"  ✓ 權重已儲存：{save_path.name}")
    return test_l7, test_l2, test_reg, swa_state


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SCAF Final — 4 協議 × 3 種子 = 12 個 SWA 模型集成訓練")
    print("=" * 70)

    common_cfg = {
        "batch_size": 8, "num_epochs": 60,
        "lang_lr": 4e-6, "head_lr": 8e-5,
        "weight_decay": 0.01, "dropout": 0.15,
        "label_smoothing": 0.05, "rdrop_alpha": 0.05,
        "focal_gamma": 2.0, "swa_start": 42, "swa_step": 2,
        "emd_weight": 0.25,
    }

    print(f"\n載入資料：{DATA_PATH}")
    with open(DATA_PATH, "rb") as f: data = pickle.load(f)
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")

    train_ds = MOSIDataset(data["train"], tokenizer)
    val_ds   = MOSIDataset(data["valid"], tokenizer)
    test_ds  = MOSIDataset(data["test"],  tokenizer)
    print(f"  Train={len(train_ds)} Valid={len(val_ds)} Test={len(test_ds)}")

    bs = common_cfg["batch_size"]
    trainval_ds = ConcatDataset([train_ds, val_ds])
    train_loader = DataLoader(trainval_ds, bs, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, bs, shuffle=False, num_workers=2, pin_memory=True)

    all_train_labels = np.concatenate([data["train"]["regression_labels"],
                                        data["valid"]["regression_labels"]])
    class_weights = compute_class_weights(all_train_labels)

    test_labels_np = np.array(data["test"]["regression_labels"])
    test_cls7_true = np.clip(np.round(test_labels_np).astype(int), -3, 3) + 3

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  設備：{device}")
    print(f"  類別權重：{[f'{w:.3f}' for w in class_weights.tolist()]}")
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # ─── Train 4 protocols × 3 seeds ───
    protocol_logits = {}     # {pid: list of [3 seeds] of [test_l7, test_l2, test_reg]}
    protocol_states = {}     # {pid: list of state_dicts}

    for proto in PROTOCOLS:
        print(f"\n\n╔══════════════════════════════════════════════════════════╗")
        print(f"║  協議 {proto['id'].upper()} : {proto['name']} ({len(proto['seeds'])} seeds)")
        print(f"╚══════════════════════════════════════════════════════════╝")
        proto_l7, proto_l2, proto_reg = [], [], []
        for seed in proto["seeds"]:
            save_path = MODEL_DIR / f'sacf_final_{proto["id"]}_seed{seed}.pt'
            logits_path = MODEL_DIR / f'_proto_{proto["id"]}_seed{seed}_logits.npz'
            if save_path.exists() and logits_path.exists():
                print(f"  ⊙ 跳過 {proto['id']}/seed{seed}（已存在）")
                d = np.load(logits_path)
                proto_l7.append(d['l7']); proto_l2.append(d['l2']); proto_reg.append(d['reg'])
                continue
            l7, l2, reg, _state = train_one_protocol_seed(
                proto, seed, train_loader, test_loader, class_weights, device, common_cfg)
            np.savez(logits_path, l7=l7, l2=l2, reg=reg)
            proto_l7.append(l7); proto_l2.append(l2); proto_reg.append(reg)
        # Save per-protocol mean logits
        np.save(MODEL_DIR / f'raw_logits_final_{proto["id"]}.npy', np.stack(proto_l7))
        protocol_logits[proto["id"]] = (np.mean(proto_l7, axis=0),
                                          np.mean(proto_l2, axis=0),
                                          np.mean(proto_reg, axis=0))

    # ─── Final ensemble ───
    print(f"\n\n{'═'*70}")
    print(f"  最終多版本集成（4 協議的等權平均，無後處理）")
    print(f"{'═'*70}")
    ens_l7  = np.mean([protocol_logits[p["id"]][0] for p in PROTOCOLS], axis=0)
    ens_l2  = np.mean([protocol_logits[p["id"]][1] for p in PROTOCOLS], axis=0)
    ens_reg = np.mean([protocol_logits[p["id"]][2] for p in PROTOCOLS], axis=0)

    pred7 = ens_l7.argmax(1)
    test_cls2_true = (test_labels_np >= 0).astype(int)
    pred2 = ens_l2.argmax(1)
    acc7 = (pred7 == test_cls7_true).mean() * 100
    acc2 = (pred2 == test_cls2_true).mean() * 100
    f1   = f1_score(test_cls2_true, pred2, average='weighted') * 100
    mae  = np.abs(ens_reg - test_labels_np).mean()
    corr = pearsonr(ens_reg.astype(float), test_labels_np.astype(float))[0]
    within1 = (np.abs(pred7 - test_cls7_true) <= 1).mean() * 100

    print(f"\n  ╭─────────────────────────────────────────────────────╮")
    print(f"  │  Acc-7     ：{acc7:>6.2f} %                                  │")
    print(f"  │  Acc-2     ：{acc2:>6.2f} %                                  │")
    print(f"  │  F1        ：{f1:>6.2f} %                                  │")
    print(f"  │  MAE       ：{mae:>6.4f}                                    │")
    print(f"  │  Corr      ：{corr:>6.4f}                                    │")
    print(f"  │  Within-1  ：{within1:>6.2f} %                                  │")
    print(f"  ╰─────────────────────────────────────────────────────╯")
    print(f"\n  vs MOSI Acc-7 SOTA (MSAmba 49.67%)：{acc7-49.67:+.2f} %")

    # Save final summary
    np.save(MODEL_DIR / "raw_logits_final.npy", ens_l7)
    summary = {
        "protocols": [{"id": p["id"], "name": p["name"], "seeds": p["seeds"],
                       "n_tta": p["n_tta"], "use_mmaffin": p["use_mmaffin"]}
                      for p in PROTOCOLS],
        "common_cfg": common_cfg,
        "final_metrics": {"Acc-7": round(acc7, 2), "Acc-2": round(acc2, 2),
                           "F1": round(f1, 2), "MAE": round(mae, 4),
                           "Corr": round(corr, 4), "Within-1": round(within1, 2)},
        "n_models_in_ensemble": sum(len(p["seeds"]) for p in PROTOCOLS),
        "weights_pattern": "sacf_final_p{1..4}_seed{seed}.pt",
    }
    with open(MODEL_DIR / "sacf_final_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ 摘要已儲存：sacf_final_summary.json")
    print(f"  ✓ 共 {summary['n_models_in_ensemble']} 個 SWA 權重存於 {MODEL_DIR}")


if __name__ == "__main__":
    main()
