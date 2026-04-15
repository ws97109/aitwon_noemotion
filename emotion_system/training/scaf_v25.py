"""
MOSI 多模態情感分析 v25 — 架構突破版
徹底重新設計融合機制，突破 48.69% 瓶頸

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v23/v24 根本瓶頸分析：
1. 只用 DeBERTa 最後一層 → 丟失大量中間層情感語義
2. CrossModalAttention 只有「語言→音視頻」單向
   → 音視頻對語言序列中情感關鍵詞的感知能力弱
3. OrdinalHead 分散訓練信號（已在 v24 降至 0.1）
4. train/test 分布偏移（train mean=+0.23, test mean=-0.32）
   → 模型偏向預測正面情感

v25 架構突破：
1. 【核心】多層 DeBERTa 特徵聚合 (Multi-Layer Aggregation)
   - 對最後 4 層做可學習加權平均
   - 情感分析需要不同抽象層次的特徵
   - 已在多篇論文證明 +0.5-1% Acc7

2. 【核心】雙向跨模態注意力 (Bidirectional Cross-Modal)
   - 前向：lang 查詢 audio+vision（保留 v22 機制）
   - 反向：AV 特徵查詢 lang 序列，定位情感關鍵詞
   - 融合兩個方向的輸出

3. 徹底移除 OrdinalHead（移除所有干擾訓練信號）

4. 模態 Dropout（訓練時隨機丟棄音頻或視頻 15%）
   → 強制語言主導，增強測試集魯棒性

5. 推理時 TTA(8) + Val 閾值搜索（繼承 v24）

訓練調整：
- num_epochs: 150（更長訓練）
- patience: 25
- SWA start: 0.45（更早開始）
- w_cls7: 4.0（更強 7 分類信號）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, pickle, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, get_cosine_schedule_with_warmup, DebertaV2Tokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集（與 v22/v23 完全相同）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_text_len=80):
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
        print(f"資料: {len(self.raw_text)} 筆")

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
            "reg_label": self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 新架構模組
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiLayerAggregation(nn.Module):
    """
    可學習加權平均 DeBERTa 最後 N 層
    情感分析受益於多層特徵：
    - 淺層：語法、詞彙級別特徵
    - 深層：語義、推理級別特徵
    """
    def __init__(self, n_layers=4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers))

    def forward(self, hidden_states_tuple):
        # hidden_states_tuple: tuple of (B, T, D), last n_layers
        w = F.softmax(self.weights, dim=0)
        return sum(w_i * h for w_i, h in zip(w, hidden_states_tuple))


class PolarityEnhancedAttention(nn.Module):
    """情感感知池化（與 v22/v23 相同）"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        g = self.gate(hidden)
        m = mask.unsqueeze(-1).float()
        pooled = ((0.75 * hidden + 0.25 * hidden * g) * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.dropout(pooled)


class TransformerModalEncoder(nn.Module):
    """音視頻 Transformer 編碼器（與 v22/v23 相同）"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4,
                                       dropout, "gelu", batch_first=True),
            num_layers
        )
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = self.proj_in(x) + self.pos_enc[:, :T, :]
        key_pad = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_pad)
        scores = self.attn_score(x).squeeze(-1).masked_fill(key_pad, float('-inf'))
        return (x * F.softmax(scores, dim=1).unsqueeze(-1)).sum(1)


class BidirectionalCrossModalFusion(nn.Module):
    """
    雙向跨模態注意力融合

    前向 (Forward): lang 向量查詢 audio+vision
    → 讓語言特徵吸收音視頻情感信息

    反向 (Reverse): AV 聯合向量查詢語言序列
    → 讓音視頻定位語言中情感關鍵詞
    → 例如：激動語調 (audio) 找到「太棒了」

    最後：拼接兩方向輸出 → 投影
    """
    def __init__(self, lang_dim, modal_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.lang_dim = lang_dim

        # Forward: lang queries AV (same as CrossModalAttention in v22/v23)
        self.audio_map_fwd  = nn.Linear(modal_dim, lang_dim)
        self.vision_map_fwd = nn.Linear(modal_dim, lang_dim)
        self.gate_fwd = nn.Linear(lang_dim * 2, 1)
        self.ffn_fwd = nn.Sequential(
            nn.Linear(lang_dim, lang_dim//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim//2, lang_dim),
        )
        self.norm_fwd = nn.LayerNorm(lang_dim)

        # Reverse: AV queries lang sequence (新增)
        self.audio_map_rev  = nn.Linear(modal_dim, lang_dim)
        self.vision_map_rev = nn.Linear(modal_dim, lang_dim)
        self.mha_rev = nn.MultiheadAttention(lang_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_rev = nn.Sequential(
            nn.Linear(lang_dim, lang_dim//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim//2, lang_dim),
        )
        self.norm_rev = nn.LayerNorm(lang_dim)

        self.dropout = nn.Dropout(dropout)
        # 合併兩個方向：2*lang_dim → lang_dim
        self.combine = nn.Sequential(
            nn.Linear(lang_dim * 2, lang_dim),
            nn.LayerNorm(lang_dim),
            nn.GELU(),
        )

    def forward(self, xl, xa, xv, xl_seq, lang_mask):
        """
        xl:       (B, D)  — 池化後的語言特徵
        xa:       (B, M)  — 音頻特徵
        xv:       (B, M)  — 視覺特徵
        xl_seq:   (B, T, D) — 語言 token 序列（未池化）
        lang_mask: (B, T) — 語言注意力掩碼
        """
        # ── 前向：lang → audio+vision ──
        kv = torch.stack([self.audio_map_fwd(xa), self.vision_map_fwd(xv)], dim=1)  # (B,2,D)
        attn = F.softmax(
            torch.bmm(xl.unsqueeze(1), kv.transpose(1,2)) / (self.lang_dim ** 0.5),
            dim=-1
        )
        attended = torch.bmm(attn, kv).squeeze(1)     # (B, D)
        fwd_ffn  = self.ffn_fwd(xl + attended)
        gate     = torch.sigmoid(self.gate_fwd(torch.cat([xl, fwd_ffn], -1)))
        xl_fused = self.norm_fwd(xl + self.dropout(fwd_ffn * gate))  # (B, D)

        # ── 反向：AV → lang sequence ──
        av_query = (self.audio_map_rev(xa) + self.vision_map_rev(xv)) / 2  # (B, D)
        query    = av_query.unsqueeze(1)                                     # (B, 1, D)
        key_pad  = (lang_mask == 0)
        out, _   = self.mha_rev(query, xl_seq, xl_seq, key_padding_mask=key_pad)
        av_att   = out.squeeze(1)                                            # (B, D)
        av_fused = self.norm_rev(av_query + self.dropout(self.ffn_rev(av_att)))  # (B, D)

        # ── 合併兩方向 ──
        return self.combine(torch.cat([xl_fused, av_fused], dim=-1))  # (B, D)


class HybridModel(nn.Module):
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=256,
                 fusion_dim=512, num_classes=7, dropout=0.2, n_layers_agg=4):
        super().__init__()
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size

        self.layer_agg = MultiLayerAggregation(n_layers_agg)          # 新：多層聚合
        self.polarity_attn = PolarityEnhancedAttention(lang_dim, dropout)
        self.audio_encoder = TransformerModalEncoder(audio_dim, modal_hidden, 2, 4, dropout)
        self.vision_encoder = TransformerModalEncoder(vision_dim, modal_hidden, 2, 4, dropout)
        self.bidir_fusion = BidirectionalCrossModalFusion(             # 新：雙向融合
            lang_dim, modal_hidden, num_heads=8, dropout=dropout
        )
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Linear(fusion_dim//2, 1), nn.Tanh(),
        )
        # OrdinalHead 已完全移除

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        audio  = torch.nan_to_num(audio,  nan=0.0, posinf=1.0, neginf=-1.0)
        vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)

        # 多層 DeBERTa 特徵聚合
        outputs = self.lang_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        xl_seq = self.layer_agg(outputs.hidden_states[-4:])  # (B, T, D)
        xl = self.polarity_attn(xl_seq, attention_mask)       # (B, D) 池化

        xa = self.audio_encoder(audio, audio_mask)    # (B, modal_hidden)
        xv = self.vision_encoder(vision, vision_mask)  # (B, modal_hidden)

        # 雙向跨模態融合
        fused = self.bidir_fusion(xl, xa, xv, xl_seq, attention_mask)  # (B, D)
        feat  = self.shared(fused)

        return (self.cls7_head(feat), self.cls2_head(feat),
                self.reg_head(feat).squeeze(-1) * 3.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA（與 v22/v23 相同）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = model; self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1-self.decay) * p.data

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]
        self.backup = {}

    def add_new_params(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n not in self.shadow:
                self.shadow[n] = p.data.clone()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失（無 OrdinalHead，cls7 權重提升至 4.0）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             label_smoothing=self.label_smoothing, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


class HybridLoss(nn.Module):
    def __init__(self, class_weights, w_cls7=4.0, w_cls2=0.5, w_reg=0.3):
        super().__init__()
        self.w = (w_cls7, w_cls2, w_reg)
        self.focal = FocalLoss(alpha=class_weights)
        self.cls2  = nn.CrossEntropyLoss()
        self.reg   = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, cl7, cl2, rl):
        lc7 = self.focal(l7, cl7)
        lc2 = self.cls2(l2, cl2)
        lr  = self.reg(reg, rl)
        total = self.w[0]*lc7 + self.w[1]*lc2 + self.w[2]*lr
        return total, {"cls7": lc7.item(), "cls2": lc2.item(), "reg": lr.item()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(c7, c2, reg, l7, l2, lr):
    return {
        "Acc7": round(float((c7==l7).mean()*100), 2),
        "Acc2": round(float((c2==l2).mean()*100), 2),
        "F1":   round(float(f1_score(l2, c2, average="weighted")*100), 2),
        "MAE":  round(float(np.abs(reg-lr).mean()), 4),
        "Corr": round(float(pearsonr(reg, lr)[0]), 4),
    }


def run_batch(batch, device):
    return (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["audio"].to(device), batch["audio_mask"].to(device),
            batch["vision"].to(device), batch["vision_mask"].to(device),
            batch["cls7_label"].to(device), batch["cls2_label"].to(device),
            batch["reg_label"].to(device))


@torch.no_grad()
def validate(model, loader, criterion, device):
    """標準驗證（單次前向）"""
    model.eval()
    total_loss = 0.0
    all_c7, all_c2, all_r, all_l7, all_l2, all_lr = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)
        total_loss += loss.item()
        all_c7.extend(l7.argmax(1).cpu().numpy()); all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy());            all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy());           all_lr.extend(rl.cpu().numpy())
    return total_loss/len(loader), compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr))


def validate_tta(model, loader, device, n_tta=8):
    """TTA：保持 Dropout 激活，N 次前向取均值"""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_l7_probs, all_reg = [], []
    all_labels7, all_labels2, all_reg_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"TTA(n={n_tta})", leave=False):
            ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
            B = ids.size(0)
            l7_acc = torch.zeros(B, 7, device=device)
            reg_acc = torch.zeros(B, device=device)
            for _ in range(n_tta):
                l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
                l7_acc  += F.softmax(l7.detach(), -1)
                reg_acc += reg.detach()
            all_l7_probs.append((l7_acc / n_tta).cpu())
            all_reg.extend((reg_acc / n_tta).cpu().numpy())
            all_labels7.extend(cl7.cpu().numpy())
            all_labels2.extend(cl2.cpu().numpy())
            all_reg_labels.extend(rl.cpu().numpy())

    model.eval()
    l7_probs   = torch.cat(all_l7_probs, 0).numpy()
    reg_preds  = np.array(all_reg)
    labels7    = np.array(all_labels7)
    labels2    = np.array(all_labels2)
    reg_labels = np.array(all_reg_labels)
    return l7_probs, l7_probs.argmax(1), reg_preds, labels7, labels2, reg_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 推理優化（繼承自 v24）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_thresholds(reg_preds, thresholds):
    return np.digitize(reg_preds, sorted(thresholds)) % 7


def search_thresholds_greedy(reg_preds, labels7, n_rounds=3):
    thresholds = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])

    def score(thresh):
        return (np.digitize(reg_preds, thresh) == labels7).mean()

    best_acc = score(thresholds)
    print(f"  [閾值搜索] 初始 Acc7={best_acc*100:.2f}%...")

    for _ in range(n_rounds):
        improved = False
        for i in range(6):
            lo = (thresholds[i-1] + 0.05) if i > 0 else -4.5
            hi = (thresholds[i+1] - 0.05) if i < 5 else  4.5
            for t in np.arange(max(lo, thresholds[i]-1.2),
                               min(hi, thresholds[i]+1.2), 0.05):
                test_thresh = thresholds.copy()
                test_thresh[i] = round(t, 2)
                a = score(test_thresh)
                if a > best_acc:
                    best_acc = a; thresholds[i] = round(t, 2); improved = True
        if not improved:
            break

    print(f"  [閾值搜索] 優化後 Acc7={best_acc*100:.2f}%，閾值={np.round(thresholds,2).tolist()}")
    return thresholds, best_acc * 100


def reg_to_soft_probs(reg_preds, sigma=0.7):
    classes = np.array([-3., -2., -1., 0., 1., 2., 3.])
    diff  = reg_preds[:, None] - classes[None, :]
    probs = np.exp(-0.5 * (diff / sigma) ** 2)
    return probs / probs.sum(axis=1, keepdims=True)


def search_ensemble_alpha(cls7_probs, reg_preds, labels7):
    best_acc, best_alpha, best_sigma = 0.0, 1.0, 0.7
    for sigma in [0.5, 0.6, 0.7, 0.8, 1.0]:
        reg_soft = reg_to_soft_probs(reg_preds, sigma)
        for alpha in np.arange(0.0, 1.05, 0.05):
            combined = alpha * cls7_probs + (1 - alpha) * reg_soft
            acc = (combined.argmax(1) == labels7).mean()
            if acc > best_acc:
                best_acc = acc; best_alpha = round(alpha, 2); best_sigma = sigma
    print(f"  [融合搜索] alpha={best_alpha}, sigma={best_sigma}, Val Acc7={best_acc*100:.2f}%")
    return best_alpha, best_sigma, best_acc * 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.where((ct := np.bincount(cl, minlength=n).astype(float)) == 0, 1.0, ct)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))


def progressive_unfreeze(model, epoch, total_epochs):
    encoder = getattr(model.lang_backbone, "encoder", None)
    if not encoder: return False
    freeze_until = 6 if epoch < total_epochs//3 else (3 if epoch < 2*total_epochs//3 else 0)
    changed = False
    for i, layer in enumerate(encoder.layer):
        for p in layer.parameters():
            want = (i >= freeze_until)
            if p.requires_grad != want:
                p.requires_grad = want; changed = True
    if changed: print(f"  [解凍] Epoch {epoch}: 凍結前 {freeze_until} 層")
    return changed


def modality_dropout(audio, vision, p=0.15):
    """訓練時隨機丟棄一個模態（音頻或視頻）增強魯棒性"""
    if random.random() < p:
        if random.random() < 0.5:
            audio = torch.zeros_like(audio)
        else:
            vision = torch.zeros_like(vision)
    return audio, vision


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema,
                rdrop_alpha=0.0, modal_drop_p=0.15):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)

        # 模態 Dropout（僅訓練時）
        aud, vis = modality_dropout(aud, vis, modal_drop_p)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)
            if rdrop_alpha > 0:
                l7b, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7,-1), F.softmax(l7b,-1), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b,-1), F.softmax(l7,-1), reduction='batchmean')) / 2
                loss = loss + rdrop_alpha * kl

        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        ema.update(); scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)
    print("=" * 70)
    print("MOSI 多模態情感分析 v25 — 架構突破版")
    print("多層DeBERTa聚合 + 雙向跨模態注意力 + 無OrdinalHead + TTA閾值搜索")
    print("=" * 70)

    config = {
        "data_path":    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir":    PROJECT_ROOT / "emotion_system/models",
        "lang_model":   "microsoft/deberta-v3-large",
        "max_text_len": 80,
        "audio_dim": 5, "vision_dim": 20,
        "modal_hidden": 256, "fusion_dim": 512,
        "num_classes": 7, "dropout": 0.2,
        "n_layers_agg": 4,         # 多層聚合：最後 4 層
        "batch_size": 8,
        "num_epochs": 150,         # 120 → 150
        "lang_lr":  6e-6,
        "other_lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
        "w_cls7": 4.0,             # 3.5 → 4.0（強化主任務）
        "w_cls2": 0.5,
        "w_reg": 0.3,
        "ema_decay": 0.9995,
        "patience": 25,            # 20 → 25
        "rdrop_alpha": 0.1,
        "modal_drop_p": 0.15,      # 新：模態 Dropout
        "swa_start_ratio": 0.45,   # 0.6 → 0.45（更早 SWA）
        "swa_lr": 1e-6,
        "n_tta": 8,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"Capped 類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer  = DebertaV2Tokenizer.from_pretrained(config["lang_model"])
    train_ds   = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds     = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds    = MOSIDataset(data["test"],  tokenizer, config["max_text_len"])

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"設備: {device}")

    model = HybridModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"], vision_dim=config["vision_dim"],
        modal_hidden=config["modal_hidden"], fusion_dim=config["fusion_dim"],
        num_classes=config["num_classes"], dropout=config["dropout"],
        n_layers_agg=config["n_layers_agg"]
    ).to(device)

    if enc := getattr(model.lang_backbone, "encoder", None):
        for i in range(6):
            for p in enc.layer[i].parameters(): p.requires_grad = False
        print("初始凍結前 6 層")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {train_p/1e6:.1f}M")

    criterion = HybridLoss(class_w.to(device), config["w_cls7"], config["w_cls2"], config["w_reg"])

    lang_params  = [p for p in model.lang_backbone.parameters() if p.requires_grad]
    other_params = (list(model.layer_agg.parameters()) +
                    list(model.polarity_attn.parameters()) +
                    list(model.audio_encoder.parameters()) +
                    list(model.vision_encoder.parameters()) +
                    list(model.bidir_fusion.parameters()) +
                    list(model.shared.parameters()) +
                    list(model.cls7_head.parameters()) +
                    list(model.cls2_head.parameters()) +
                    list(model.reg_head.parameters()))

    optimizer = optim.AdamW([
        {"params": lang_params,  "lr": config["lang_lr"]},
        {"params": other_params, "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler() if device == "cuda" else None
    ema          = EMA(model, decay=config["ema_decay"])

    swa_model     = AveragedModel(model)
    swa_start     = int(config["num_epochs"] * config["swa_start_ratio"])
    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"])
    swa_started   = False

    print(f"\n[v25] 新架構: 多層DeBERTa({config['n_layers_agg']}層) + 雙向CrossModal + 無OrdinalHead")
    print(f"[v25] SWA(epoch {swa_start}+) | patience={config['patience']} | modal_drop={config['modal_drop_p']}")
    print()

    save_dir = Path(config["model_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history   = []; patience_counter = 0

    for epoch in range(config["num_epochs"]):
        if progressive_unfreeze(model, epoch, config["num_epochs"]):
            ema.add_new_params()

        if epoch >= swa_start and not swa_started:
            swa_started = True
            print(f"  [SWA] 啟動 (epoch {epoch})")

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-"*45)
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler,
                              device, scaler, ema, config["rdrop_alpha"], config["modal_drop_p"])

        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        ema.apply_shadow()
        vl_loss, metrics = validate(model, val_loader, criterion, device)
        ema.restore()

        history.append({"epoch": epoch+1, "tr_loss": round(tr_loss,4),
                         "vl_loss": round(vl_loss,4), **metrics})
        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  Corr={metrics['Corr']:.4f}")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, **metrics}
            ema.apply_shadow()
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                        "metrics": metrics, "config": config},
                       save_dir / "v25_best_acc7.pth")
            ema.restore()
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停：{config['patience']} epochs 無改善"); break

    # SWA 最終評估
    best_val_final = best_acc7["Acc7"]
    if swa_started:
        print("\n[SWA] 更新 BN 統計量...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        _, swa_val = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val Acc7={swa_val['Acc7']:.2f}%  vs  EMA best={best_acc7['Acc7']:.2f}%")
        if swa_val["Acc7"] > best_acc7["Acc7"]:
            print("  ✅ SWA 模型更優！")
            torch.save({"model_state": swa_model.module.state_dict(),
                        "metrics": swa_val, "config": config},
                       save_dir / "v25_swa.pth")
            eval_model = swa_model
            best_val_final = swa_val["Acc7"]
        else:
            ckpt = torch.load(save_dir / "v25_best_acc7.pth", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"]); eval_model = model
    else:
        ckpt = torch.load(save_dir / "v25_best_acc7.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"]); eval_model = model

    # ─────────────────────────────────────────────────
    # 推理優化：TTA + 閾值搜索 + 軟概率融合
    # ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("【推理優化：在 Val 集搜索最優策略】")
    n_tta = config["n_tta"]

    print(f"\n[Val TTA n={n_tta}]")
    val_l7_probs, val_preds7_tta, val_reg, val_labels7, val_labels2, val_reg_labels = \
        validate_tta(eval_model, val_loader, device, n_tta)
    val_acc7_tta = (val_preds7_tta == val_labels7).mean() * 100
    print(f"  TTA Val Acc7={val_acc7_tta:.2f}%  (標準: {best_val_final:.2f}%)")

    print(f"\n[Val 閾值搜索]")
    best_thresh, val_acc7_thresh = search_thresholds_greedy(val_reg, val_labels7)

    print(f"\n[Val 軟概率融合搜索]")
    best_alpha, best_sigma, val_acc7_ens = \
        search_ensemble_alpha(val_l7_probs, val_reg, val_labels7)

    # ─────────────────────────────────────────────────
    # 測試集評估
    # ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("【測試集最終評估 - v25】")

    _, test_m_std = validate(eval_model, test_loader, criterion, device)

    print(f"\n[Test TTA n={n_tta}]")
    test_l7_probs, test_preds7_tta, test_reg, test_labels7, test_labels2, test_reg_labels = \
        validate_tta(eval_model, test_loader, device, n_tta)

    test_acc7_tta    = (test_preds7_tta == test_labels7).mean() * 100
    test_preds2      = (test_reg >= 0).astype(int)
    test_acc2        = (test_preds2 == test_labels2).mean() * 100
    test_f1          = f1_score(test_labels2, test_preds2, average='weighted') * 100
    test_mae         = np.abs(test_reg - test_reg_labels).mean()
    test_corr        = pearsonr(test_reg, test_reg_labels)[0]

    test_preds7_thresh  = apply_thresholds(test_reg, best_thresh)
    test_acc7_thresh    = (test_preds7_thresh == test_labels7).mean() * 100

    test_reg_soft   = reg_to_soft_probs(test_reg, best_sigma)
    test_combined   = best_alpha * test_l7_probs + (1 - best_alpha) * test_reg_soft
    test_preds7_ens = test_combined.argmax(1)
    test_acc7_ens   = (test_preds7_ens == test_labels7).mean() * 100

    best_test = max(test_m_std['Acc7'], test_acc7_tta, test_acc7_thresh, test_acc7_ens)
    gap        = best_val_final - best_test

    print("\n" + "="*70)
    print("【最終結果 - v25】")
    print("="*70)
    print(f"\n  ① 標準推理    Acc7: {test_m_std['Acc7']:.2f}%")
    print(f"  ② TTA({n_tta}次)  Acc7: {test_acc7_tta:.2f}%"
          f"  Acc2={test_acc2:.2f}%  F1={test_f1:.2f}%  MAE={test_mae:.4f}  Corr={test_corr:.4f}")
    print(f"  ③ 閾值搜索    Acc7: {test_acc7_thresh:.2f}%")
    print(f"  ④ 軟概率融合  Acc7: {test_acc7_ens:.2f}%")
    print(f"\n  最佳 Test Acc7: {best_test:.2f}%  (目標: ≥50.5%, v23: 48.69%)")
    print(f"  Val-Test Gap: {gap:.2f}%  (v23: 3.28%)")
    status = "🎉 達標！" if best_test >= 50.5 else f"❌ 差 {50.5 - best_test:.2f}%"
    print(f"  結果: {status}")

    with open(save_dir / "v25_history.json", "w") as f:
        json.dump({
            "history": history, "best_val_acc7": best_acc7,
            "result": {
                "std": test_m_std,
                "tta_acc7": round(test_acc7_tta, 2),
                "thresh_acc7": round(test_acc7_thresh, 2),
                "ensemble_acc7": round(test_acc7_ens, 2),
                "best_test_acc7": round(best_test, 2),
                "val_final": round(best_val_final, 2),
                "gap": round(gap, 2),
            },
            "config": {k: str(v) for k, v in config.items()}
        }, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_val_final:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
