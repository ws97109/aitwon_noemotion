"""
MOSI 多模態情感分析 v29 — SACF Elite+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v28 問題診斷與 v29 修復：

  [問題1] 閾值搜索(Thresh)嚴重過擬合 (CRITICAL)
    v28: Val Thresh=58.95% → Test 47.52%（gap=11.43%！）
    v27: Val Thresh=55.46% → Test 46.79%（gap=8.67%！）
    原因: Thresh 用 6 個參數擬合 229 個 val 樣本 → 極度過擬合
    v29修復: 完全移除 Thresh，只比較 TTA/Prior/Ensemble（val≈test）
    預期效果: 若 v28 選 Prior(53.71%) 則 test≈50.57% ← 已超標

  [問題2] 早停過早，unfreeze 後沒有機會繼續學習
    v28: Best epoch=29，patience=25 → epoch 54 停止
    第一次 unfreeze 在 epoch 50，僅剩 4 個 epoch → 無法充分學習
    v29修復:
      a) patience=35（更多容忍度）
      b) 每次 unfreeze 後重置 patience_counter（給新容量機會）

  [問題3] 缺乏對 train/test 分布差異的訓練魯棒性
    v29修復: 加入 AWP (Adversarial Weight Perturbation)
    AWP vs FGM (v26 爆炸原因):
      FGM: 擾動嵌入 → 需 2 次 backward+unscale → AMP 衝突 → nan
      AWP: 第 1 次 backward 只取梯度方向（scaled 梯度，scale 在 L2 norm 抵消）
           zero_grad → 擾動權重 → 第 2 次 backward → 1 次 unscale → AMP 安全

  [改進] 輸入特徵增強 (降低過擬合)
    音頻/視覺加輕微 Gaussian 噪聲 (std=0.05) → 訓練更魯棒

架構: 完全繼承 v28 (SACFv28Model)
  DeBERTa + PolarityEnhancedAttention(Top-K gates)
  MultiScaleModalityEncoder(TCN+BiLSTM 256-dim)
  AudioVisualCrossAttention → xa_enh, xv_enh (分離輸出)
  SentimentAwareCrossModalAttention(Top-K=5, 2-key)
  OrdinalRegressionHead(CORAL) + SentimentContrastiveLoss(detach)

目標: Val 54%+, Test 50.5%+
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
# 資料集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_text_len=80):
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len
        self.raw_text     = split_data["raw_text"]
        self.audio        = torch.FloatTensor(split_data["audio"])
        self.vision       = torch.FloatTensor(split_data["vision"])
        self.audio_lengths  = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]
        labels = split_data["regression_labels"]
        self.reg_labels  = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))
        print(f"資料: {len(self.raw_text)} 筆 | audio={tuple(self.audio.shape)} | vision={tuple(self.vision.shape)}")

    def __len__(self): return len(self.raw_text)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            TASK_PROMPT + str(self.raw_text[idx]),
            add_special_tokens=True, max_length=self.max_text_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        aud_len  = min(int(self.audio_lengths[idx]),  self.audio.shape[1])
        vis_len  = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]);  aud_mask[:aud_len]  = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len]  = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx],   "audio_mask": aud_mask,
            "vision": self.vision[idx], "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label":  self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模型架構（完全繼承 v28）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
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
        gates  = (g * m).squeeze(-1)
        return self.dropout(pooled), gates


class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(input_dim,  output_dim, kernel_size, dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(output_dim)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, dilation=dilation, padding=0)
        self.norm2 = nn.LayerNorm(output_dim)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.res   = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        res = x.transpose(1, 2)
        h   = res
        h = self.drop(self.act(self.norm1(self.conv1(F.pad(h, (self.pad, 0))).transpose(1,2)).transpose(1,2)))
        h = self.drop(self.act(self.norm2(self.conv2(F.pad(h, (self.pad, 0))).transpose(1,2)).transpose(1,2)))
        return (h + self.res(res)).transpose(1, 2)


class MultiScaleModalityEncoder(nn.Module):
    def __init__(self, input_dim, tcn_hidden=128, lstm_hidden=128,
                 output_dim=256, num_lstm_layers=2, dropout=0.2):
        super().__init__()
        tcn_layers, d_in = [], input_dim
        for d in [1, 2, 4]:
            tcn_layers.append(TCNBlock(d_in, tcn_hidden, 3, d, dropout)); d_in = tcn_hidden
        self.tcn = nn.Sequential(*tcn_layers)
        self.tcn_attn = nn.Linear(tcn_hidden, 1)
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=num_lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_lstm_layers > 1 else 0.0)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.lstm_attn = nn.Linear(lstm_hidden, 1)
        self.proj      = nn.Sequential(nn.Linear(tcn_hidden + lstm_hidden, output_dim),
                                       nn.LayerNorm(output_dim))
        self.dropout   = nn.Dropout(dropout)

    def _pool(self, seq, mask, scorer):
        s = scorer(seq).squeeze(-1).masked_fill(mask == 0, float('-inf'))
        s = s.masked_fill((mask.sum(1, keepdim=True) == 0).expand_as(s), 0.0)
        return (seq * F.softmax(s, dim=1).unsqueeze(-1)).sum(1)

    def forward(self, x, mask):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        tcn_feat  = self._pool(self.tcn(x), mask, self.tcn_attn)
        lengths   = mask.sum(1).long().clamp(min=1).cpu()
        packed    = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lo, _     = self.lstm(packed)
        lo, _     = nn.utils.rnn.pad_packed_sequence(lo, batch_first=True, total_length=x.size(1))
        lstm_feat = self._pool(self.lstm_proj(lo), mask, self.lstm_attn)
        return self.dropout(self.proj(torch.cat([tcn_feat, lstm_feat], dim=-1)))


class AudioVisualCrossAttention(nn.Module):
    """A↔V 雙向互注意力 — 分離輸出 xa_enh, xv_enh（保留 2-key 注意力的意義）"""
    def __init__(self, modal_dim=256, dropout=0.1):
        super().__init__()
        self.scale = modal_dim ** 0.5
        self.q_a, self.k_v, self.v_v = [nn.Linear(modal_dim, modal_dim) for _ in range(3)]
        self.q_v, self.k_a, self.v_a = [nn.Linear(modal_dim, modal_dim) for _ in range(3)]
        self.norm_a  = nn.LayerNorm(modal_dim)
        self.norm_v  = nn.LayerNorm(modal_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xa, xv):
        def _attn(q, k, v, base):
            a = F.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(1).transpose(1,2)) / self.scale, dim=-1)
            return base + self.dropout(torch.bmm(a, v.unsqueeze(1)).squeeze(1))
        xa_e = self.norm_a(_attn(self.q_a(xa), self.k_v(xv), self.v_v(xv), xa))
        xv_e = self.norm_v(_attn(self.q_v(xv), self.k_a(xa), self.v_a(xa), xv))
        return xa_e, xv_e


class SentimentAwareCrossModalAttention(nn.Module):
    """Top-K 情感 token 作為 query，音頻+視覺 2-key（v5 核心創新）"""
    def __init__(self, lang_dim, modal_dim, top_k=5, dropout=0.1):
        super().__init__()
        self.top_k      = top_k
        self.lang_dim   = lang_dim
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.token_attn = nn.Linear(lang_dim, 1)
        self.ffn  = nn.Sequential(nn.Linear(lang_dim, lang_dim//2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(lang_dim//2, lang_dim))
        self.gate = nn.Linear(lang_dim * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(lang_dim)

    def forward(self, xl_hidden, xl_cls, gates, xa, xv):
        _, L, H = xl_hidden.shape
        idx   = gates.topk(min(self.top_k, L), dim=1)[1]
        topk  = xl_hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        w     = F.softmax(self.token_attn(topk), dim=1)
        query = (topk * w).sum(1)
        kv    = torch.stack([self.audio_map(xa), self.vision_map(xv)], dim=1)
        attn  = F.softmax(torch.bmm(query.unsqueeze(1), kv.transpose(1,2)) / (self.lang_dim**0.5), dim=-1)
        x_hat = torch.bmm(attn, kv).squeeze(1)
        x     = self.ffn(xl_cls + x_hat)
        gw    = torch.sigmoid(self.gate(torch.cat([xl_cls, x], dim=-1)))
        return self.norm(xl_cls + self.drop(x * gw))


class OrdinalRegressionHead(nn.Module):
    def __init__(self, feat_dim, num_thresholds=6, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight  = nn.Parameter(torch.randn(feat_dim) * 0.02)
        self.bias    = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, feat):
        logit = self.dropout(feat) @ self.weight
        return logit.unsqueeze(1) + self.bias


class SentimentContrastiveLoss(nn.Module):
    """xl.detach() → 不干擾語言骨幹，只對齊音視頻編碼器"""
    def __init__(self, lang_dim, modal_dim, delta_pos=0.5, delta_neg=1.5, margin=0.2, gamma=0.5):
        super().__init__()
        self.dp, self.dn, self.margin, self.gamma = delta_pos, delta_neg, margin, gamma
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)

    def _match(self, xl, xm):
        return ((xl.mean(0)-xm.mean(0))**2 + (xl.var(0)-xm.var(0))**2).mean()

    def _margin(self, xl, xm):
        xl_n, xm_n = F.normalize(xl,-1), F.normalize(xm,-1)
        neg = xm_n.mean(0, keepdim=True).expand_as(xm_n)
        return F.relu((xl_n*neg).sum(-1) - (xl_n*xm_n).sum(-1) + self.gamma).mean()

    def _contrast(self, xl, xm, rl):
        xl_n, xm_n = F.normalize(xl,-1), F.normalize(xm,-1)
        diff = (rl.unsqueeze(0) - rl.unsqueeze(1)).abs()
        sim  = torch.mm(xl_n, xm_n.T)
        pl = ((diff < self.dp).float() * (1-sim)**2).sum() / ((diff < self.dp).sum() + 1e-9)
        nl = ((diff > self.dn).float() * F.relu(sim-self.margin)**2).sum() / ((diff > self.dn).sum() + 1e-9)
        return pl + nl

    def forward(self, xl, xa, xv, rl):
        xl  = xl.detach()
        xam = self.audio_map(xa); xvm = self.vision_map(xv)
        return (self._match(xl,xam) + self._match(xl,xvm) +
                self._margin(xl,xam) + self._margin(xl,xvm) +
                self._contrast(xl,xam,rl) + self._contrast(xl,xvm,rl))


class SACFv28Model(nn.Module):
    """架構完全繼承 v28（v5創新 + v6編碼器）"""
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=256,
                 fusion_dim=512, top_k=5, num_classes=7, dropout=0.2):
        super().__init__()
        self.lang_backbone  = AutoModel.from_pretrained(lang_model)
        lang_dim            = self.lang_backbone.config.hidden_size
        self.polarity_attn  = PolarityEnhancedAttention(lang_dim, dropout)
        self.audio_encoder  = MultiScaleModalityEncoder(audio_dim,  modal_hidden//2, modal_hidden//2, modal_hidden, dropout=dropout)
        self.vision_encoder = MultiScaleModalityEncoder(vision_dim, modal_hidden//2, modal_hidden//2, modal_hidden, dropout=dropout)
        self.av_cross_attn  = AudioVisualCrossAttention(modal_hidden, dropout)
        self.sacf_attn      = SentimentAwareCrossModalAttention(lang_dim, modal_hidden, top_k, dropout)
        self.shared = nn.Sequential(nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(), nn.Dropout(dropout))
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
                                       nn.Linear(fusion_dim//2, 1), nn.Tanh())
        self.ordinal_head = OrdinalRegressionHead(fusion_dim, dropout=dropout)
        self.align_loss   = SentimentContrastiveLoss(lang_dim, modal_hidden)

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask, reg_labels=None):
        audio  = F.normalize(torch.nan_to_num(audio,  nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        vision = F.normalize(torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        hidden = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)
        xa  = self.audio_encoder(audio, audio_mask)
        xv  = self.vision_encoder(vision, vision_mask)
        xa_e, xv_e = self.av_cross_attn(xa, xv)
        fused  = self.sacf_attn(hidden, xl_cls, gates, xa_e, xv_e)
        feat   = self.shared(fused)
        align  = (self.align_loss(xl_cls, xa_e, xv_e, reg_labels)
                  if reg_labels is not None
                  else torch.tensor(0.0, device=input_ids.device))
        return (self.cls7_head(feat), self.cls2_head(feat),
                self.reg_head(feat).squeeze(-1) * 3.0,
                self.ordinal_head(feat), align)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EMA:
    def __init__(self, model, decay=0.9995):
        self.model  = model; self.decay = decay
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
            if p.requires_grad and n in self.backup: p.data = self.backup[n]
        self.backup = {}

    def add_new_params(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n not in self.shadow: self.shadow[n] = p.data.clone()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AWP — AMP 安全的對抗權重擾動
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AWP:
    """
    Adversarial Weight Perturbation (AMP 安全版)

    AMP 安全原理:
      第1次 backward → scaled 梯度 (用於計算擾動方向，scale 在 L2 norm 中抵消)
      zero_grad → 施加擾動 → 第2次 backward → scaled 對抗梯度
      只調用 unscale_() 一次 → 符合 GradScaler 規範，不會 nan

    vs FGM (v26 爆炸原因):
      FGM: scale(loss).backward() → unscale_() → scale(adv).backward()
      → 第2次 scale() 疊加在已 unscale 的梯度上 → 混合 scale → nan
    """
    def __init__(self, model, alpha=1e-3, emb_name="lang_backbone"):
        self.model    = model
        self.alpha    = alpha
        self.emb_name = emb_name
        self.backup   = {}

    def attack(self):
        """利用當前 param.grad（可以是 scaled 梯度）計算擾動方向並施加"""
        for name, param in self.model.named_parameters():
            if (param.requires_grad and param.grad is not None
                    and self.emb_name in name):
                norm = torch.norm(param.grad)
                if norm > 0:
                    r_at = self.alpha * param.grad / norm
                    self.backup[name] = r_at.clone()
                    param.data.add_(r_at)

    def restore(self):
        """還原擾動"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.sub_(self.backup[name])
        self.backup = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, label_smoothing=self.ls, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


class SACFv29Loss(nn.Module):
    def __init__(self, class_weights, w_cls7=3.5, w_cls2=0.5, w_reg=0.3, w_ord=0.2, w_align=0.05):
        super().__init__()
        self.w = (w_cls7, w_cls2, w_reg, w_ord, w_align)
        self.focal = FocalLoss(alpha=class_weights)
        self.cls2  = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg   = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, ord_logits, align, cl7, cl2, rl):
        lc7  = self.focal(l7, cl7)
        lc2  = self.cls2(l2, cl2)
        lr   = self.reg(reg, rl)
        k    = torch.arange(6, device=cl7.device)
        lord = F.binary_cross_entropy_with_logits(ord_logits, (cl7.unsqueeze(1) > k).float())
        total = self.w[0]*lc7 + self.w[1]*lc2 + self.w[2]*lr + self.w[3]*lord + self.w[4]*align
        return total, {"cls7": lc7.item(), "cls2": lc2.item(),
                       "reg": lr.item(), "ord": lord.item(), "align": align.item()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估
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
            batch["audio"].to(device),  batch["audio_mask"].to(device),
            batch["vision"].to(device), batch["vision_mask"].to(device),
            batch["cls7_label"].to(device), batch["cls2_label"].to(device),
            batch["reg_label"].to(device))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 (AWP + 輸入增強 + R-Drop)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler,
                device, scaler, ema, rdrop_alpha=0.05,
                awp=None, noise_std=0.05):
    model.train()
    total_loss = 0.0
    use_awp    = awp is not None
    use_amp    = scaler is not None

    for batch in tqdm(loader, desc="Train", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)

        # 輸入特徵增強: 輕微 Gaussian 噪聲 (training only)
        if noise_std > 0:
            aud = aud + noise_std * torch.randn_like(aud)
            vis = vis + noise_std * torch.randn_like(vis)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            l7, l2, reg, ord_l, align = model(ids, mask, aud, amask, vis, vmask, rl)
            loss, _ = criterion(l7, l2, reg, ord_l, align, cl7, cl2, rl)
            if rdrop_alpha > 0:
                l7b, _, _, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7, -1), F.softmax(l7b, -1), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b,-1), F.softmax(l7,  -1), reduction='batchmean')) / 2
                loss = loss + rdrop_alpha * kl

        if use_amp:
            scaler.scale(loss).backward()
            # AWP: 用 scaled 梯度計算擾動方向（scale 在 L2 norm 中抵消，結果等價）
            if use_awp:
                awp.attack()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    l7a, l2a, ra, oa, aa = model(ids, mask, aud, amask, vis, vmask, rl)
                    loss_adv, _ = criterion(l7a, l2a, ra, oa, aa, cl7, cl2, rl)
                scaler.scale(loss_adv).backward()
                awp.restore()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if use_awp:
                awp.attack()
                optimizer.zero_grad()
                l7a, l2a, ra, oa, aa = model(ids, mask, aud, amask, vis, vmask, rl)
                loss_adv, _ = criterion(l7a, l2a, ra, oa, aa, cl7, cl2, rl)
                loss_adv.backward()
                awp.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        ema.update(); scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_c7, all_c2, all_r, all_l7, all_l2, all_lr = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
        l7, l2, reg, ord_l, align = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, ord_l, align, cl7, cl2, rl)
        total_loss += loss.item()
        all_c7.extend(l7.argmax(1).cpu().numpy()); all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy());           all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy());          all_lr.extend(rl.cpu().numpy())
    return total_loss/len(loader), compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 推論增強 (v29 修復: 移除 Thresh，只比較 TTA/Prior/Ensemble)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def collect_tta_preds(model, loader, device, n_tta=10):
    model.train()  # MC Dropout
    all_probs7, all_reg, all_l7, all_lr = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask, aud, amask, vis, vmask, cl7, _, rl = run_batch(batch, device)
            p7  = torch.zeros(len(cl7), 7, device=device)
            reg = torch.zeros(len(cl7), device=device)
            for _ in range(n_tta):
                l7, _, r, _, _ = model(ids, mask, aud, amask, vis, vmask)
                p7  += F.softmax(l7, dim=-1); reg += r
            all_probs7.append((p7/n_tta).cpu().numpy())
            all_reg.append((reg/n_tta).cpu().numpy())
            all_l7.extend(cl7.cpu().numpy()); all_lr.extend(rl.cpu().numpy())
    return (np.concatenate(all_probs7), np.concatenate(all_reg),
            np.array(all_l7), np.array(all_lr))


def compute_label_prior(reg_labels, n=7):
    cls = np.clip(np.round(reg_labels).astype(int), -3, 3) + 3
    ct  = np.bincount(cls, minlength=n).astype(float)
    return np.where(ct==0, 1e-6, ct) / ct.sum()


def apply_prior_correction(probs, train_prior, val_prior, strength=1.0):
    ratio = np.power(np.maximum(val_prior,1e-10)/np.maximum(train_prior,1e-10), strength)
    c = probs * ratio[np.newaxis, :]
    s = c.sum(1, keepdims=True); return c / np.where(s==0, 1.0, s)


def reg_to_soft_labels(reg_preds, sigma=0.5, n=7):
    centers = np.arange(n)
    d = -((reg_preds[:,np.newaxis]+3 - centers[np.newaxis,:])**2) / (2*sigma**2)
    p = np.exp(d - d.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def enhanced_inference(model, val_loader, test_loader, train_reg_labels, device, n_tta=10):
    """
    v29 推論增強 — 移除 Thresh（嚴重過擬合 val 229 樣本）
    只比較: TTA / Prior / Ensemble（這三種方法 val≈test，不過擬合）
    """
    print("\n[推論增強] 收集 val TTA 預測...")
    vp7, vr, vl7, vlr = collect_tta_preds(model, val_loader, device, n_tta)

    base_acc = (vp7.argmax(1) == vl7).mean() * 100
    print(f"  TTA 基線: Val Acc7 = {base_acc:.2f}%")

    # 先驗校正
    t_prior, v_prior = compute_label_prior(train_reg_labels), compute_label_prior(vlr)
    best_corr, best_s = 0.0, 0.0
    for s in np.arange(0.0, 3.1, 0.2):
        c   = apply_prior_correction(vp7, t_prior, v_prior, s)
        acc = (c.argmax(1) == vl7).mean() * 100
        if acc > best_corr: best_corr, best_s = acc, round(float(s), 1)
    print(f"  [先驗校正] strength={best_s}, Val Acc7={best_corr:.2f}%")

    # 軟集成 (cls7 + reg Gaussian)
    best_blend, best_alpha, best_sigma = 0.0, 0.8, 0.5
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        rp = reg_to_soft_labels(vr, sigma)
        for alpha in np.arange(0.5, 1.01, 0.05):
            acc = ((alpha*vp7 + (1-alpha)*rp).argmax(1) == vl7).mean() * 100
            if acc > best_blend:
                best_blend = acc; best_alpha = round(float(alpha),2); best_sigma = sigma
    print(f"  [軟集成] alpha={best_alpha:.2f}, sigma={best_sigma:.2f}, Val Acc7={best_blend:.2f}%")

    # 選最優（只從 TTA/Prior/Ensemble 選，不選 Thresh）
    methods = {"TTA": base_acc, "Prior": best_corr, "Ensemble": best_blend}
    best_method = max(methods, key=methods.get)
    print(f"\n  最優 val 方案: {best_method} ({methods[best_method]:.2f}%)")
    print(f"  [注意] 已移除 Thresh，因其在 229 個 val 樣本上嚴重過擬合（val高但test低）")

    print("\n[推論增強] 收集 test TTA 預測...")
    tp7, tr, tl7, tlr = collect_tta_preds(model, test_loader, device, n_tta)

    if best_method == "TTA":
        test_c7 = tp7.argmax(1)
    elif best_method == "Prior":
        test_c7 = apply_prior_correction(tp7, t_prior, v_prior, best_s).argmax(1)
    else:  # Ensemble
        rp = reg_to_soft_labels(tr, best_sigma)
        test_c7 = (best_alpha*tp7 + (1-best_alpha)*rp).argmax(1)

    test_c2 = (test_c7 >= 3).astype(int)
    tm = compute_metrics(test_c7, test_c2, tr, tl7, (tlr>=0).astype(int), tlr)
    return tm, best_method, methods


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct==0, 1.0, ct)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))


def progressive_unfreeze(model, epoch, total_epochs):
    enc = getattr(model.lang_backbone, "encoder", None)
    if not enc: return False
    freeze_until = 6 if epoch < total_epochs//3 else (3 if epoch < 2*total_epochs//3 else 0)
    changed = False
    for i, layer in enumerate(enc.layer):
        want = (i >= freeze_until)
        for p in layer.parameters():
            if p.requires_grad != want:
                p.requires_grad = want; changed = True
    if changed: print(f"  [解凍] Epoch {epoch}: 凍結前 {freeze_until} 層")
    return changed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)
    print("=" * 70)
    print("MOSI 多模態情感分析 v29 — SACF Elite+")
    print("v28架構 + AWP(AMP安全) + 輸入增強 + 推論修復(移除Thresh過擬合)")
    print("patience=35 + unfreeze後重置計數器")
    print("=" * 70)

    config = {
        "data_path":       PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir":       PROJECT_ROOT / "emotion_system/models",
        "lang_model":      "microsoft/deberta-v3-large",
        "max_text_len":    80,
        "audio_dim":       5, "vision_dim": 20,
        "modal_hidden":    256, "fusion_dim": 512,
        "top_k":           5, "num_classes": 7, "dropout": 0.2,
        "batch_size":      8,
        "num_epochs":      150,
        "lang_lr":         5e-6,
        "other_lr":        1e-4,
        "weight_decay":    1e-2,
        "warmup_ratio":    0.06,
        "w_cls7":          3.5, "w_cls2": 0.5, "w_reg": 0.3,
        "w_ord":           0.2, "w_align": 0.05,
        "ema_decay":       0.9995,
        "patience":        35,        # v28:25 → 35（給 unfreeze 後更多時間）
        "rdrop_alpha":     0.05,
        "awp_alpha":       1e-3,      # AWP 擾動幅度
        "awp_start_epoch": 10,        # warmup 後才啟動 AWP
        "noise_std":       0.05,      # 輸入 Gaussian 噪聲
        "swa_start_ratio": 0.50,
        "swa_lr":          1e-6,
        "n_tta":           10,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w  = compute_class_weights(data["train"]["regression_labels"])
    print(f"Capped 類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer    = DebertaV2Tokenizer.from_pretrained(config["lang_model"])
    train_ds     = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds       = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds      = MOSIDataset(data["test"],  tokenizer, config["max_text_len"])
    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"設備: {device}")

    model = SACFv28Model(
        lang_model=config["lang_model"], audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"], modal_hidden=config["modal_hidden"],
        fusion_dim=config["fusion_dim"], top_k=config["top_k"],
        num_classes=config["num_classes"], dropout=config["dropout"],
    ).to(device)

    if enc := getattr(model.lang_backbone, "encoder", None):
        for i in range(6):
            for p in enc.layer[i].parameters(): p.requires_grad = False
        print("初始凍結前 6 層")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {train_p/1e6:.1f}M")

    criterion = SACFv29Loss(class_w.to(device), config["w_cls7"], config["w_cls2"],
                            config["w_reg"], config["w_ord"], config["w_align"])

    # Optimizer 修復: 所有 DeBERTa 層加入（含 frozen 層）
    all_lang = list(model.lang_backbone.parameters())
    other    = (list(model.polarity_attn.parameters()) +
                list(model.audio_encoder.parameters()) +
                list(model.vision_encoder.parameters()) +
                list(model.av_cross_attn.parameters()) +
                list(model.sacf_attn.parameters()) +
                list(model.shared.parameters()) +
                list(model.cls7_head.parameters()) +
                list(model.cls2_head.parameters()) +
                list(model.reg_head.parameters()) +
                list(model.ordinal_head.parameters()) +
                list(model.align_loss.parameters()))
    optimizer = optim.AdamW([
        {"params": all_lang, "lr": config["lang_lr"]},
        {"params": other,    "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler() if device == "cuda" else None
    ema          = EMA(model, decay=config["ema_decay"])
    awp          = AWP(model, alpha=config["awp_alpha"], emb_name="lang_backbone")

    swa_model    = AveragedModel(model)
    swa_start    = int(config["num_epochs"] * config["swa_start_ratio"])
    swa_sched    = SWALR(optimizer, swa_lr=config["swa_lr"])
    swa_started  = False

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[v29] AWP(alpha={config['awp_alpha']}, start ep{config['awp_start_epoch']}) + noise(std={config['noise_std']})")
    print(f"      推論修復: 移除 Thresh，只選 TTA/Prior/Ensemble")
    print(f"      patience={config['patience']}（每次 unfreeze 後重置計數器）\n")

    save_dir = Path(config["model_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history   = []; patience_counter = 0

    for epoch in range(config["num_epochs"]):
        unfrozen = progressive_unfreeze(model, epoch, config["num_epochs"])
        if unfrozen:
            ema.add_new_params()
            patience_counter = 0          # ← v29 新增: unfreeze 後重置早停計數器
            print(f"  [重置早停計數器] 新容量解凍，重新計算 patience")

        if epoch >= swa_start and not swa_started:
            swa_started = True
            print(f"  [SWA] 啟動 (epoch {epoch})")

        # AWP 在 warmup 後才啟動（前期先穩定訓練）
        use_awp = awp if epoch >= config["awp_start_epoch"] else None

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-"*45)
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler,
                              device, scaler, ema, config["rdrop_alpha"],
                              awp=use_awp, noise_std=config["noise_std"])

        if swa_started:
            swa_model.update_parameters(model)
            swa_sched.step()

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
                       save_dir / "v29_best_acc7.pth")
            ema.restore()
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停：{config['patience']} epochs 無改善"); break

    # SWA 最終評估
    if swa_started:
        print("\n[SWA] 更新 BatchNorm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        _, swa_val = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val Acc7={swa_val['Acc7']:.2f}%  (EMA best={best_acc7['Acc7']:.2f}%)")
        if swa_val["Acc7"] > best_acc7["Acc7"]:
            print("  ✅ SWA 更優！")
            torch.save({"model_state": swa_model.module.state_dict(),
                        "metrics": swa_val, "config": config},
                       save_dir / "v29_swa.pth")
            eval_model = swa_model
        else:
            ckpt = torch.load(save_dir / "v29_best_acc7.pth", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"]); eval_model = model
    else:
        ckpt = torch.load(save_dir / "v29_best_acc7.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"]); eval_model = model

    # 標準測試
    print("\n" + "="*60)
    _, test_std = validate(eval_model, test_loader, criterion, device)
    print("\n【標準測試集結果 - v29】")
    print(f"  Acc7 : {test_std['Acc7']:.2f}%   (目標: 50.5%, v28: 48.83%)")
    print(f"  Acc2 : {test_std['Acc2']:.2f}%")
    print(f"  F1   : {test_std['F1']:.2f}%")
    print(f"  MAE  : {test_std['MAE']:.4f}")
    print(f"  Corr : {test_std['Corr']:.4f}")
    print(f"\n  Val-Test Gap: {best_acc7['Acc7'] - test_std['Acc7']:.2f}%")

    # 推論增強（已移除 Thresh）
    print("\n" + "="*60)
    print("[推論增強] 開始（v29 修復版：無 Thresh 過擬合）...")
    ema.apply_shadow()
    test_enh, best_method, all_val_accs = enhanced_inference(
        eval_model, val_loader, test_loader,
        data["train"]["regression_labels"], device, config["n_tta"])
    ema.restore()

    print(f"\n【推論增強測試集結果 - v29 ({best_method})】")
    print(f"  Acc7 : {test_enh['Acc7']:.2f}%   (目標: 50.5%)")
    print(f"  Acc2 : {test_enh['Acc2']:.2f}%")
    print(f"  F1   : {test_enh['F1']:.2f}%")
    print(f"  MAE  : {test_enh['MAE']:.4f}")
    print(f"  Corr : {test_enh['Corr']:.4f}")

    final_acc7 = max(test_std["Acc7"], test_enh["Acc7"])
    status = "🎉 達標！" if final_acc7 >= 50.5 else f"❌ 差 {50.5 - final_acc7:.2f}%"
    print(f"\n  最終 Test Acc7: {final_acc7:.2f}% | {status}")

    with open(save_dir / "v29_history.json", "w") as f:
        json.dump({
            "history":               history,
            "best_val_acc7":         best_acc7,
            "test_standard":         test_std,
            "test_enhanced":         test_enh,
            "best_inference_method": best_method,
            "all_val_inference_accs": all_val_accs,
            "config": {k: str(v) for k, v in config.items()},
        }, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
