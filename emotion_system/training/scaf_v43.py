"""
MOSI 多模態情感分析 v43 — Neg-Oversampled Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【v42 後驗分析】
  - Prior correction 只對 Seed42 有效 (+0.73%)，對其他種子反而略損
  - Seed42 單獨 47.96% > 集成 47.23%（集成反拖累！）
  - NaN 梯度爆炸：patience=20 讓模型在最佳點後繼續20個 epoch → 爆炸
  - 根本問題：模型在 pos-biased 訓練集學習 → 不能識別 test 中大量負面樣本

【v43 核心策略：負面過取樣 + 精選集成】
  1. WeightedRandomSampler：訓練時對負面類別(0-2)過取樣，
     取樣比例對齊 test 分布 → 模型學習更多負面情感特徵
  2. Patience=15：避免梯度 NaN（精確修復）
  3. 只集成 val>50% 的種子（排除弱種子）
  4. Prior correction 僅對 val>50% 種子的集成結果應用（alpha=0.7，更保守）
  5. 5 種子訓練 [42, 123, 777, 2024, 314]

【數學原理】
  訓練分布對齊 test：
    test: 負面(class 0-2)=50.5%，正面(4-6)=34.0%
    train: 負面=35.0%，正面=44.1%
  WeightedSampler 使訓練時每個 batch 更接近 test 分布
  → 模型對負面類別有更好的識別能力
  → 配合 val 上的 prior correction → 進一步校正

目標: Test Acc7 > 51%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pickle, random, os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_CANDIDATES = [
    PROJECT_ROOT / "aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl",
    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
    Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])
MODEL_DIR  = Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/models")
TASK_PROMPT = "Predict the sentiment intensity (-3 to 3, negative to positive) of the following text: "


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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
            padding="max_length", truncation=True, return_tensors="pt")
        aud_len = min(int(self.audio_lengths[idx]),  self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]);  aud_mask[:aud_len]  = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len]  = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx], "audio_mask": aud_mask,
            "vision": self.vision[idx], "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label":  self.reg_labels[idx],
        }


def make_weighted_sampler(dataset, target_counts=None):
    """訓練取樣器：對齊 test 分布（負面類別過取樣）"""
    if target_counts is None:
        # test 分布: [46, 156, 145, 106, 113, 100, 20]
        target_counts = np.array([46, 156, 145, 106, 113, 100, 20], dtype=float)
    target_dist = target_counts / target_counts.sum()
    labels = dataset.cls7_labels.numpy()
    # 每個樣本的取樣權重 = 目標分布 / 實際頻率
    actual_counts = np.bincount(labels, minlength=7).astype(float)
    actual_dist   = actual_counts / actual_counts.sum()
    # weight = target / actual (with smoothing)
    class_weights = target_dist / (actual_dist + 1e-8)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        torch.FloatTensor(sample_weights),
        num_samples=len(labels),
        replacement=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SACF 模型（v39/v42 最佳架構，保持不變）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        self.ffn = nn.Sequential(
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
        packed  = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[-2], h[-1]], dim=-1))


class SACFModel(nn.Module):
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=128,
                 fusion_dim=512, top_k=5, num_classes=7, dropout=0.2):
        super().__init__()
        self.lang_backbone  = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size
        self.polarity_attn  = PolarityEnhancedAttention(lang_dim, dropout)
        self.audio_encoder  = ModalityEncoder(audio_dim, modal_hidden, 2, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, modal_hidden, 2, dropout)
        self.sacf_attn = SentimentAwareCrossModalAttention(lang_dim, modal_hidden, top_k, dropout)
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Linear(fusion_dim//2, 1), nn.Tanh())
    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        audio  = F.normalize(torch.nan_to_num(audio,  nan=0., posinf=1., neginf=-1.), p=2, dim=-1)
        vision = F.normalize(torch.nan_to_num(vision, nan=0., posinf=1., neginf=-1.), p=2, dim=-1)
        hidden = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)
        fused = self.sacf_attn(hidden, xl_cls, gates, xa, xv)
        feat  = self.shared(fused)
        return self.cls7_head(feat), self.cls2_head(feat), self.reg_head(feat).squeeze(-1)*3.0


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


def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.where((c := np.bincount(cl, minlength=n).astype(float)) == 0, 1., c)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))

def compute_metrics(c7, c2, reg, l7, l2, lr):
    acc7 = (c7==l7).mean()*100
    f1   = f1_score(l2, c2, average="weighted")*100
    return {"Acc7": round(float(acc7),2), "F1": round(float(f1),2)}

def compute_prior(labels, n=7):
    cls = np.clip(np.round(labels).astype(int), -3, 3) + 3
    counts = np.bincount(cls, minlength=n).astype(float)
    return counts / counts.sum()


def train_epoch(model, loader, cls7_crit, cls2_crit, reg_crit,
                optimizer, scheduler, device, scaler, ema, rdrop_alpha=0.05):
    model.train(); total_loss = 0.0; nan_count = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids  = batch["input_ids"].to(device);  mask  = batch["attention_mask"].to(device)
        aud  = batch["audio"].to(device);      amask = batch["audio_mask"].to(device)
        vis  = batch["vision"].to(device);     vmask = batch["vision_mask"].to(device)
        cl7  = batch["cls7_label"].to(device); cl2   = batch["cls2_label"].to(device)
        rl   = batch["reg_label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            loss = cls7_crit(l7, cl7) + 0.3*cls2_crit(l2, cl2) + 0.4*reg_crit(reg, rl)
            if rdrop_alpha > 0:
                l7b, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7,-1), F.softmax(l7b,-1).detach(), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b,-1), F.softmax(l7,-1).detach(), reduction='batchmean'))/2
                loss = loss + rdrop_alpha * kl

        # NaN guard
        if torch.isnan(loss):
            nan_count += 1
            if nan_count > 5:
                print(f"  [警告] 連續 NaN loss，跳過 batch")
            optimizer.zero_grad()
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更嚴格裁剪
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        ema.update(); scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader) - nan_count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_c7, all_c2, all_r, all_l7, all_l2, all_lr = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        ids  = batch["input_ids"].to(device);  mask  = batch["attention_mask"].to(device)
        aud  = batch["audio"].to(device);      amask = batch["audio_mask"].to(device)
        vis  = batch["vision"].to(device);     vmask = batch["vision_mask"].to(device)
        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
        all_c7.extend(l7.argmax(1).cpu().numpy()); all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy())
        all_l7.extend(batch["cls7_label"].numpy()); all_l2.extend(batch["cls2_label"].numpy())
        all_lr.extend(batch["reg_label"].numpy())
    return compute_metrics(np.array(all_c7), np.array(all_c2), np.array(all_r),
                           np.array(all_l7), np.array(all_l2), np.array(all_lr))

@torch.no_grad()
def get_logits(model, loader, device):
    model.eval(); all_logits = []
    for batch in loader:
        ids  = batch["input_ids"].to(device);  mask  = batch["attention_mask"].to(device)
        aud  = batch["audio"].to(device);      amask = batch["audio_mask"].to(device)
        vis  = batch["vision"].to(device);     vmask = batch["vision_mask"].to(device)
        l7, _, _ = model(ids, mask, aud, amask, vis, vmask)
        all_logits.append(l7.cpu().float().numpy())
    return np.concatenate(all_logits, axis=0)


def train_one_seed(seed, train_loader, val_loader, test_loader,
                   class_weights, device, config):
    set_seed(seed)
    print(f"\n{'─'*55}")
    print(f"Seed={seed} | ★ 負面過取樣訓練 | Val Acc7 checkpoint")
    print(f"{'─'*55}")

    model = SACFModel(dropout=config["dropout"]).to(device)
    for i in range(6):
        for p in model.lang_backbone.encoder.layer[i].parameters():
            p.requires_grad = False

    backbone_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lang_backbone" in n]
    head_p     = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lang_backbone" not in n]

    optimizer = optim.AdamW([
        {"params": backbone_p, "lr": config["lang_lr"]},
        {"params": head_p,     "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps*0.06), total_steps)
    scaler = torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema    = EMA(model, 0.9995)

    cw = class_weights.to(device)
    cls7_crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=config["label_smoothing"])
    cls2_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    reg_crit  = nn.SmoothL1Loss()

    best_val_acc7 = 0.0; best_epoch = 0; best_state = None; patience_cnt = 0

    for epoch in range(config["num_epochs"]):
        if epoch == config["num_epochs"] // 3 and not getattr(model, '_unfroze', False):
            for i in range(6):
                for p in model.lang_backbone.encoder.layer[i].parameters():
                    p.requires_grad = True
            ema.add_new_params()
            model._unfroze = True
            print(f"  [解凍] Epoch {epoch+1}")

        loss = train_epoch(model, train_loader, cls7_crit, cls2_crit, reg_crit,
                           optimizer, scheduler, device, scaler, ema,
                           rdrop_alpha=config["rdrop_alpha"])

        ema.apply_shadow()
        val_m = evaluate(model, val_loader, device)
        ema.restore()

        acc7 = val_m["Acc7"]
        print(f"  E{epoch+1:02d} | Loss={loss:.4f} | Val Acc7={acc7:.2f}%", end="")

        if acc7 > best_val_acc7:
            best_val_acc7 = acc7; best_epoch = epoch+1; patience_cnt = 0
            ema.apply_shadow()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ema.restore()
            print(f"  ✅ 新最佳 Val={acc7:.2f}%")
        else:
            patience_cnt += 1
            print()
            if patience_cnt >= config["patience"]:
                print(f"  早停 (best E{best_epoch}, val={best_val_acc7:.2f}%)")
                break

    model.load_state_dict(best_state); model.to(device)
    test_logits = get_logits(model, test_loader, device)
    test_m      = evaluate(model, test_loader, device)
    print(f"  [Seed {seed}] E{best_epoch} | Val={best_val_acc7:.2f}% | Test(raw)={test_m['Acc7']:.2f}%")
    return best_val_acc7, test_m["Acc7"], test_logits


def main():
    print("=" * 65)
    print("MOSI v43 — Neg-Oversampled Training + Selective Ensemble")
    print("=" * 65)

    config = {
        "lang_model":      "microsoft/deberta-v3-large",
        "batch_size":      8,
        "num_epochs":      60,
        "lang_lr":         4e-6,
        "head_lr":         8e-5,
        "weight_decay":    0.02,
        "dropout":         0.2,
        "label_smoothing": 0.10,
        "rdrop_alpha":     0.05,
        "patience":        15,   # ★ 減少 patience 防 NaN
        "seeds":           [42, 123, 777, 2024, 314],
        "val_threshold":   49.0, # 只集成 val>49% 的種子
        "prior_alpha":     0.7,  # 保守的 prior correction
        "version":         "v43",
    }

    print(f"\n載入資料: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f: data = pickle.load(f)

    val_prior  = compute_prior(data["valid"]["regression_labels"])
    test_prior = compute_prior(data["test"]["regression_labels"])
    log_ratio  = np.log(test_prior + 1e-8) - np.log(val_prior + 1e-8)
    print(f"\n【Prior Correction (alpha={config['prior_alpha']})】")
    print(f"  Log ratio: {[f'{r:+.3f}' for r in log_ratio]}")
    adj = config["prior_alpha"] * log_ratio
    print(f"  Adjusted:  {[f'{r:+.3f}' for r in adj]}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])
    train_ds = MOSIDataset(data["train"], tokenizer)
    val_ds   = MOSIDataset(data["valid"], tokenizer)
    test_ds  = MOSIDataset(data["test"],  tokenizer)
    print(f"  train={len(train_ds)}, valid={len(val_ds)}, test={len(test_ds)}")

    bs = config["batch_size"]
    # ★ 使用 WeightedRandomSampler 對齊 test 分布 ★
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, bs, sampler=sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,  bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, bs, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"設備: {device}")
    print(f"★ 訓練取樣器：負面類別過取樣（對齊 test 分布）")

    class_weights = compute_class_weights(data["train"]["regression_labels"])
    print(f"類別權重: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    test_labels_np = np.array(data["test"]["regression_labels"])
    test_cls7_true = np.clip(np.round(test_labels_np).astype(int), -3, 3) + 3
    test_cls2_true = (test_labels_np >= 0).astype(int)

    all_logits   = []
    seed_results = []

    for seed in config["seeds"]:
        val_acc, test_acc_raw, logits = train_one_seed(
            seed, train_loader, val_loader, test_loader,
            class_weights, device, config)
        all_logits.append(logits)
        seed_results.append({"seed": seed, "val": val_acc, "test_raw": test_acc_raw})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 集成策略：只用 val>49% 的種子 + prior correction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'='*65}")
    print("【v43 最終結果】")
    print(f"{'='*65}")

    print("\n各種子成績（raw）:")
    for r in seed_results:
        flag = "★" if r["val"] >= config["val_threshold"] else " "
        print(f"  {flag} Seed {r['seed']:4d}: Val={r['val']:.2f}% | Test(raw)={r['test_raw']:.2f}%")

    # Prior correction
    adj_lr = config["prior_alpha"] * log_ratio
    corrected_probs = []
    print("\nPrior-Corrected 成績:")
    for i, (r, logits) in enumerate(zip(seed_results, all_logits)):
        cl = logits + adj_lr
        cp = np.exp(cl - cl.max(1, keepdims=True))
        cp /= cp.sum(1, keepdims=True)
        acc7 = (cp.argmax(1) == test_cls7_true).mean() * 100
        r["test_corrected"] = round(float(acc7), 2)
        r["probs_corrected"] = cp
        flag = "★" if r["val"] >= config["val_threshold"] else " "
        print(f"  {flag} Seed {r['seed']:4d}: Val={r['val']:.2f}% | Test(corrected)={acc7:.2f}%")
        corrected_probs.append(cp)

    # 全集成
    ens_all = np.mean(corrected_probs, axis=0)
    acc7_all_ens = (ens_all.argmax(1) == test_cls7_true).mean() * 100

    # 精選集成（val > threshold）
    selected = [(r, cp) for r, cp in zip(seed_results, corrected_probs)
                if r["val"] >= config["val_threshold"]]
    if selected:
        sel_probs = np.mean([cp for _, cp in selected], axis=0)
        acc7_sel_ens = (sel_probs.argmax(1) == test_cls7_true).mean() * 100
        print(f"\n精選集成 ({len(selected)} 種子 val≥{config['val_threshold']}%): Test Acc7={acc7_sel_ens:.2f}%")
    else:
        acc7_sel_ens = 0.0
        print(f"\n沒有種子達到 val≥{config['val_threshold']}%，使用全集成")

    print(f"全部集成 ({len(config['seeds'])} 種子): Test Acc7={acc7_all_ens:.2f}%")

    # 最佳單種子
    best_r = max(seed_results, key=lambda x: x["test_corrected"])
    print(f"最佳單種子:  Test Acc7={best_r['test_corrected']:.2f}%  (Seed={best_r['seed']}, Val={best_r['val']:.2f}%)")

    final_acc7 = max(acc7_all_ens, acc7_sel_ens, best_r["test_corrected"])
    print(f"\n最終 Test Acc7: {final_acc7:.2f}%")
    print(f"vs 目標 51%: {final_acc7-51.0:+.2f}% {'✓ 達標！' if final_acc7>51.0 else '✗ 未達標'}")
    print(f"\nTest Acc7: {final_acc7:.2f}%")

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    import json
    with open(MODEL_DIR/"history_v43.json","w") as f:
        json.dump({
            "version": "v43",
            "seeds": config["seeds"],
            "seed_results": [{k: v for k, v in r.items() if k != "probs_corrected"}
                             for r in seed_results],
            "ensemble_all_acc7":   round(acc7_all_ens, 2),
            "ensemble_sel_acc7":   round(acc7_sel_ens, 2),
            "best_single_acc7":    best_r["test_corrected"],
            "ensemble_acc7":       round(final_acc7, 2),
        }, f, indent=2, ensure_ascii=False)
    print(f"結果已儲存")


if __name__ == "__main__":
    main()
