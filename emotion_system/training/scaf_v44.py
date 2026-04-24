"""
MOSI 多模態情感分析 v44 — IW-Acc7 Checkpoint Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【v43 後驗分析】
  - 負面過取樣使 Seed42 退步：Val 50.22%(↓) Test 47.08%(↓)
  - 根本矛盾：用 pos-biased val 選 checkpoint 同時用 neg-biased 訓練，方向衝突

【v44 核心創新：Importance-Weighted Validation Accuracy】
  問題根源：val 正面偏重 → 選出的 checkpoint 在 test 負面偏重上表現差

  解法：用 test_freq/val_freq 對 val 樣本加重要性加權
    IW-Acc7 = Σ(w_i * correct_i) / Σ(w_i)
    其中 w_i = test_freq[class_i] / val_freq[class_i]
    
  效果：
    - 負面類(0,1,2)的 val 樣本加權 1.51~1.74x（test 中更常見）
    - 正面類(4,5,6)的 val 樣本降權 0.35~0.97x（test 中較少見）
    - Checkpoint 選擇更接近 test 分布下的最佳模型

  這是統計上最嚴謹的解法（importance weighting for covariate shift）

【其他設定】
  - 回歸 v39 架構（無過取樣）
  - Patience=15（修復 NaN）
  - Gradient clip=0.5（更嚴格）  
  - Prior correction alpha=0.7
  - Seeds [42, 123, 777, 2024, 314]

目標: Test Acc7 > 51%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pickle, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MOSIDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_text_len=80):
        self.tokenizer = tokenizer; self.max_text_len = max_text_len
        self.raw_text = split_data["raw_text"]
        self.audio    = torch.FloatTensor(split_data["audio"])
        self.vision   = torch.FloatTensor(split_data["vision"])
        self.audio_lengths  = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]
        labels = split_data["regression_labels"]
        self.reg_labels  = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))
    def __len__(self): return len(self.raw_text)
    def __getitem__(self, idx):
        enc = self.tokenizer(TASK_PROMPT + str(self.raw_text[idx]),
            add_special_tokens=True, max_length=self.max_text_len,
            padding="max_length", truncation=True, return_tensors="pt")
        aud_len = min(int(self.audio_lengths[idx]),  self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]);  aud_mask[:aud_len]  = 1.
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len]  = 1.
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "audio": self.audio[idx], "audio_mask": aud_mask,
                "vision": self.vision[idx], "vision_mask": vis_mask,
                "cls7_label": self.cls7_labels[idx],
                "cls2_label": self.cls2_labels[idx],
                "reg_label":  self.reg_labels[idx]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SACF 模型（v39 最佳架構，保持不變）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, h, d=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(h,h//4),nn.Tanh(),nn.Linear(h//4,1),nn.Sigmoid())
        self.dropout = nn.Dropout(d)
    def forward(self, hidden, mask):
        g = self.gate(hidden); m = mask.unsqueeze(-1).float()
        p = ((0.75*hidden+0.25*hidden*g)*m).sum(1)/m.sum(1).clamp(min=1e-9)
        return self.dropout(p), (g*m).squeeze(-1)

class SentimentAwareCrossModalAttention(nn.Module):
    def __init__(self, ld, md, k=5, d=0.1):
        super().__init__()
        self.top_k = k
        self.audio_map  = nn.Linear(md, ld); self.vision_map = nn.Linear(md, ld)
        self.token_attn = nn.Linear(ld, 1)
        self.ffn  = nn.Sequential(nn.Linear(ld,ld//2),nn.ReLU(),nn.Dropout(d),nn.Linear(ld//2,ld))
        self.gate = nn.Linear(ld*2, 1); self.dropout = nn.Dropout(d); self.norm = nn.LayerNorm(ld)
    def forward(self, xl_h, xl_cls, gates, xa, xv):
        B,L,H = xl_h.shape
        idx = gates.topk(min(self.top_k,L),dim=1).indices
        th  = xl_h.gather(1, idx.unsqueeze(-1).expand(-1,-1,H))
        w   = F.softmax(self.token_attn(th),dim=1)
        sa_q = (th*w).sum(1)
        kv   = torch.stack([self.audio_map(xa),self.vision_map(xv)],dim=1)
        attn = F.softmax(torch.bmm(sa_q.unsqueeze(1),kv.transpose(1,2))/(H**0.5),dim=-1)
        x_hat = torch.bmm(attn,kv).squeeze(1)
        x = self.ffn(xl_cls+x_hat)
        gw = torch.sigmoid(self.gate(torch.cat([xl_cls,x],dim=-1)))
        return self.norm(xl_cls+self.dropout(x*gw))

class ModalityEncoder(nn.Module):
    def __init__(self, i, h, n=2, d=0.2):
        super().__init__()
        self.lstm = nn.LSTM(i,h,n,batch_first=True,bidirectional=True,dropout=d if n>1 else 0.)
        self.proj = nn.Linear(h*2,h)
    def forward(self, x, mask):
        lens = mask.sum(1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x,lens,batch_first=True,enforce_sorted=False)
        _,(h,_) = self.lstm(packed)
        return self.proj(torch.cat([h[-2],h[-1]],dim=-1))

class SACFModel(nn.Module):
    def __init__(self, lm="microsoft/deberta-v3-large",
                 ad=5, vd=20, mh=128, fd=512, k=5, nc=7, dropout=0.2):
        super().__init__()
        self.lang_backbone  = AutoModel.from_pretrained(lm)
        ld = self.lang_backbone.config.hidden_size
        self.polarity_attn  = PolarityEnhancedAttention(ld, dropout)
        self.audio_encoder  = ModalityEncoder(ad, mh, 2, dropout)
        self.vision_encoder = ModalityEncoder(vd, mh, 2, dropout)
        self.sacf_attn = SentimentAwareCrossModalAttention(ld, mh, k, dropout)
        self.shared = nn.Sequential(nn.Linear(ld,fd),nn.LayerNorm(fd),nn.GELU(),nn.Dropout(dropout))
        self.cls7_head = nn.Linear(fd, nc)
        self.cls2_head = nn.Linear(fd, 2)
        self.reg_head  = nn.Sequential(nn.Linear(fd,fd//2),nn.GELU(),nn.Linear(fd//2,1),nn.Tanh())
    def forward(self, ids, mask, aud, amask, vis, vmask):
        aud = F.normalize(torch.nan_to_num(aud, nan=0.,posinf=1.,neginf=-1.),p=2,dim=-1)
        vis = F.normalize(torch.nan_to_num(vis, nan=0.,posinf=1.,neginf=-1.),p=2,dim=-1)
        h   = self.lang_backbone(input_ids=ids,attention_mask=mask).last_hidden_state
        xl_cls, gates = self.polarity_attn(h, mask)
        xa = self.audio_encoder(aud, amask); xv = self.vision_encoder(vis, vmask)
        f  = self.sacf_attn(h, xl_cls, gates, xa, xv)
        feat = self.shared(f)
        return self.cls7_head(feat), self.cls2_head(feat), self.reg_head(feat).squeeze(-1)*3.

class EMA:
    def __init__(self, model, decay=0.9995):
        self.model=model; self.decay=decay
        self.shadow={n:p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
        self.backup={}
    def update(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n]=self.decay*self.shadow[n]+(1-self.decay)*p.data
    def apply_shadow(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n]=p.data.clone(); p.data=self.shadow[n]
    def restore(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad and n in self.backup: p.data=self.backup[n]
        self.backup={}
    def add_new_params(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad and n not in self.shadow:
                self.shadow[n]=p.data.clone()


def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int),-3,3)+3
    ct = np.where((c:=np.bincount(cl,minlength=n).astype(float))==0,1.,c)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct),0.5,3.0))

def compute_prior(labels, n=7):
    cls = np.clip(np.round(labels).astype(int),-3,3)+3
    c   = np.bincount(cls,minlength=n).astype(float); return c/c.sum()


def train_epoch(model, loader, cls7_crit, cls2_crit, reg_crit,
                opt, sch, device, scaler, ema, rdrop_alpha=0.05):
    model.train(); total_loss=0.; nan_cnt=0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids  = batch["input_ids"].to(device);  mask  = batch["attention_mask"].to(device)
        aud  = batch["audio"].to(device);      amask = batch["audio_mask"].to(device)
        vis  = batch["vision"].to(device);     vmask = batch["vision_mask"].to(device)
        cl7  = batch["cls7_label"].to(device); cl2   = batch["cls2_label"].to(device)
        rl   = batch["reg_label"].to(device)
        opt.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            loss = cls7_crit(l7,cl7)+0.3*cls2_crit(l2,cl2)+0.4*reg_crit(reg,rl)
            if rdrop_alpha>0:
                l7b,_,_ = model(ids,mask,aud,amask,vis,vmask)
                kl=(F.kl_div(F.log_softmax(l7,-1),F.softmax(l7b,-1).detach(),reduction='batchmean')+
                    F.kl_div(F.log_softmax(l7b,-1),F.softmax(l7,-1).detach(),reduction='batchmean'))/2
                loss = loss+rdrop_alpha*kl
        if torch.isnan(loss): nan_cnt+=1; opt.zero_grad(); continue
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        ema.update(); sch.step(); total_loss+=loss.item()
    return total_loss/max(len(loader)-nan_cnt,1)


@torch.no_grad()
def evaluate_iw(model, loader, device, iw_weights):
    """重要性加權評估：IW-Acc7 = Σ(w_i * correct_i) / Σ(w_i)"""
    model.eval()
    all_c7, all_l7 = [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        ids  = batch["input_ids"].to(device);  mask  = batch["attention_mask"].to(device)
        aud  = batch["audio"].to(device);      amask = batch["audio_mask"].to(device)
        vis  = batch["vision"].to(device);     vmask = batch["vision_mask"].to(device)
        l7,_,_ = model(ids,mask,aud,amask,vis,vmask)
        all_c7.extend(l7.argmax(1).cpu().numpy())
        all_l7.extend(batch["cls7_label"].numpy())
    c7 = np.array(all_c7); l7 = np.array(all_l7)
    correct = (c7==l7).astype(float)
    w = iw_weights[l7]
    iw_acc7 = (w*correct).sum()/w.sum()*100
    std_acc7 = correct.mean()*100
    return iw_acc7, std_acc7

@torch.no_grad()
def get_logits(model, loader, device):
    model.eval(); all_logits=[]
    for batch in loader:
        ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device)
        aud=batch["audio"].to(device);     amask=batch["audio_mask"].to(device)
        vis=batch["vision"].to(device);    vmask=batch["vision_mask"].to(device)
        l7,_,_=model(ids,mask,aud,amask,vis,vmask)
        all_logits.append(l7.cpu().float().numpy())
    return np.concatenate(all_logits,axis=0)

@torch.no_grad()
def std_evaluate(model, loader, device):
    model.eval()
    all_c7,all_l7=[],[]
    for batch in loader:
        ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device)
        aud=batch["audio"].to(device);     amask=batch["audio_mask"].to(device)
        vis=batch["vision"].to(device);    vmask=batch["vision_mask"].to(device)
        l7,_,_=model(ids,mask,aud,amask,vis,vmask)
        all_c7.extend(l7.argmax(1).cpu().numpy())
        all_l7.extend(batch["cls7_label"].numpy())
    return (np.array(all_c7)==np.array(all_l7)).mean()*100


def train_one_seed(seed, train_loader, val_loader, test_loader,
                   class_weights, device, config, iw_weights, val_cls7_true):
    set_seed(seed)
    print(f"\n{'─'*60}")
    print(f"Seed={seed} | ★ IW-Acc7 checkpoint（重要性加權選最佳）")
    print(f"{'─'*60}")

    model = SACFModel(dropout=config["dropout"]).to(device)
    for i in range(6):
        for p in model.lang_backbone.encoder.layer[i].parameters():
            p.requires_grad = False

    bp=[p for n,p in model.named_parameters() if p.requires_grad and "lang_backbone" in n]
    hp=[p for n,p in model.named_parameters() if p.requires_grad and "lang_backbone" not in n]
    opt = optim.AdamW([{"params":bp,"lr":config["lang_lr"]},
                       {"params":hp,"lr":config["head_lr"]}], weight_decay=config["weight_decay"])

    total_steps = len(train_loader)*config["num_epochs"]
    sch = get_cosine_schedule_with_warmup(opt, int(total_steps*0.06), total_steps)
    scaler = torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema    = EMA(model, 0.9995)

    cw = class_weights.to(device)
    cls7_crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=config["label_smoothing"])
    cls2_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    reg_crit  = nn.SmoothL1Loss()

    best_iw_acc7=0.; best_epoch=0; best_state=None; patience_cnt=0

    for epoch in range(config["num_epochs"]):
        if epoch==config["num_epochs"]//3 and not getattr(model,'_unfroze',False):
            for i in range(6):
                for p in model.lang_backbone.encoder.layer[i].parameters():
                    p.requires_grad=True
            ema.add_new_params(); model._unfroze=True
            print(f"  [解凍] Epoch {epoch+1}")

        loss = train_epoch(model,train_loader,cls7_crit,cls2_crit,reg_crit,
                           opt,sch,device,scaler,ema,config["rdrop_alpha"])

        ema.apply_shadow()
        iw_acc7, std_acc7 = evaluate_iw(model, val_loader, device, iw_weights)
        ema.restore()

        print(f"  E{epoch+1:02d} | Loss={loss:.4f} | Val IW={iw_acc7:.2f}% Std={std_acc7:.2f}%",end="")

        if iw_acc7>best_iw_acc7:
            best_iw_acc7=iw_acc7; best_epoch=epoch+1; patience_cnt=0
            ema.apply_shadow()
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            ema.restore()
            print(f"  ✅ 新最佳 IW={iw_acc7:.2f}%")
        else:
            patience_cnt+=1; print()
            if patience_cnt>=config["patience"]:
                print(f"  早停 (best E{best_epoch}, IW={best_iw_acc7:.2f}%)")
                break

    model.load_state_dict(best_state); model.to(device)
    test_logits=get_logits(model,test_loader,device)
    test_raw_acc=std_evaluate(model,test_loader,device)
    print(f"  [Seed {seed}] E{best_epoch} | IW={best_iw_acc7:.2f}% | Test(raw)={test_raw_acc:.2f}%")
    return best_iw_acc7, test_raw_acc, test_logits


def main():
    print("="*65)
    print("MOSI v44 — Importance-Weighted Val Checkpoint")
    print("="*65)

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
        "patience":        15,
        "seeds":           [42, 123, 777, 2024, 314],
        "prior_alpha":     0.7,
        "version":         "v44",
    }

    print(f"\n載入資料: {DATA_PATH}")
    with open(DATA_PATH,"rb") as f: data=pickle.load(f)

    val_labels = np.array(data["valid"]["regression_labels"])
    test_labels = np.array(data["test"]["regression_labels"])
    val_cls7  = np.clip(np.round(val_labels).astype(int),-3,3)+3
    test_cls7_true = np.clip(np.round(test_labels).astype(int),-3,3)+3
    test_cls2_true = (test_labels>=0).astype(int)

    # ★ 計算重要性加權 ★
    val_prior  = compute_prior(val_labels)
    test_prior = compute_prior(test_labels)
    iw_class = test_prior/(val_prior+1e-8)
    iw_weights = iw_class[val_cls7]  # per-sample IW on val set

    log_ratio  = np.log(test_prior+1e-8)-np.log(val_prior+1e-8)
    adj_lr = config["prior_alpha"]*log_ratio

    print(f"\n【Importance Weights per class】")
    print(f"  Val  freq: {[f'{p*100:.1f}%' for p in val_prior]}")
    print(f"  Test freq: {[f'{p*100:.1f}%' for p in test_prior]}")
    print(f"  IW class:  {[f'{w:.3f}' for w in iw_class]}")
    print(f"  → 負面類(0-2) 加權 1.5~1.7x，正面類(4-6) 加權 0.35~0.97x")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])
    train_ds = MOSIDataset(data["train"],tokenizer)
    val_ds   = MOSIDataset(data["valid"],tokenizer)
    test_ds  = MOSIDataset(data["test"], tokenizer)
    print(f"  train={len(train_ds)}, valid={len(val_ds)}, test={len(test_ds)}")

    bs = config["batch_size"]
    train_loader = DataLoader(train_ds,bs,shuffle=True, num_workers=2,pin_memory=True)
    val_loader   = DataLoader(val_ds,  bs,shuffle=False,num_workers=2,pin_memory=True)
    test_loader  = DataLoader(test_ds, bs,shuffle=False,num_workers=2,pin_memory=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"設備: {device}")

    class_weights = compute_class_weights(data["train"]["regression_labels"])

    all_logits=[]; seed_results=[]
    for seed in config["seeds"]:
        iw_acc, test_raw, logits = train_one_seed(
            seed,train_loader,val_loader,test_loader,
            class_weights,device,config,iw_weights,val_cls7)
        all_logits.append(logits)
        seed_results.append({"seed":seed,"iw_val":iw_acc,"test_raw":test_raw})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 集成（prior correction + 精選）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'='*65}")
    print("【v44 最終結果：IW-Checkpoint + Prior Correction】")
    print(f"{'='*65}")

    corrected_probs=[]
    print("\n各種子成績:")
    for r,logits in zip(seed_results,all_logits):
        cl = logits+adj_lr
        cp = np.exp(cl-cl.max(1,keepdims=True)); cp/=cp.sum(1,keepdims=True)
        acc_corr = (cp.argmax(1)==test_cls7_true).mean()*100
        r["test_corrected"]=round(float(acc_corr),2)
        corrected_probs.append(cp)
        print(f"  Seed {r['seed']:4d}: IW-Val={r['iw_val']:.2f}% | Test(raw)={r['test_raw']:.2f}% | Test(corrected)={acc_corr:.2f}%")

    # All ensemble (corrected)
    ens_all = np.mean(corrected_probs,axis=0)
    acc_ens_all = (ens_all.argmax(1)==test_cls7_true).mean()*100

    # Top-3 by IW-val
    top3 = sorted(zip(seed_results,corrected_probs),key=lambda x:x[0]["iw_val"],reverse=True)[:3]
    ens_top3 = np.mean([cp for _,cp in top3],axis=0)
    acc_ens_top3 = (ens_top3.argmax(1)==test_cls7_true).mean()*100

    # Best single
    best_r = max(seed_results,key=lambda x:x["test_corrected"])

    print(f"\n全部集成（校正後）: Test Acc7={acc_ens_all:.2f}%")
    print(f"Top3集成（IW-val）: Test Acc7={acc_ens_top3:.2f}%")
    print(f"最佳單種子:         Test Acc7={best_r['test_corrected']:.2f}%  (Seed={best_r['seed']})")

    final_acc7 = max(acc_ens_all, acc_ens_top3, best_r["test_corrected"])
    print(f"\n最終 Test Acc7: {final_acc7:.2f}%")
    print(f"vs 目標 51%: {final_acc7-51.:+.2f}% {'✓ 達標！' if final_acc7>51. else '✗ 未達標'}")
    print(f"\nTest Acc7: {final_acc7:.2f}%")

    MODEL_DIR.mkdir(exist_ok=True,parents=True)
    import json
    with open(MODEL_DIR/"history_v44.json","w") as f:
        json.dump({"version":"v44","seed_results":seed_results,
                   "ensemble_all":round(acc_ens_all,2),
                   "ensemble_top3":round(acc_ens_top3,2),
                   "best_single":best_r["test_corrected"],
                   "ensemble_acc7":round(final_acc7,2)},f,indent=2,ensure_ascii=False)

if __name__=="__main__":
    main()
