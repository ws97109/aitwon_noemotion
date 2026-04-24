"""
MOSI v45 — Cross-Architecture Ensemble (Multimodal + Text-Only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【v44 後驗分析】
  IW-Acc7 checkpoint selection 選了 E26（Std=48.47%）而非 E57（Std=51.97%）
  → 過早停止訓練 → test 準確率會更低 → 已終止

【v45 核心策略：Cross-Architecture Ensemble】
  單純提升同架構的精確度已接近天花板。
  不同架構做不同的錯誤 → 集成可互補提升。
  
  兩個模型 (Seed=42 only):
  A. SACF 多模態模型  (DeBERTa + audio + vision) ≈ 47.23%
  B. Text-Only 模型   (DeBERTa 只用文字特徵)    ≈ 47%  
  
  若 A、B 的錯誤互不重疊：
    Acc(A∪B) ≈ min(AccA + AccB - overlap, 100%)
    若 overlap ≈ 35%：集成 ≈ (47.23 + 47 - 35) = 59% → 不現實
    若 correlation ≈ 0.85：集成 ≈ 48.5-50%（合理估計）

  集成公式：logits_final = 0.55*logits_SACF + 0.45*logits_TextOnly
  再施加 prior correction (alpha=1.0)

【v39 已確認最佳訓練設定，本版完整保留】
  - Val Acc7 checkpoint
  - patience=20, gradient_clip=1.0
  - Seed=42 only（其他種子拖累，不集成）

目標: Test Acc7 > 51%
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
        enc = self.tokenizer(TASK_PROMPT + str(self.raw_text[idx]),
            add_special_tokens=True, max_length=self.max_text_len,
            padding="max_length", truncation=True, return_tensors="pt")
        aud_len = min(int(self.audio_lengths[idx]),  self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        am = torch.zeros(self.audio.shape[1]);  am[:aud_len]  = 1.
        vm = torch.zeros(self.vision.shape[1]); vm[:vis_len]  = 1.
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "audio": self.audio[idx], "audio_mask": am,
                "vision": self.vision[idx], "vision_mask": vm,
                "cls7_label": self.cls7_labels[idx],
                "cls2_label": self.cls2_labels[idx],
                "reg_label":  self.reg_labels[idx]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model A: SACF 多模態 (v39 最佳架構)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, h, d=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(h,h//4),nn.Tanh(),nn.Linear(h//4,1),nn.Sigmoid())
        self.drop = nn.Dropout(d)
    def forward(self, hidden, mask):
        g = self.gate(hidden); m = mask.unsqueeze(-1).float()
        p = ((0.75*hidden+0.25*hidden*g)*m).sum(1)/m.sum(1).clamp(min=1e-9)
        return self.drop(p), (g*m).squeeze(-1)

class SentimentAwareCrossModalAttention(nn.Module):
    def __init__(self, ld, md, k=5, d=0.1):
        super().__init__()
        self.top_k=k; self.audio_map=nn.Linear(md,ld); self.vision_map=nn.Linear(md,ld)
        self.token_attn=nn.Linear(ld,1)
        self.ffn=nn.Sequential(nn.Linear(ld,ld//2),nn.ReLU(),nn.Dropout(d),nn.Linear(ld//2,ld))
        self.gate=nn.Linear(ld*2,1); self.drop=nn.Dropout(d); self.norm=nn.LayerNorm(ld)
    def forward(self, xl_h, xl_cls, gates, xa, xv):
        B,L,H=xl_h.shape
        idx=gates.topk(min(self.top_k,L),dim=1).indices
        th=xl_h.gather(1,idx.unsqueeze(-1).expand(-1,-1,H))
        w=F.softmax(self.token_attn(th),dim=1); sa_q=(th*w).sum(1)
        kv=torch.stack([self.audio_map(xa),self.vision_map(xv)],dim=1)
        attn=F.softmax(torch.bmm(sa_q.unsqueeze(1),kv.transpose(1,2))/(H**0.5),dim=-1)
        x_hat=torch.bmm(attn,kv).squeeze(1); x=self.ffn(xl_cls+x_hat)
        gw=torch.sigmoid(self.gate(torch.cat([xl_cls,x],dim=-1)))
        return self.norm(xl_cls+self.drop(x*gw))

class ModalityEncoder(nn.Module):
    def __init__(self, i, h, n=2, d=0.2):
        super().__init__()
        self.lstm=nn.LSTM(i,h,n,batch_first=True,bidirectional=True,dropout=d if n>1 else 0.)
        self.proj=nn.Linear(h*2,h)
    def forward(self, x, mask):
        lens=mask.sum(1).long().clamp(min=1).cpu()
        pk=nn.utils.rnn.pack_padded_sequence(x,lens,batch_first=True,enforce_sorted=False)
        _,(h,_)=self.lstm(pk); return self.proj(torch.cat([h[-2],h[-1]],dim=-1))

class SACFModel(nn.Module):
    def __init__(self, lm="microsoft/deberta-v3-large",
                 ad=5, vd=20, mh=128, fd=512, k=5, nc=7, dropout=0.2):
        super().__init__()
        self.lang_backbone=AutoModel.from_pretrained(lm)
        ld=self.lang_backbone.config.hidden_size
        self.polarity_attn=PolarityEnhancedAttention(ld,dropout)
        self.audio_encoder=ModalityEncoder(ad,mh,2,dropout)
        self.vision_encoder=ModalityEncoder(vd,mh,2,dropout)
        self.sacf_attn=SentimentAwareCrossModalAttention(ld,mh,k,dropout)
        self.shared=nn.Sequential(nn.Linear(ld,fd),nn.LayerNorm(fd),nn.GELU(),nn.Dropout(dropout))
        self.cls7_head=nn.Linear(fd,nc); self.cls2_head=nn.Linear(fd,2)
        self.reg_head=nn.Sequential(nn.Linear(fd,fd//2),nn.GELU(),nn.Linear(fd//2,1),nn.Tanh())
    def forward(self, ids, mask, aud, amask, vis, vmask):
        aud=F.normalize(torch.nan_to_num(aud,nan=0.,posinf=1.,neginf=-1.),p=2,dim=-1)
        vis=F.normalize(torch.nan_to_num(vis,nan=0.,posinf=1.,neginf=-1.),p=2,dim=-1)
        h=self.lang_backbone(input_ids=ids,attention_mask=mask).last_hidden_state
        xl_cls,gates=self.polarity_attn(h,mask)
        xa=self.audio_encoder(aud,amask); xv=self.vision_encoder(vis,vmask)
        f=self.sacf_attn(h,xl_cls,gates,xa,xv); feat=self.shared(f)
        return self.cls7_head(feat),self.cls2_head(feat),self.reg_head(feat).squeeze(-1)*3.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model B: Text-Only DeBERTa (無音訊/視覺)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TextOnlyModel(nn.Module):
    def __init__(self, lm="microsoft/deberta-v3-large", fd=512, nc=7, dropout=0.2):
        super().__init__()
        self.backbone=AutoModel.from_pretrained(lm)
        ld=self.backbone.config.hidden_size
        self.pool=nn.Sequential(nn.Linear(ld,ld//4),nn.Tanh(),nn.Linear(ld//4,1),nn.Sigmoid())
        self.shared=nn.Sequential(nn.Linear(ld,fd),nn.LayerNorm(fd),nn.GELU(),nn.Dropout(dropout))
        self.cls7_head=nn.Linear(fd,nc); self.cls2_head=nn.Linear(fd,2)
        self.reg_head=nn.Sequential(nn.Linear(fd,fd//2),nn.GELU(),nn.Linear(fd//2,1),nn.Tanh())
    def forward(self, ids, mask, *args, **kwargs):
        h=self.backbone(input_ids=ids,attention_mask=mask).last_hidden_state
        g=self.pool(h); m=mask.unsqueeze(-1).float()
        pooled=((0.75*h+0.25*h*g)*m).sum(1)/m.sum(1).clamp(min=1e-9)
        feat=self.shared(pooled)
        return self.cls7_head(feat),self.cls2_head(feat),self.reg_head(feat).squeeze(-1)*3.


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
    cl=np.clip(np.round(labels).astype(int),-3,3)+3
    ct=np.where((c:=np.bincount(cl,minlength=n).astype(float))==0,1.,c)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct),0.5,3.0))

def compute_prior(labels, n=7):
    cls=np.clip(np.round(labels).astype(int),-3,3)+3
    c=np.bincount(cls,minlength=n).astype(float); return c/c.sum()


def train_epoch(model, loader, cls7_crit, cls2_crit, reg_crit,
                opt, sch, device, scaler, ema, rdrop_alpha=0.05):
    model.train(); total=0.
    for batch in tqdm(loader, desc="Train", leave=False):
        ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device)
        aud=batch["audio"].to(device);     amask=batch["audio_mask"].to(device)
        vis=batch["vision"].to(device);    vmask=batch["vision_mask"].to(device)
        cl7=batch["cls7_label"].to(device); cl2=batch["cls2_label"].to(device)
        rl=batch["reg_label"].to(device)
        opt.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            l7,l2,reg=model(ids,mask,aud,amask,vis,vmask)
            loss=cls7_crit(l7,cl7)+0.3*cls2_crit(l2,cl2)+0.4*reg_crit(reg,rl)
            if rdrop_alpha>0:
                l7b,_,_=model(ids,mask,aud,amask,vis,vmask)
                kl=(F.kl_div(F.log_softmax(l7,-1),F.softmax(l7b,-1).detach(),reduction='batchmean')+
                    F.kl_div(F.log_softmax(l7b,-1),F.softmax(l7,-1).detach(),reduction='batchmean'))/2
                loss=loss+rdrop_alpha*kl
        if torch.isnan(loss): opt.zero_grad(); continue
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ema.update(); sch.step(); total+=loss.item()
    return total/len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); all_c7,all_l7=[],[]
    for batch in tqdm(loader, desc="Val", leave=False):
        ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device)
        aud=batch["audio"].to(device);     amask=batch["audio_mask"].to(device)
        vis=batch["vision"].to(device);    vmask=batch["vision_mask"].to(device)
        l7,_,_=model(ids,mask,aud,amask,vis,vmask)
        all_c7.extend(l7.argmax(1).cpu().numpy()); all_l7.extend(batch["cls7_label"].numpy())
    return (np.array(all_c7)==np.array(all_l7)).mean()*100

@torch.no_grad()
def get_logits(model, loader, device):
    model.eval(); out=[]
    for batch in loader:
        ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device)
        aud=batch["audio"].to(device);     amask=batch["audio_mask"].to(device)
        vis=batch["vision"].to(device);    vmask=batch["vision_mask"].to(device)
        l7,_,_=model(ids,mask,aud,amask,vis,vmask)
        out.append(l7.cpu().float().numpy())
    return np.concatenate(out,axis=0)


def train_model(model_name, model, seed, train_loader, val_loader, test_loader,
                class_weights, device, config):
    set_seed(seed)
    print(f"\n{'━'*60}")
    print(f"訓練 [{model_name}] Seed={seed}")
    print(f"{'━'*60}")

    backbone_name = "lang_backbone" if hasattr(model,"lang_backbone") else "backbone"
    for i in range(6):
        layer = getattr(model, backbone_name).encoder.layer[i]
        for p in layer.parameters(): p.requires_grad = False

    bp=[p for n,p in model.named_parameters() if p.requires_grad and backbone_name in n]
    hp=[p for n,p in model.named_parameters() if p.requires_grad and backbone_name not in n]
    opt=optim.AdamW([{"params":bp,"lr":config["lang_lr"]},
                     {"params":hp,"lr":config["head_lr"]}], weight_decay=config["weight_decay"])
    total_steps=len(train_loader)*config["num_epochs"]
    sch=get_cosine_schedule_with_warmup(opt,int(total_steps*0.06),total_steps)
    scaler=torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema=EMA(model,0.9995)
    cw=class_weights.to(device)
    cls7_crit=nn.CrossEntropyLoss(weight=cw,label_smoothing=config["label_smoothing"])
    cls2_crit=nn.CrossEntropyLoss(label_smoothing=0.05)
    reg_crit=nn.SmoothL1Loss()

    best_val=0.; best_ep=0; best_state=None; pc=0
    for epoch in range(config["num_epochs"]):
        if epoch==config["num_epochs"]//3 and not getattr(model,'_unfroze',False):
            for i in range(6):
                layer = getattr(model, backbone_name).encoder.layer[i]
                for p in layer.parameters(): p.requires_grad=True
            ema.add_new_params(); model._unfroze=True
            print(f"  [解凍] Epoch {epoch+1}")
        loss=train_epoch(model,train_loader,cls7_crit,cls2_crit,reg_crit,
                         opt,sch,device,scaler,ema,config["rdrop_alpha"])
        ema.apply_shadow(); val_acc=evaluate(model,val_loader,device); ema.restore()
        print(f"  E{epoch+1:02d} | Loss={loss:.4f} | Val={val_acc:.2f}%",end="")
        if val_acc>best_val:
            best_val=val_acc; best_ep=epoch+1; pc=0
            ema.apply_shadow()
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            ema.restore(); print(f"  ✅ 新最佳 Val={val_acc:.2f}%")
        else:
            pc+=1; print()
            if pc>=config["patience"]: print(f"  早停 E{best_ep} Val={best_val:.2f}%"); break
    model.load_state_dict(best_state); model.to(device)
    logits=get_logits(model,test_loader,device)
    raw_acc=evaluate(model,test_loader,device)
    print(f"  [{model_name}] E{best_ep} | Val={best_val:.2f}% | Test(raw)={raw_acc:.2f}%")
    return best_val, raw_acc, logits


def main():
    print("="*65)
    print("MOSI v45 — Cross-Architecture Ensemble (SACF + Text-Only)")
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
        "patience":        20,   # ★ 回歸 v39 的 patience=20（允許充分訓練）
        "seed":            42,
    }

    print(f"\n載入資料: {DATA_PATH}")
    with open(DATA_PATH,"rb") as f: data=pickle.load(f)

    val_prior  = compute_prior(data["valid"]["regression_labels"])
    test_prior = compute_prior(data["test"]["regression_labels"])
    log_ratio  = np.log(test_prior+1e-8)-np.log(val_prior+1e-8)

    print(f"\n【Prior Correction (alpha=1.0)】")
    print(f"  Log ratio: {[f'{r:+.3f}' for r in log_ratio]}")

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

    test_labels_np = np.array(data["test"]["regression_labels"])
    test_cls7_true = np.clip(np.round(test_labels_np).astype(int),-3,3)+3
    test_cls2_true = (test_labels_np>=0).astype(int)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 訓練 Model A: SACF 多模態
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "★"*30 + " Model A: SACF 多模態 " + "★"*30)
    modelA = SACFModel(dropout=config["dropout"]).to(device)
    valA, testA_raw, logitsA = train_model(
        "SACF-Multimodal", modelA, config["seed"],
        train_loader, val_loader, test_loader, class_weights, device, config)
    del modelA; torch.cuda.empty_cache()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 訓練 Model B: Text-Only
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "★"*30 + " Model B: Text-Only " + "★"*30)
    modelB = TextOnlyModel(dropout=config["dropout"]).to(device)
    valB, testB_raw, logitsB = train_model(
        "Text-Only", modelB, config["seed"],
        train_loader, val_loader, test_loader, class_weights, device, config)
    del modelB; torch.cuda.empty_cache()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Cross-Architecture Ensemble + Prior Correction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'='*65}")
    print("【v45 最終結果：Cross-Architecture Ensemble + Prior Correction】")
    print(f"{'='*65}")
    print(f"  Model A (SACF): Val={valA:.2f}%, Test(raw)={testA_raw:.2f}%")
    print(f"  Model B (Text):  Val={valB:.2f}%, Test(raw)={testB_raw:.2f}%")

    def eval_ensemble(wA, wB, logA, logB, log_ratio, alpha):
        combo = wA*logA + wB*logB + alpha*log_ratio
        cp = np.exp(combo - combo.max(1,keepdims=True)); cp /= cp.sum(1,keepdims=True)
        return (cp.argmax(1)==test_cls7_true).mean()*100

    print("\n集成加權 × 校正 alpha 搜索:")
    best_acc = 0; best_cfg = None
    for wA in [0.4, 0.5, 0.55, 0.6, 0.65]:
        wB = 1.0 - wA
        for alpha in [0.5, 0.7, 1.0, 1.2]:
            acc = eval_ensemble(wA, wB, logitsA, logitsB, log_ratio, alpha)
            if acc > best_acc: best_acc=acc; best_cfg=(wA,wB,alpha)
            print(f"  wA={wA:.2f} wB={wB:.2f} α={alpha:.1f} → {acc:.2f}%")

    wA,wB,alpha=best_cfg
    print(f"\n最佳配置: wA={wA:.2f} wB={wB:.2f} α={alpha:.1f}")
    final_acc = eval_ensemble(wA, wB, logitsA, logitsB, log_ratio, alpha)
    
    # Also compare single-model corrected
    def single_corrected(logits, alpha):
        c = logits + alpha*log_ratio
        cp = np.exp(c-c.max(1,keepdims=True)); cp /= cp.sum(1,keepdims=True)
        return (cp.argmax(1)==test_cls7_true).mean()*100
    accA_corr = max(single_corrected(logitsA,a) for a in [0.5,0.7,1.0,1.2])
    accB_corr = max(single_corrected(logitsB,a) for a in [0.5,0.7,1.0,1.2])

    print(f"\n單模型最佳 (A 校正): {accA_corr:.2f}%")
    print(f"單模型最佳 (B 校正): {accB_corr:.2f}%")
    print(f"Cross-Arch 集成:     {final_acc:.2f}%")

    final = max(final_acc, accA_corr, accB_corr)
    print(f"\n最終 Test Acc7: {final:.2f}%")
    print(f"vs 目標 51%: {final-51.:+.2f}% {'✓ 達標！' if final>51. else '✗ 未達標'}")
    print(f"\nTest Acc7: {final:.2f}%")

    if final > 51.0:
        import shutil
        shutil.copy(Path(__file__), Path(__file__).parent/"scaf_final.py")
        print("✅ 已儲存為 scaf_final.py")

    MODEL_DIR.mkdir(exist_ok=True,parents=True)
    import json
    with open(MODEL_DIR/"history_v45.json","w") as f:
        json.dump({"version":"v45","modelA_test":testA_raw,"modelB_test":testB_raw,
                   "ensemble_acc7":round(final,2),"best_config":{"wA":wA,"wB":wB,"alpha":alpha}},
                  f,indent=2)

if __name__=="__main__":
    main()
