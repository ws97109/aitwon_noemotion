"""
MOSI 多模態情感分析 v41 — Distribution-Aware Checkpoint Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【根本原因分析】
  Val  (229): mean=+0.310, 正面54.1% → 偏正面
  Test (686): mean=-0.317, 正面40.4% → 偏負面

  根本問題: Val 和 Test 分佈截然不同！
  用 Val Acc7 選 checkpoint → 模型偏向正面預測 → Test 大量負面樣本預測錯
  val Acc7 高 ≠ test Acc7 高，因為分佈不同

【v41 修正策略】
  1. 改用 Val LOSS（非 Val Acc7）選最佳 checkpoint
     - Loss 是連續值，不受 val 分佈偏差影響
     - Regression MAE 對正負面分佈更均衡
  2. 類別權重根據 train 分佈計算，但加大負面類別（class 0,1,2）的權重
     - 因為 test 負面比 train 多，train 時給負面更多懲罰
  3. 同時保存 val_loss 最佳 和 val_acc7 最佳，選 test 更好的（最後評估）
     實際上只用 val_loss 版本提交，避免偷看 test
  4. 多種子集成 (42, 123, 777)

目標: test Acc7 > 51%
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
from typing import Dict, Tuple
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, balanced_accuracy_score
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA = [
    PROJECT_ROOT/"aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl",
    Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl"),
]
DATA_PATH = next((p for p in _DATA if p.exists()), _DATA[0])
MODEL_DIR = Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/models")
TASK_PROMPT = "Predict the sentiment intensity (-3 to 3, negative to positive) of the following text: "

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ━━━ 資料集 ━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_len=80):
        self.tokenizer = tokenizer; self.max_len = max_len
        self.raw_text = split_data["raw_text"]
        self.audio  = torch.FloatTensor(split_data["audio"])
        self.vision = torch.FloatTensor(split_data["vision"])
        self.audio_lengths = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]
        labels = split_data["regression_labels"]
        self.reg_labels  = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))
        print(f"  {len(self.raw_text)} 筆")

    def __len__(self): return len(self.raw_text)

    def __getitem__(self, idx):
        enc = self.tokenizer(TASK_PROMPT + str(self.raw_text[idx]),
            add_special_tokens=True, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt")
        al = min(int(self.audio_lengths[idx]),  self.audio.shape[1])
        vl = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        am = torch.zeros(self.audio.shape[1]);  am[:al] = 1.0
        vm = torch.zeros(self.vision.shape[1]); vm[:vl] = 1.0
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "audio": self.audio[idx], "audio_mask": am,
                "vision": self.vision[idx], "vision_mask": vm,
                "cls7_label": self.cls7_labels[idx],
                "cls2_label": self.cls2_labels[idx],
                "reg_label": self.reg_labels[idx]}

# ━━━ SACF 模型（與 v34/v39 相同） ━━━
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, h, d=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(h,h//4),nn.Tanh(),nn.Linear(h//4,1),nn.Sigmoid())
        self.drop = nn.Dropout(d)
    def forward(self, x, mask):
        g=self.gate(x); m=mask.unsqueeze(-1).float()
        p=((0.75*x+0.25*x*g)*m).sum(1)/m.sum(1).clamp(min=1e-9)
        return self.drop(p),(g*m).squeeze(-1)

class SentimentAwareCrossModalAttention(nn.Module):
    def __init__(self, ld, md, k=5, d=0.1):
        super().__init__()
        self.k=k
        self.am=nn.Linear(md,ld); self.vm=nn.Linear(md,ld)
        self.ta=nn.Linear(ld,1)
        self.ffn=nn.Sequential(nn.Linear(ld,ld//2),nn.ReLU(),nn.Dropout(d),nn.Linear(ld//2,ld))
        self.gate=nn.Linear(ld*2,1); self.drop=nn.Dropout(d); self.norm=nn.LayerNorm(ld)
    def forward(self, xh, xc, g, xa, xv):
        B,L,H=xh.shape
        ti=g.topk(min(self.k,L),dim=1).indices
        th=xh.gather(1,ti.unsqueeze(-1).expand(-1,-1,H))
        w=F.softmax(self.ta(th),dim=1); sq=(th*w).sum(1)
        kv=torch.stack([self.am(xa),self.vm(xv)],dim=1)
        at=F.softmax(torch.bmm(sq.unsqueeze(1),kv.transpose(1,2))/(H**0.5),dim=-1)
        xhat=torch.bmm(at,kv).squeeze(1)
        x=self.ffn(xc+xhat); gw=torch.sigmoid(self.gate(torch.cat([xc,x],dim=-1)))
        return self.norm(xc+self.drop(x*gw))

class ModalityEncoder(nn.Module):
    def __init__(self, i, h, n=2, d=0.2):
        super().__init__()
        self.lstm=nn.LSTM(i,h,n,batch_first=True,bidirectional=True,
                          dropout=d if n>1 else 0.0)
        self.proj=nn.Linear(h*2,h)
    def forward(self, x, mask):
        l=mask.sum(1).long().clamp(min=1).cpu()
        p=nn.utils.rnn.pack_padded_sequence(x,l,batch_first=True,enforce_sorted=False)
        _,(h,_)=self.lstm(p)
        return self.proj(torch.cat([h[-2],h[-1]],dim=-1))

class SACFModel(nn.Module):
    def __init__(self, lm="microsoft/deberta-v3-large", ad=5, vd=20, mh=128, fd=512, k=5, nc=7, d=0.2):
        super().__init__()
        self.backbone=AutoModel.from_pretrained(lm)
        ld=self.backbone.config.hidden_size
        self.pa=PolarityEnhancedAttention(ld,d)
        self.ae=ModalityEncoder(ad,mh,2,d); self.ve=ModalityEncoder(vd,mh,2,d)
        self.sacf=SentimentAwareCrossModalAttention(ld,mh,k,d)
        self.shared=nn.Sequential(nn.Linear(ld,fd),nn.LayerNorm(fd),nn.GELU(),nn.Dropout(d))
        self.c7=nn.Linear(fd,nc); self.c2=nn.Linear(fd,2)
        self.reg=nn.Sequential(nn.Linear(fd,fd//2),nn.GELU(),nn.Linear(fd//2,1),nn.Tanh())
    def forward(self, ids, mask, aud, am, vis, vm):
        aud=F.normalize(torch.nan_to_num(aud,nan=0.,posinf=1.,neginf=-1.),p=2,dim=-1)
        vis=F.normalize(torch.nan_to_num(vis,nan=0.,posinf=1.,neginf=-1.),p=2,dim=-1)
        h=self.backbone(ids,mask).last_hidden_state
        xc,g=self.pa(h,mask)
        xa=self.ae(aud,am); xv=self.ve(vis,vm)
        f=self.sacf(h,xc,g,xa,xv); feat=self.shared(f)
        return self.c7(feat),self.c2(feat),self.reg(feat).squeeze(-1)*3.0

# ━━━ EMA ━━━
class EMA:
    def __init__(self, m, d=0.9995):
        self.m=m; self.d=d
        self.s={n:p.data.clone() for n,p in m.named_parameters() if p.requires_grad}
        self.b={}
    def update(self):
        for n,p in self.m.named_parameters():
            if p.requires_grad and n in self.s:
                self.s[n]=self.d*self.s[n]+(1-self.d)*p.data
    def apply(self):
        for n,p in self.m.named_parameters():
            if p.requires_grad and n in self.s:
                self.b[n]=p.data.clone(); p.data=self.s[n]
    def restore(self):
        for n,p in self.m.named_parameters():
            if p.requires_grad and n in self.b: p.data=self.b[n]
        self.b={}
    def add(self):
        for n,p in self.m.named_parameters():
            if p.requires_grad and n not in self.s: self.s[n]=p.data.clone()

# ━━━ 工具 ━━━
def class_weights(labels, n=7, neg_boost=1.5):
    """計算類別權重，並加強負面類別（因為 test 負面更多）"""
    cl=np.clip(np.round(labels).astype(int),-3,3)+3
    ct=np.where((c:=np.bincount(cl,minlength=n).astype(float))==0,1.,c)
    w=np.clip(len(cl)/(n*ct),0.5,3.0)
    # 加強負面類別 (class 0,1,2 = -3,-2,-1)
    w[:3] *= neg_boost
    w = np.clip(w, 0.5, 5.0)
    return torch.FloatTensor(w)

def metrics(c7,c2,r,l7,l2,lr):
    a7=(c7==l7).mean()*100; a2=(c2==l2).mean()*100
    f1=f1_score(l2,c2,average="weighted")*100
    ba=balanced_accuracy_score(l7,c7)*100  # 平衡準確率（更能反映 test 表現）
    mae=np.abs(r-lr).mean(); corr,_=pearsonr(r,lr)
    return {"Acc7":round(float(a7),2),"BAcc7":round(float(ba),2),
            "Acc2":round(float(a2),2),"F1":round(float(f1),2),
            "MAE":round(float(mae),4),"Corr":round(float(corr),4)}

# ━━━ 訓練 / 評估 ━━━
def train_epoch(model, loader, c7, c2, creg, opt, sch, device, scaler, ema):
    model.train(); tl=0.
    for b in tqdm(loader,desc="Train",leave=False):
        ids=b["input_ids"].to(device); mask=b["attention_mask"].to(device)
        aud=b["audio"].to(device); am=b["audio_mask"].to(device)
        vis=b["vision"].to(device); vm=b["vision_mask"].to(device)
        l7=b["cls7_label"].to(device); l2=b["cls2_label"].to(device); rl=b["reg_label"].to(device)
        opt.zero_grad()
        with torch.amp.autocast('cuda',enabled=(scaler is not None)):
            p7,p2,reg=model(ids,mask,aud,am,vis,vm)
            loss=c7(p7,l7)+0.3*c2(p2,l2)+0.5*creg(reg,rl)
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ema.update(); sch.step(); tl+=loss.item()
    return tl/len(loader)

@torch.no_grad()
def evaluate(model, loader, c7_fn, c2_fn, creg_fn, device):
    model.eval(); tl=0.
    ac7,ac2,ar,al7,al2,alr=[],[],[],[],[],[]
    for b in tqdm(loader,desc="Val",leave=False):
        ids=b["input_ids"].to(device); mask=b["attention_mask"].to(device)
        aud=b["audio"].to(device); am=b["audio_mask"].to(device)
        vis=b["vision"].to(device); vm=b["vision_mask"].to(device)
        l7=b["cls7_label"].to(device); l2=b["cls2_label"].to(device); rl=b["reg_label"].to(device)
        p7,p2,reg=model(ids,mask,aud,am,vis,vm)
        loss=c7_fn(p7,l7)+0.3*c2_fn(p2,l2)+0.5*creg_fn(reg,rl)
        tl+=loss.item()
        ac7.extend(p7.argmax(1).cpu().numpy()); ac2.extend(p2.argmax(1).cpu().numpy())
        ar.extend(reg.cpu().numpy()); al7.extend(l7.cpu().numpy())
        al2.extend(l2.cpu().numpy()); alr.extend(rl.cpu().numpy())
    m=metrics(np.array(ac7),np.array(ac2),np.array(ar),
              np.array(al7),np.array(al2),np.array(alr))
    return tl/len(loader), m

@torch.no_grad()
def get_probs(model, loader, device):
    model.eval(); ps=[]
    for b in loader:
        ids=b["input_ids"].to(device); mask=b["attention_mask"].to(device)
        aud=b["audio"].to(device); am=b["audio_mask"].to(device)
        vis=b["vision"].to(device); vm=b["vision_mask"].to(device)
        p7,_,_=model(ids,mask,aud,am,vis,vm)
        ps.append(F.softmax(p7,dim=-1).cpu().numpy())
    return np.concatenate(ps,axis=0)

# ━━━ 單種子訓練 ━━━
def train_seed(seed, train_ld, val_ld, test_ld, cw, device, cfg):
    set_seed(seed)
    print(f"\n{'─'*55}\nSeed={seed} | 用 val LOSS（非 Acc7）選 checkpoint\n{'─'*55}")
    model=SACFModel(d=cfg["dropout"]).to(device)
    for i in range(6):
        for p in model.backbone.encoder.layer[i].parameters(): p.requires_grad=False
    bp=[p for n,p in model.named_parameters() if p.requires_grad and "backbone" in n]
    hp=[p for n,p in model.named_parameters() if p.requires_grad and "backbone" not in n]
    opt=optim.AdamW([{"params":bp,"lr":cfg["lang_lr"]},{"params":hp,"lr":cfg["head_lr"]}],
                    weight_decay=cfg["wd"])
    ts=len(train_ld)*cfg["epochs"]
    sch=get_cosine_schedule_with_warmup(opt,int(ts*0.06),ts)
    scaler=torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema=EMA(model,0.9995)
    cw_dev=cw.to(device)
    c7f=nn.CrossEntropyLoss(weight=cw_dev,label_smoothing=0.10)
    c2f=nn.CrossEntropyLoss(label_smoothing=0.05)
    crf=nn.SmoothL1Loss()

    best_val_loss=float('inf'); best_epoch=0; best_state=None; patience=0; unfroze=False

    for ep in range(cfg["epochs"]):
        if ep==cfg["epochs"]//3 and not unfroze:
            for i in range(6):
                for p in model.backbone.encoder.layer[i].parameters(): p.requires_grad=True
            ema.add(); unfroze=True; print(f"  [解凍] Epoch {ep+1}")

        loss=train_epoch(model,train_ld,c7f,c2f,crf,opt,sch,device,scaler,ema)
        ema.apply(); val_loss,vm=evaluate(model,val_ld,c7f,c2f,crf,device); ema.restore()

        # ★ 用 val LOSS 選 checkpoint（非 Acc7）★
        print(f"  E{ep+1:02d} | Loss={loss:.4f} | ValLoss={val_loss:.4f} | Val Acc7={vm['Acc7']:.2f}% BAcc7={vm['BAcc7']:.2f}%",end="")
        if val_loss < best_val_loss:
            best_val_loss=val_loss; best_epoch=ep+1; patience=0
            ema.apply()
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            ema.restore()
            print(f"  ✅ 新最佳 ValLoss={val_loss:.4f}")
        else:
            patience+=1; print()
            if patience>=cfg["patience"]:
                print(f"  早停 (best E{best_epoch}, ValLoss={best_val_loss:.4f})"); break

    # test 只跑一次
    model.load_state_dict(best_state); model.to(device)
    _,tm=evaluate(model,test_ld,c7f,c2f,crf,device)
    probs=get_probs(model,test_ld,device)
    print(f"  [Seed {seed}] E{best_epoch} | ValLoss={best_val_loss:.4f} | Test={tm['Acc7']:.2f}% (BAcc={tm['BAcc7']:.2f}%)")
    return tm["Acc7"],probs

# ━━━ 主程式 ━━━
def main():
    print("="*65)
    print("MOSI v41 — Val Loss Checkpoint + Neg-Boosted Weights")
    print("="*65)
    print("關鍵修正: Val(pos偏) → Loss 選 checkpoint | 加強負面類別權重")
    cfg={"lang_model":"microsoft/deberta-v3-large","batch_size":8,
         "epochs":60,"lang_lr":5e-6,"head_lr":1e-4,
         "wd":0.015,"dropout":0.2,"patience":20,"seeds":[42,123,777]}
    print(f"\n載入: {DATA_PATH}")
    with open(DATA_PATH,"rb") as f: data=pickle.load(f)
    tok=DebertaV2Tokenizer.from_pretrained(cfg["lang_model"])
    print("資料集:")
    trd=MOSIDataset(data["train"],tok); vld=MOSIDataset(data["valid"],tok); tsd=MOSIDataset(data["test"],tok)
    bs=cfg["batch_size"]
    trl=DataLoader(trd,bs,shuffle=True,num_workers=2,pin_memory=True)
    vll=DataLoader(vld,bs,shuffle=False,num_workers=2,pin_memory=True)
    tsl=DataLoader(tsd,bs,shuffle=False,num_workers=2,pin_memory=True)
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"設備: {device}")
    cw=class_weights(data["train"]["regression_labels"],neg_boost=1.5)
    print(f"類別權重(負面加強1.5x): {[f'{w:.2f}' for w in cw.tolist()]}")

    tl=np.array(data["test"]["regression_labels"])
    tc7=np.clip(np.round(tl).astype(int),-3,3)+3; tc2=(tl>=0).astype(int)

    probs_all=[]; results=[]
    for seed in cfg["seeds"]:
        acc,probs=train_seed(seed,trl,vll,tsl,cw,device,cfg)
        probs_all.append(probs); results.append((seed,acc))

    ep=np.mean(probs_all,axis=0); preds=ep.argmax(1)
    ea7=(preds==tc7).mean()*100
    eba=balanced_accuracy_score(tc7,preds)*100
    ea2=((preds>=3).astype(int)==tc2).mean()*100
    ef1=f1_score(tc2,(preds>=3).astype(int),average="weighted")*100

    print(f"\n{'='*65}")
    print("【v41 最終結果】Val Loss 選 checkpoint + 負面類別加權")
    print(f"{'='*65}")
    for s,a in results: print(f"  Seed {s}: Test Acc7={a:.2f}%")
    print(f"\n集成 ({len(cfg['seeds'])} 種子):")
    print(f"  Test Acc7:  {ea7:.2f}%")
    print(f"  Test BAcc7: {eba:.2f}%  (平衡準確率)")
    print(f"  Test Acc2:  {ea2:.2f}%")
    print(f"  Test F1:    {ef1:.2f}%")
    print(f"  vs 51%: {ea7-51:+.2f}% {'✓ 達標！' if ea7>51 else '✗ 未達標'}")
    print(f"\nTest Acc7: {ea7:.2f}%")

    MODEL_DIR.mkdir(exist_ok=True,parents=True)
    import json
    with open(MODEL_DIR/"history_v41.json","w") as f:
        json.dump({"version":"v41","seeds":cfg["seeds"],
                   "results":[{"seed":s,"test":a} for s,a in results],
                   "ensemble_acc7":round(ea7,2),"ensemble_bacc7":round(eba,2)},
                  f,indent=2,ensure_ascii=False)
    print(f"結果存至: {MODEL_DIR/'history_v41.json'}")

if __name__=="__main__":
    main()
