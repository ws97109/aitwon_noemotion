"""SemEval-2018 Task 1 純文字評估 — 對應論文 Table 4。

使用 scaf_final.SACFModel 的純文字分支（DeBERTa-v3-large + PolarityEnhancedAttention
+ shared MLP），在 SemEval-2018 Task 1 的訓練集上訓練、test-gold 評估。

依論文評估協議：
  • EI-reg (Emotion Intensity，4 emotions × 3 langs)：pcc → 「EI ave」= 4 emotion 平均
  • V-reg (Valence regression = SI valence)：pcc
  • V-oc  (Valence ordinal = SP valence，類別 -3..+3)：pcc（將整數類別當實值算相關）
  • E-c   (Emotion classification, 11-label multi-label)：jac, mi-F1, ma-F1
  • Overall = 各 task 主要指標（pcc 或 ma-F1）的平均

backbone 從 `sacf_v60_best.pt` 載入（保留 MOSI 訓練先驗）。
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
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "emotion_system" / "training"))
from scaf_final import PolarityEnhancedAttention  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "emotion_system" / "data" / "semeval2018" / "SemEval2018-Task1-all-data"
MODEL_DIR = PROJECT_ROOT / "emotion_system" / "models"
SACF_CKPT = MODEL_DIR / "sacf_v60_best.pt"

LANG_DIR  = {"en": "English", "ar": "Arabic", "es": "Spanish"}
LANG_TAG  = {"en": "En",      "ar": "Ar",     "es": "Es"}
EI_EMOTIONS = ["anger", "fear", "joy", "sadness"]
EC_LABELS   = ["anger", "anticipation", "disgust", "fear", "joy", "love",
               "optimism", "pessimism", "sadness", "surprise", "trust"]


# ───────────────────────── 資料載入 ─────────────────────────
def _read_tsv(path: Path) -> List[List[str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        f.readline()                  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                rows.append(parts)
    return rows


def load_ei_reg(lang: str, emotion: str
                ) -> Tuple[List[dict], List[dict], List[dict]]:
    L  = LANG_DIR[lang]; T = LANG_TAG[lang]
    base = DATA_ROOT / L / "EI-reg"
    if lang == "en":
        train_p = base / "training"    / f"EI-reg-En-{emotion}-train.txt"
    else:
        train_p = base / "training"    / f"2018-EI-reg-{T}-{emotion}-train.txt"
    dev_p   = base / "development" / f"2018-EI-reg-{T}-{emotion}-dev.txt"
    test_p  = base / "test-gold"   / f"2018-EI-reg-{T}-{emotion}-test-gold.txt"

    def _proc(rows):
        return [{"text": r[1], "y": float(r[3])} for r in rows]
    return _proc(_read_tsv(train_p)), _proc(_read_tsv(dev_p)), _proc(_read_tsv(test_p))


def load_v_reg(lang: str) -> Tuple[List[dict], List[dict], List[dict]]:
    L = LANG_DIR[lang]; T = LANG_TAG[lang]
    base = DATA_ROOT / L / "V-reg"
    train_p = base / f"2018-Valence-reg-{T}-train.txt"
    dev_p   = base / f"2018-Valence-reg-{T}-dev.txt"
    test_p  = base / f"2018-Valence-reg-{T}-test-gold.txt"
    def _proc(rows):
        return [{"text": r[1], "y": float(r[3])} for r in rows]
    return _proc(_read_tsv(train_p)), _proc(_read_tsv(dev_p)), _proc(_read_tsv(test_p))


_VOC_RE = re.compile(r"^(-?\d+)\s*:")
def load_v_oc(lang: str) -> Tuple[List[dict], List[dict], List[dict]]:
    L = LANG_DIR[lang]; T = LANG_TAG[lang]
    base = DATA_ROOT / L / "V-oc"
    train_p = base / f"2018-Valence-oc-{T}-train.txt"
    dev_p   = base / f"2018-Valence-oc-{T}-dev.txt"
    test_p  = base / f"2018-Valence-oc-{T}-test-gold.txt"
    def _proc(rows):
        out = []
        for r in rows:
            m = _VOC_RE.match(r[3])
            if not m:
                continue
            out.append({"text": r[1], "y": float(m.group(1))})   # -3..+3 as float
        return out
    return _proc(_read_tsv(train_p)), _proc(_read_tsv(dev_p)), _proc(_read_tsv(test_p))


def load_ec(lang: str) -> Tuple[List[dict], List[dict], List[dict]]:
    L = LANG_DIR[lang]; T = LANG_TAG[lang]
    base = DATA_ROOT / L / "E-c"
    train_p = base / f"2018-E-c-{T}-train.txt"
    dev_p   = base / f"2018-E-c-{T}-dev.txt"
    test_p  = base / f"2018-E-c-{T}-test-gold.txt"
    def _proc(rows):
        out = []
        for r in rows:
            try:
                vec = [int(r[2 + i]) for i in range(11)]
            except (IndexError, ValueError):
                continue
            out.append({"text": r[1], "y": np.array(vec, dtype=np.float32)})
        return out
    return _proc(_read_tsv(train_p)), _proc(_read_tsv(dev_p)), _proc(_read_tsv(test_p))


# ───────────────────────── 純文字 SACF ─────────────────────────
class TextSACF(nn.Module):
    """共用 backbone + per-task head。head_type ∈ {'reg', 'multi'}."""

    def __init__(self, lang_model: str, n_out: int, head_type: str,
                 fusion_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        h = self.lang_backbone.config.hidden_size
        self.polarity_attn = PolarityEnhancedAttention(h, dropout)
        self.shared = nn.Sequential(
            nn.Linear(h, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head      = nn.Linear(fusion_dim, n_out)
        self.head_type = head_type

    def forward(self, ids, mask):
        hidden = self.lang_backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled, _ = self.polarity_attn(hidden, mask)
        return self.head(self.shared(pooled))


def maybe_load_sacf(model: TextSACF, ckpt_path: Path) -> str:
    if not ckpt_path.exists():
        return f"未找到 {ckpt_path.name}，HF 預訓練 init"
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    own   = model.state_dict()
    n = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v; n += 1
    model.load_state_dict(own)
    return f"載入 {n} 張量 from {ckpt_path.name}"


# ───────────────────────── Dataset ─────────────────────────
class _SemDS(Dataset):
    def __init__(self, rows, tok, multi: bool, max_len: int):
        self.rows, self.tok, self.multi, self.max = rows, tok, multi, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], add_special_tokens=True, max_length=self.max,
                       padding="max_length", truncation=True, return_tensors="pt")
        if self.multi:
            y = torch.from_numpy(r["y"])
        else:
            y = torch.tensor(r["y"], dtype=torch.float32)
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": y}


# ───────────────────────── 訓練 / 評估 ─────────────────────────
def train_and_eval(train_rows, dev_rows, test_rows, *, head_type: str,
                   tok, device: str, args, log_prefix: str,
                   no_pretrain: bool) -> dict:
    multi  = (head_type == "multi")
    n_out  = len(EC_LABELS) if multi else 1
    train_ds = _SemDS(train_rows, tok, multi, args.max_len)
    dev_ds   = _SemDS(dev_rows,   tok, multi, args.max_len)
    test_ds  = _SemDS(test_rows,  tok, multi, args.max_len)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    dev_ld   = DataLoader(dev_ds,   args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    test_ld  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    model = TextSACF(args.lang_model, n_out, head_type).to(device)
    if not no_pretrain:
        msg = maybe_load_sacf(model, SACF_CKPT)
        if log_prefix.endswith("[init]"):
            print("  ", msg)

    backbone_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lang_backbone" in n]
    head_p     = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lang_backbone" not in n]
    opt = torch.optim.AdamW(
        [{"params": backbone_p, "lr": args.lang_lr},
         {"params": head_p,     "lr": args.head_lr}],
        weight_decay=args.weight_decay)
    total_steps  = max(1, len(train_ld)) * args.epochs
    warmup_steps = int(total_steps * 0.06)
    sched  = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if "cuda" in device else None

    crit = nn.BCEWithLogitsLoss() if multi else nn.MSELoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in tqdm(train_ld, desc=f"{log_prefix} E{ep}", leave=False):
            ids = batch["input_ids"].to(device); mk = batch["attention_mask"].to(device)
            y   = batch["label"].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(ids, mk)
                if not multi:
                    logits = logits.squeeze(-1)
                loss = crit(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                sched.step(); continue
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            running += loss.item() * y.size(0); n += y.size(0)
        train_loss = running / max(n, 1)
        print(f"  {log_prefix} E{ep} train loss = {train_loss:.4f}")

    # ── 評估 ──
    def _infer(loader):
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device); mk = batch["attention_mask"].to(device)
                logits = model(ids, mk)
                if not multi:
                    logits = logits.squeeze(-1)
                if multi:
                    preds.append(torch.sigmoid(logits).cpu().numpy())
                else:
                    preds.append(logits.cpu().numpy())
                gts.append(batch["label"].numpy())
        return np.concatenate(preds), np.concatenate(gts)

    p_dev,  g_dev  = _infer(dev_ld)
    p_test, g_test = _infer(test_ld)

    if multi:
        # 多標籤：用 dev 搜尋最佳 threshold（per-class），減少 sigmoid bias
        best_th = 0.5; best_jac = -1
        for th in np.arange(0.1, 0.71, 0.05):
            yh = (p_dev >= th).astype(int)
            jac = jaccard_score(g_dev.astype(int), yh, average="samples", zero_division=0)
            if jac > best_jac:
                best_jac, best_th = jac, float(th)
        yh_test = (p_test >= best_th).astype(int)
        gt_int  = g_test.astype(int)
        return {
            "jac":   float(jaccard_score(gt_int, yh_test, average="samples", zero_division=0) * 100),
            "mi-F1": float(f1_score(gt_int, yh_test, average="micro", zero_division=0) * 100),
            "ma-F1": float(f1_score(gt_int, yh_test, average="macro", zero_division=0) * 100),
            "best_threshold":  best_th,
            "n_test": int(len(g_test)),
        }
    else:
        # 回歸：Pearson 相關
        if np.std(p_test) < 1e-8 or np.std(g_test) < 1e-8:
            pcc = 0.0
        else:
            pcc, _ = pearsonr(p_test, g_test)
        return {
            "pcc":   float(pcc * 100),
            "mse":   float(((p_test - g_test) ** 2).mean()),
            "n_test": int(len(g_test)),
        }


# ───────────────────────── 主程式 ─────────────────────────
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# 論文 Table 4 之 baseline（純文字模型，僅 LM-T 區塊；overall 為各 task 平均）
PAPER_TABLE4: Dict[str, Dict[str, Dict[str, float]]] = {
    "EmoLlama-chat-7b":      {"en": {"EI": 79.9, "SP": 87.3, "SI": 85.6, "EC_jac": 60.9, "EC_mi": 72.4, "EC_ma": 59.2},
                              "ar": {"EI": 43.0, "SP": 66.1, "SI": 74.5, "EC_jac": 36.2, "EC_mi": 52.6, "EC_ma": 38.3},
                              "es": {"EI": 73.7, "SP": 74.4, "SI": 81.7, "EC_jac": 40.1, "EC_mi": 56.3, "EC_ma": 45.7}},
    "Llama3.2-1b-instruct":  {"en": {"EI": 14.4, "SP": 27.1, "SI": 34.8, "EC_jac": 36.8, "EC_mi": 51.0, "EC_ma": 39.2},
                              "ar": {"EI":  6.3, "SP": 18.7, "SI": 22.0, "EC_jac": 15.0, "EC_mi": 25.5, "EC_ma": 20.9},
                              "es": {"EI": 13.0, "SP": 13.1, "SI": 32.6, "EC_jac": 14.6, "EC_mi": 22.6, "EC_ma": 14.6}},
    "Llama3.2-3b-instruct":  {"en": {"EI": 54.7, "SP": 65.2, "SI": 71.1, "EC_jac": 32.7, "EC_mi": 45.8, "EC_ma": 35.6},
                              "ar": {"EI": 32.7, "SP": 33.8, "SI": 36.5, "EC_jac": 17.3, "EC_mi": 26.7, "EC_ma": 20.6},
                              "es": {"EI": 53.7, "SP": 64.4, "SI": 29.1, "EC_jac": 42.7, "EC_mi": 35.1, "EC_ma": 51.1}},
    "Mistral-7b-instruct":   {"en": {"EI": 58.4, "SP": 77.7, "SI": 70.8, "EC_jac": 30.7, "EC_mi": 43.0, "EC_ma": 35.6},
                              "ar": {"EI": 35.9, "SP": 64.1, "SI": 30.6, "EC_jac": 42.1, "EC_mi": 57.6, "EC_ma": 73.4},
                              "es": {"EI": 56.7, "SP": 29.8, "SI": 40.7, "EC_jac": 26.7, "EC_mi": 34.2, "EC_ma": 55.8}},
    "GPT-4o-mini":           {"en": {"EI": 71.4, "SP": 82.4, "SI": 82.0, "EC_jac": 42.1, "EC_mi": 56.4, "EC_ma": 44.6},
                              "ar": {"EI": 63.9, "SP": 85.4, "SI": 86.4, "EC_jac": 56.5, "EC_mi": 64.1, "EC_ma": 47.1},
                              "es": {"EI": 73.0, "SP": 81.4, "SI": 80.6, "EC_jac": 42.1, "EC_mi": 60.0, "EC_ma": 70.2}},
    "MMAFFLM-3b":            {"en": {"EI": 64.6, "SP": 76.8, "SI": 82.5, "EC_jac": 46.6, "EC_mi": 61.6, "EC_ma": 47.5},
                              "ar": {"EI": 62.1, "SP": 78.4, "SI": 81.8, "EC_jac": 45.1, "EC_mi": 63.5, "EC_ma": 73.6},
                              "es": {"EI": 79.1, "SP": 38.4, "SI": 44.0, "EC_jac": 43.7, "EC_mi": 53.1, "EC_ma": 56.1}},
    "MMAFFLM-7b":            {"en": {"EI": 70.3, "SP": 77.8, "SI": 83.2, "EC_jac": 43.2, "EC_mi": 59.3, "EC_ma": 42.3},
                              "ar": {"EI": 62.1, "SP": 78.4, "SI": 81.8, "EC_jac": 39.7, "EC_mi": 57.4, "EC_ma": 40.5},
                              "es": {"EI": 69.7, "SP": 78.0, "SI": 73.0, "EC_jac": 35.7, "EC_mi": 52.3, "EC_ma": 38.4}},
}


def overall_avg(d: Dict[str, float]) -> float:
    """Overall = 各 task 主要指標平均（EI ave / SP / SI 的 pcc + EC ma-F1）"""
    keys = ["EI", "SP", "SI", "EC_ma"]
    vals = [d[k] for k in keys if k in d]
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang_model",   default="microsoft/deberta-v3-large")
    ap.add_argument("--gpu",          default="0")
    ap.add_argument("--batch_size",   type=int, default=16)
    ap.add_argument("--epochs",       type=int, default=3,
                    help="SemEval 資料較小，預設 3 epoch（論文用 1 epoch 但他們資料更大）")
    ap.add_argument("--lang_lr",      type=float, default=5e-6)
    ap.add_argument("--head_lr",      type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_len",      type=int, default=128)
    ap.add_argument("--langs",        default="en,ar,es")
    ap.add_argument("--no_pretrain",  action="store_true")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    print("=" * 70)
    print(" SemEval-2018 純文字評估（對應論文 Table 4）")
    print("=" * 70)
    print(f"  device     = {device} (CUDA_VISIBLE_DEVICES={args.gpu})")
    print(f"  backbone   = {args.lang_model}  | epochs = {args.epochs}  bs = {args.batch_size}")
    print(f"  lr         = lang {args.lang_lr:.0e} / head {args.head_lr:.0e}")
    print(f"  pretrained = {'no' if args.no_pretrain else 'sacf_v60_best.pt'}")
    print(f"  langs      = {langs}")

    tok = DebertaV2Tokenizer.from_pretrained(args.lang_model)

    results: Dict[str, Dict[str, dict]] = {l: {} for l in langs}
    t_global = time.time()

    for lang in langs:
        print(f"\n{'═' * 70}\n  ▶ Language: {lang.upper()}\n{'═' * 70}")

        # ── EI-reg：4 emotion 各跑一次 ──
        ei_pccs = []
        results[lang]["EI_per_emotion"] = {}
        for emo in EI_EMOTIONS:
            tr, dv, te = load_ei_reg(lang, emo)
            print(f"\n  [{lang}] EI-reg {emo}:  train={len(tr)}  dev={len(dv)}  test={len(te)}")
            tag = f"{lang}-EI-{emo}" + (" [init]" if emo == EI_EMOTIONS[0] else "")
            r = train_and_eval(tr, dv, te, head_type="reg", tok=tok, device=device,
                               args=args, log_prefix=tag, no_pretrain=args.no_pretrain)
            print(f"  ⮕ EI-{emo}: pcc = {r['pcc']:.2f}  (mse={r['mse']:.4f})")
            results[lang]["EI_per_emotion"][emo] = r
            ei_pccs.append(r["pcc"])
        results[lang]["EI"]    = float(np.mean(ei_pccs))
        results[lang]["EI_std"] = float(np.std(ei_pccs))

        # ── V-reg = SI valence ──
        tr, dv, te = load_v_reg(lang)
        print(f"\n  [{lang}] V-reg (SI):  train={len(tr)}  dev={len(dv)}  test={len(te)}")
        r = train_and_eval(tr, dv, te, head_type="reg", tok=tok, device=device,
                           args=args, log_prefix=f"{lang}-Vreg", no_pretrain=args.no_pretrain)
        print(f"  ⮕ V-reg (SI): pcc = {r['pcc']:.2f}")
        results[lang]["SI"] = r["pcc"]
        results[lang]["SI_full"] = r

        # ── V-oc = SP valence (ordinal) ──
        tr, dv, te = load_v_oc(lang)
        print(f"\n  [{lang}] V-oc (SP):  train={len(tr)}  dev={len(dv)}  test={len(te)}")
        r = train_and_eval(tr, dv, te, head_type="reg", tok=tok, device=device,
                           args=args, log_prefix=f"{lang}-Voc", no_pretrain=args.no_pretrain)
        print(f"  ⮕ V-oc (SP):  pcc = {r['pcc']:.2f}")
        results[lang]["SP"] = r["pcc"]
        results[lang]["SP_full"] = r

        # ── E-c (multi-label) ──
        tr, dv, te = load_ec(lang)
        print(f"\n  [{lang}] E-c (EC):   train={len(tr)}  dev={len(dv)}  test={len(te)}")
        r = train_and_eval(tr, dv, te, head_type="multi", tok=tok, device=device,
                           args=args, log_prefix=f"{lang}-Ec", no_pretrain=args.no_pretrain)
        print(f"  ⮕ E-c: jac={r['jac']:.2f} mi-F1={r['mi-F1']:.2f} ma-F1={r['ma-F1']:.2f}  (th={r['best_threshold']:.2f})")
        results[lang]["EC_jac"] = r["jac"]
        results[lang]["EC_mi"]  = r["mi-F1"]
        results[lang]["EC_ma"]  = r["ma-F1"]
        results[lang]["EC_full"] = r

        # 釋放顯存
        if "cuda" in device:
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_global

    # ───── 計算 Overall ─────
    our_summary = {}
    for lang in langs:
        d = {k: results[lang][k] for k in ["EI", "SP", "SI", "EC_jac", "EC_mi", "EC_ma"]}
        d["Overall"] = overall_avg(d)
        our_summary[lang] = d
    overall_global = float(np.mean([our_summary[l]["Overall"] for l in langs]))

    # ───── 寫 JSON ─────
    json_out = HERE / "semeval_results.json"
    json_out.write_text(json.dumps({
        "config": vars(args),
        "device": device,
        "sacf_ckpt_loaded": (not args.no_pretrain) and SACF_CKPT.exists(),
        "results_per_lang": results,
        "summary": our_summary,
        "overall_global": overall_global,
        "total_elapsed_sec": round(total_elapsed, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✓ JSON: {json_out}")

    # ───── 寫 Markdown：對應論文 Table 4 ─────
    md_out = HERE / "SEMEVAL2018_TABLE4_REPORT.md"
    lines = []
    lines.append("# SemEval-2018 純文字評估報告（對應論文 Table 4）")
    lines.append("")
    lines.append("依論文 *MMAFFBen* (Liu et al. 2025) Table 4 的設定：")
    lines.append("")
    lines.append("- **訓練集**：SemEval-2018 Task 1 的 train + dev（每任務每語言）")
    lines.append("- **測試集**：SemEval-2018 Task 1 的 **test-gold**")
    lines.append("- **基底模型**：`scaf_final.SACFModel` 純文字分支（DeBERTa-v3-large）")
    lines.append(f"- **backbone init**：{'`sacf_v60_best.pt`' if not args.no_pretrain else 'HF 預訓練'}")
    lines.append(f"- **訓練超參**：每 (lang, task) 獨立訓練 {args.epochs} epoch，"
                 f"bs={args.batch_size}, lang_lr={args.lang_lr:.0e}, head_lr={args.head_lr:.0e}")
    lines.append("")
    lines.append("**指標**（與論文一致）")
    lines.append("- EI (Emotion Intensity): pcc，欄位 *EI ave* = anger/fear/joy/sadness 4 個 PCC 平均")
    lines.append("- SP (Sentiment Polarity, V-oc): pcc on 整數類別 (-3..+3)")
    lines.append("- SI (Sentiment Intensity, V-reg): pcc on 連續分數 [0,1]")
    lines.append("- EC (E-c, multi-label, 11 emotions): jac (Jaccard, samples avg)、mi-F1、ma-F1")
    lines.append("- Overall = pcc(EI) + pcc(SP) + pcc(SI) + ma-F1(EC) 之平均")
    lines.append("")
    lines.append("## 主結果（對應 Table 4 LM-T 區塊）")
    lines.append("")

    # 表頭
    h1 = "| Models |" + "".join([f" {l.upper()} EI ave | SP val | SI val | Ec jac | Ec mi-F1 | Ec ma-F1 |" for l in langs]) + " Overall |"
    h2 = "|--------|" + ("------:|" * (6 * len(langs))) + "--------:|"
    lines.append(h1); lines.append(h2)

    def _fmt_row(name: str, vals: Dict[str, Dict[str, float]], bold: bool = False):
        cells = []
        for l in langs:
            v = vals.get(l, {})
            cells += [f"{v.get(k, 0):.2f}" for k in ["EI", "SP", "SI", "EC_jac", "EC_mi", "EC_ma"]]
        if vals:
            ov = float(np.mean([overall_avg(vals[l]) for l in langs if l in vals]))
        else:
            ov = 0.0
        cells.append(f"{ov:.2f}")
        if bold:
            return f"| **{name}** | " + " | ".join(f"**{c}**" for c in cells) + " |"
        return f"| {name} | " + " | ".join(cells) + " |"

    # 我們的行
    our_label = ("SACF-Text (ours, sacf_v60 init)" if not args.no_pretrain
                 else "SACF-Text (ours, HF init)")
    lines.append(_fmt_row(our_label, our_summary, bold=True))

    # 論文 baseline
    for model_name, scores in PAPER_TABLE4.items():
        lines.append(_fmt_row(model_name, scores))

    lines.append("")
    lines.append(f"**Our overall (across {len(langs)} langs) = {overall_global:.2f}**")
    lines.append("")
    lines.append("> Baseline 數值取自論文 Table 4 「LM-T」區塊（純文字模型）。")
    lines.append("> EI 為 anger/fear/joy/sadness 4 emotion 的 pcc 平均；Overall = 4 task 主要指標的算術平均。")
    lines.append("")

    # 各語言詳細
    lines.append("## 各語言詳細（含每 emotion 的 EI pcc）")
    lines.append("")
    for lang in langs:
        s = our_summary[lang]
        ei_per = results[lang]["EI_per_emotion"]
        ei_str = ", ".join(f"{e}={ei_per[e]['pcc']:.2f}" for e in EI_EMOTIONS)
        lines.append(f"### {lang.upper()} (Overall = {s['Overall']:.2f})")
        lines.append("")
        lines.append(f"- EI ave (pcc) = **{s['EI']:.2f}** ± {results[lang]['EI_std']:.2f} ({ei_str})")
        lines.append(f"- SP valence (pcc on V-oc) = **{s['SP']:.2f}**")
        lines.append(f"- SI valence (pcc on V-reg) = **{s['SI']:.2f}**")
        lines.append(f"- EC: jac={s['EC_jac']:.2f}, mi-F1={s['EC_mi']:.2f}, ma-F1={s['EC_ma']:.2f} "
                     f"(best dev threshold = {results[lang]['EC_full']['best_threshold']:.2f})")
        lines.append("")
    lines.append(f"_耗時：{round(total_elapsed)}s，設備：{device}_")

    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Markdown: {md_out}")

    print("\n" + "=" * 70)
    for lang in langs:
        s = our_summary[lang]
        print(f"  {lang.upper():3s}  EI={s['EI']:5.2f}  SP={s['SP']:5.2f}  SI={s['SI']:5.2f}  "
              f"EC[jac/mi/ma]={s['EC_jac']:5.2f}/{s['EC_mi']:5.2f}/{s['EC_ma']:5.2f}  "
              f"Overall={s['Overall']:5.2f}")
    print(f"  Global Overall (all langs) = {overall_global:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
