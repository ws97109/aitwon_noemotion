"""Prepare MMAFFIn text subset (MMS + XED) for sentiment pretraining.

Produces a unified (text, label) corpus with 3-class sentiment:
  0 = negative, 1 = neutral, 2 = positive

Sources (all languages kept):
  - MMS  (train+val): direct 3-class (-1/0/+1) -> 0/1/2
  - XED  (train+val): multi-label Ekman-like 8 emotions -> map by valence majority

XED -> valence mapping:
  positive : 1.joy, 7.anticipation, 8.trust
  negative : 2.sadness, 3.anger, 4.fear, 6.disgust
  neutral  : 0.neutral, 5.surprise   (surprise is ambivalent)

Multi-label reconciliation: sum the (pos, neg) votes; higher wins; tie -> neutral.

Output: one pickle file with {'train': [...], 'val': [...]}, each entry is
{'text': str, 'label': int in {0,1,2}, 'source': 'MMS'|'XED', 'raw': str}
"""
import json
import re
import pickle
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent.parent / "data" / "mmaffin_text" / "texts"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "pretrain_corpus.pkl"

TEXT_RE = re.compile(r"Text:\s*(.*)", flags=re.DOTALL)

XED_POS = {1, 7, 8}
XED_NEG = {2, 3, 4, 6}


def parse_text(instruction: str) -> str:
    m = TEXT_RE.search(instruction)
    return m.group(1).strip() if m else ""


def parse_mms_label(out: str):
    s = out.strip().lower()
    if s.startswith("-1") or "negative" in s:
        return 0
    if s.startswith("1") or "positive" in s:
        return 2
    if s.startswith("0") or "neutral" in s:
        return 1
    return None


def parse_xed_label(out: str):
    """'1. joy, 8. trust' -> 2 (pos); '3. anger, 6. disgust' -> 0 (neg); '0. neutral' -> 1."""
    nums = [int(x) for x in re.findall(r"(?<!\d)(\d+)\.\s", out)]
    if not nums:
        return None
    pos = sum(1 for n in nums if n in XED_POS)
    neg = sum(1 for n in nums if n in XED_NEG)
    if pos == 0 and neg == 0:
        return 1                       # neutral / surprise only
    if pos > neg:
        return 2
    if neg > pos:
        return 0
    return 1                           # tie -> neutral


def load_file(fn: str, parser):
    with open(BASE / fn, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for ex in raw:
        t = parse_text(ex["instruction"])
        lbl = parser(ex["output"])
        if not t or lbl is None:
            continue
        src = "MMS" if fn.startswith("MMS") else "XED"
        out.append({"text": t, "label": lbl, "source": src, "raw": ex["output"]})
    return out


def main():
    print(f"Reading from {BASE}")
    train, val = [], []
    train += load_file("MMS_train.json", parse_mms_label)
    val   += load_file("MMS_val.json",   parse_mms_label)
    train += load_file("XED_train.json", parse_xed_label)
    val   += load_file("XED_val.json",   parse_xed_label)

    def stats(name, rows):
        by_src = Counter(r["source"] for r in rows)
        by_lbl = Counter(r["label"]  for r in rows)
        lens   = [len(r["text"]) for r in rows]
        print(f"\n[{name}] n={len(rows)}")
        print(f"  by source: {dict(by_src)}")
        print(f"  by label (0=neg,1=neu,2=pos): {dict(by_lbl)}")
        print(f"  char len mean/med/max: {sum(lens)/len(lens):.0f} / {sorted(lens)[len(lens)//2]} / {max(lens)}")

    stats("train", train)
    stats("val",   val)

    with open(OUT_FILE, "wb") as f:
        pickle.dump({"train": train, "val": val}, f)
    print(f"\nSaved -> {OUT_FILE}  (train={len(train)}, val={len(val)})")


if __name__ == "__main__":
    main()
