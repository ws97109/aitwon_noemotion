"""Quick sanity check:
  1. Load the unified pretrain corpus, feed 16 samples through DeBERTa+head, ensure gradient flows.
  2. Instantiate SACFModel, load a dummy pretrained backbone state, ensure forward pass works.
"""
import os, sys, pickle, torch
from pathlib import Path
from transformers import DebertaV2Tokenizer, AutoModel

HERE  = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DATA = HERE / "data" / "pretrain_corpus.pkl"
print("Loading corpus...")
with open(DATA, "rb") as f:
    corp = pickle.load(f)
print(f"  train={len(corp['train'])} val={len(corp['val'])}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

MODEL_NAME = "microsoft/deberta-v3-large"
print(f"Loading tokenizer + model ({MODEL_NAME})...")
tok   = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

from pretrain_backbone import BackbonePlusHead, SentimentCorpus
model = BackbonePlusHead(MODEL_NAME, n_classes=3, dropout=0.1).to(device)
print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("Forward + backward on 16 samples...")
ds = SentimentCorpus(corp["train"][:16], tok, max_len=80)
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=8, shuffle=False)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
for b in loader:
    ids  = b["input_ids"].to(device)
    mask = b["attention_mask"].to(device)
    y    = b["label"].to(device)
    opt.zero_grad()
    logits = model(ids, mask)
    loss   = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()
    print(f"  batch loss={loss.item():.4f}  acc={(logits.argmax(-1)==y).float().mean()*100:.1f}%")

print("\nSaving dummy backbone checkpoint...")
out = HERE / "data" / "_smoke_backbone.pt"
torch.save({
    "backbone_state_dict": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
    "model_name":          MODEL_NAME,
    "best_val_acc":        -1.0,
    "epochs":              0,
}, out)
print(f"  -> {out}")

# Now load into SACFModel
print("\nLoading into SACFModel...")
# temporarily add training dir to path so we can import the module name
sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location("scaf_final_mmaffin",
                                              HERE / "scaf_final_mmaffin.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sacf = m.SACFModel(dropout=0.1).to(device)
ckpt  = torch.load(str(out), map_location="cpu")
state = ckpt["backbone_state_dict"]
missing, unexpected = sacf.lang_backbone.load_state_dict(state, strict=False)
print(f"  missing={len(missing)} unexpected={len(unexpected)}")
assert len(unexpected) == 0, f"unexpected keys: {unexpected[:5]}"

# Forward pass on dummy multi-modal inputs
print("\nSACFModel forward pass...")
B = 4; L = 80
ids  = torch.randint(0, 10000, (B, L)).to(device)
mask = torch.ones(B, L).to(device)
aud  = torch.randn(B, 375, 5).to(device)
am   = torch.ones(B, 375).to(device)
vis  = torch.randn(B, 500, 20).to(device)
vm   = torch.ones(B, 500).to(device)
with torch.no_grad():
    l7, l2, reg = sacf(ids, mask, aud, am, vis, vm)
print(f"  l7.shape={l7.shape}  l2.shape={l2.shape}  reg.shape={reg.shape}")
assert l7.shape == (B, 7)
assert l2.shape == (B, 2)
assert reg.shape == (B,)

# Cleanup
out.unlink()
print("\n✅ Smoke test passed.")
