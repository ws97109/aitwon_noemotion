"""Orchestrator: run baseline + MMAFFIn-pretrained comparison.

Three stages:
  1. (optional) Pretrain DeBERTa backbone on MMS+XED if not already cached.
  2. Baseline     : scaf_final_mmaffin.py  --version v60_baseline
  3. Pretrained   : scaf_final_mmaffin.py  --version v60_mmaffin --pretrained_backbone ...

Each stage can be individually skipped via CLI flags.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent.parent
MODEL_DIR   = PROJECT_ROOT / "emotion_system" / "models"
BACKBONE_PT = MODEL_DIR / "mmaffin_pretrain_backbone.pt"

PRETRAIN_SCRIPT = EXP_DIR / "pretrain_backbone.py"
TRAIN_SCRIPT    = EXP_DIR / "scaf_final_mmaffin.py"


def run(cmd, tag, log_path):
    print(f"\n{'#'*78}\n# [{tag}] {' '.join(cmd)}\n{'#'*78}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# cmd: {' '.join(cmd)}\n# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, cwd=str(PROJECT_ROOT))
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            lf.write(line); lf.flush()
        proc.wait()
    dur = (time.time() - t0) / 60
    print(f"\n# [{tag}] done in {dur:.1f} min (exit={proc.returncode})", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"[{tag}] failed, see {log_path}")
    return dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",              default="0")
    ap.add_argument("--skip_pretrain",    action="store_true",
                    help="reuse existing mmaffin_pretrain_backbone.pt")
    ap.add_argument("--skip_baseline",    action="store_true")
    ap.add_argument("--skip_pretrained",  action="store_true")
    ap.add_argument("--pretrain_epochs",  type=int, default=3)
    ap.add_argument("--pretrain_batch",   type=int, default=32)
    ap.add_argument("--pretrain_lr",      type=float, default=2e-5)
    ap.add_argument("--baseline_version", default="v60_baseline")
    ap.add_argument("--pretrained_version", default="v60_mmaffin")
    args = ap.parse_args()

    EXP_DIR.mkdir(exist_ok=True, parents=True)
    log_dir = EXP_DIR / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    times = {}

    # 1) Pretrain
    if args.skip_pretrain:
        if not BACKBONE_PT.exists():
            print(f"[WARN] --skip_pretrain but {BACKBONE_PT} missing. Pretrained run will fail.")
        else:
            print(f"[skip] reuse {BACKBONE_PT}")
    else:
        cmd = [
            sys.executable, str(PRETRAIN_SCRIPT),
            "--gpu",         args.gpu,
            "--epochs",      str(args.pretrain_epochs),
            "--batch_size",  str(args.pretrain_batch),
            "--lr",          str(args.pretrain_lr),
        ]
        times["pretrain"] = run(cmd, "pretrain", log_dir / "pretrain.log")

    # 2) Baseline MOSI run
    if args.skip_baseline:
        print("[skip] baseline")
    else:
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--gpu",     args.gpu,
            "--version", args.baseline_version,
        ]
        times["baseline"] = run(cmd, "baseline", log_dir / "baseline.log")

    # 3) Pretrained MOSI run
    if args.skip_pretrained:
        print("[skip] pretrained")
    else:
        if not BACKBONE_PT.exists():
            raise FileNotFoundError(f"missing {BACKBONE_PT}; run pretrain stage first")
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--gpu",                   args.gpu,
            "--version",               args.pretrained_version,
            "--pretrained_backbone",   str(BACKBONE_PT),
        ]
        times["pretrained"] = run(cmd, "pretrained", log_dir / "pretrained.log")

    print("\n" + "=" * 72)
    print("All stages complete.")
    for k, v in times.items():
        print(f"  {k:12s}  {v:6.1f} min")
    print("=" * 72)
    print(f"\nNext: python emotion_system/training/mmaffin_exp/visualize_comparison.py "
          f"--baseline {args.baseline_version} --pretrained {args.pretrained_version}")


if __name__ == "__main__":
    main()
