"""Self-healing watchdog for the MMAFFIn comparison pipeline.

Runs independently under setsid; wakes every 5 min and checks:
  1. Is the orchestrator process (by PID file) alive?
  2. If dead AND no figs/*.png yet, determine which stages still need running
     (by checking for history_{ver}.json artifacts) and relaunch
     run_comparison.py with the correct skip flags.
  3. If figs/*.png exist, exit cleanly (training complete).
  4. Writes watchdog.log.

Detection of stage completion relies on `history_v60_baseline.json` and
`history_v60_mmaffin.json` appearing in emotion_system/models/. If baseline
history exists, skip baseline. If mmaffin history exists, skip pretrained.

Max 10 restarts to avoid infinite loops on systemic failure.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent.parent
MODEL_DIR   = PROJECT_ROOT / "emotion_system" / "models"
LOG_DIR     = EXP_DIR / "logs"
FIGS_DIR    = EXP_DIR / "figs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

PID_FILE    = LOG_DIR / "orchestrator.pid"
WATCH_LOG   = LOG_DIR / "watchdog.log"

POLL_SEC     = 300        # 5 min
MAX_RESTARTS = 10


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(WATCH_LOG, "a") as f:
        f.write(line + "\n")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def read_pid():
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def write_pid(pid):
    PID_FILE.write_text(str(pid))


def training_done():
    if not FIGS_DIR.exists():
        return False
    pngs = list(FIGS_DIR.glob("*.png"))
    return len(pngs) >= 3      # at least a few figs produced = done


def missing_stages():
    base_hist = MODEL_DIR / "history_v60_baseline.json"
    pre_hist  = MODEL_DIR / "history_v60_mmaffin.json"
    return {
        "baseline":    not base_hist.exists(),
        "pretrained":  not pre_hist.exists(),
        "figs":        not training_done(),
    }


def start_orchestrator(restart_n):
    stages = missing_stages()
    skip_flags = ["--skip_pretrain"]  # pretrain always done by this point
    if not stages["baseline"]:
        skip_flags.append("--skip_baseline")
    if not stages["pretrained"]:
        skip_flags.append("--skip_pretrained")

    if not stages["baseline"] and not stages["pretrained"]:
        log("Both MOSI runs already complete — running viz only.")
        viz_cmd = [
            sys.executable,
            str(EXP_DIR / "visualize_comparison.py"),
        ]
        subprocess.Popen(viz_cmd, cwd=str(PROJECT_ROOT),
                         stdout=open(LOG_DIR / "viz.log", "a"),
                         stderr=subprocess.STDOUT,
                         start_new_session=True)
        return None

    orch_log = open(LOG_DIR / "orchestrator.log", "a")
    orch_log.write(f"\n\n### watchdog restart #{restart_n} at {time.ctime()} "
                   f"(skip: {skip_flags})\n\n")
    orch_log.flush()

    cmd = [
        sys.executable, "-u",
        str(EXP_DIR / "run_comparison.py"),
        "--gpu", "0",
    ] + skip_flags

    log(f"Launching orchestrator restart#{restart_n}: {' '.join(cmd)}")
    p = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=orch_log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    write_pid(p.pid)
    log(f"Started orchestrator PID={p.pid}, PGID={os.getpgid(p.pid)}")
    return p.pid


def main():
    log(f"watchdog started, PID={os.getpid()}")
    log(f"polling every {POLL_SEC}s, max_restarts={MAX_RESTARTS}")

    restarts = 0
    while True:
        if training_done():
            log("Training complete (figs/*.png exists). Exiting watchdog.")
            return

        pid = read_pid()
        alive = pid_alive(pid) if pid else False

        stages = missing_stages()
        log(f"poll: pid={pid} alive={alive} missing={stages}")

        if not alive:
            if restarts >= MAX_RESTARTS:
                log(f"MAX_RESTARTS={MAX_RESTARTS} reached, giving up.")
                return
            restarts += 1
            new_pid = start_orchestrator(restarts)
            if new_pid is None:
                # viz-only launched; wait a bit then recheck
                time.sleep(60)
                continue
        else:
            log(f"  -> alive; no action.")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e!r}")
        raise
