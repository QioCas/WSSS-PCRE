#!/usr/bin/env python3
"""Automated GPU Queue Supervisor for VOC SAM-PCRE v2 Training.

Monitors GPU 0 status and free VRAM. Once GPU is available and free VRAM >= MIN_FREE_VRAM_MB
for 2 consecutive checks, automatically launches train_voc_sam_pcre_v2.sh.
"""

import os
import sys
import time
import subprocess
import datetime
from pathlib import Path

ROOT = Path("/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE")
LOG_FILE = ROOT / "experiments/voc_sam_pcre_v2/queue_supervisor.log"
MIN_FREE_VRAM_MB = 16000  # Requires 16 GB free VRAM
CHECK_INTERVAL_SEC = 25

def log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def get_free_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            text=True
        )
        return int(out.strip().split("\n")[0])
    except Exception as e:
        log(f"Warning: GPU query (nvidia-smi): {e}")
        return 0

def main():
    log("=" * 60)
    log("VOC SAM-PCRE v2 Queue Supervisor Started")
    log("Methods: SAT-CR (Method 1) + SBF-Distill (Method 2)")
    log(f"Required Free VRAM: {MIN_FREE_VRAM_MB} MiB")
    log(f"Working Directory: {ROOT}")
    log("=" * 60)

    consecutive_ready = 0
    while True:
        vram = get_free_vram_mb()
        if vram >= MIN_FREE_VRAM_MB:
            consecutive_ready += 1
            log(f"GPU check {consecutive_ready}/2: Free VRAM = {vram} MiB >= {MIN_FREE_VRAM_MB} MiB.")
            if consecutive_ready >= 2:
                log("GPU memory and status confirmed ready. Launching training pipeline now!")
                break
        else:
            if consecutive_ready > 0:
                log(f"GPU check reset: Free VRAM = {vram} MiB < {MIN_FREE_VRAM_MB} MiB.")
            consecutive_ready = 0
            log(f"Waiting for GPU to be ready (Current: {vram} MiB free, needed: {MIN_FREE_VRAM_MB} MiB). Next check in {CHECK_INTERVAL_SEC}s...")

        time.sleep(CHECK_INTERVAL_SEC)

    # Launch training
    cmd = ["bash", str(ROOT / "train_voc_sam_pcre_v2.sh")]
    log(f"Executing: {' '.join(cmd)}")
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    stdout_log = ROOT / "experiments/voc_sam_pcre_v2/queue_stdout.log"
    live_log = ROOT / "experiments/voc_sam_pcre_v2/live_run.log"
    with open(stdout_log, "w") as f_out, open(live_log, "w") as f_live:
        for line in proc.stdout:
            f_out.write(line)
            f_out.flush()
            f_live.write(line)
            f_live.flush()
            if "Iter:" in line or "m-IoU" in line or "Saved checkpoint" in line or "INFO:" in line:
                log(f"[Trainer] {line.strip()}")

    ret = proc.wait()
    duration = (time.time() - start_time) / 3600.0
    log(f"Training completed with exit code {ret} in {duration:.2f} hours.")
    sys.exit(ret)

if __name__ == "__main__":
    main()
