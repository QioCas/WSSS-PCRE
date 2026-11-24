#!/usr/bin/env python3
"""Automated GPU Queue Supervisor for VOC SAM-PCRE Training.

Monitors GPU 0 free VRAM. Once free VRAM >= MIN_FREE_VRAM_MB for 2 consecutive checks,
launches the SAM-PCRE training pipeline on Pascal VOC 2012.
"""

import os
import sys
import time
import subprocess
import datetime
from pathlib import Path

ROOT = Path("/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE")
LOG_FILE = ROOT / "experiments/voc_sam_pcre/queue_supervisor.log"
MIN_FREE_VRAM_MB = 16000  # Requires 16 GB free VRAM (available once COCO run completes)
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
        log(f"Warning: Failed to query nvidia-smi: {e}")
        return 0

def main():
    log("=" * 60)
    log("VOC SAM-PCRE Queue Supervisor Started")
    log(f"Required Free VRAM: {MIN_FREE_VRAM_MB} MiB")
    log(f"Working Directory: {ROOT}")
    log("=" * 60)

    consecutive_ready = 0

    while True:
        free_mb = get_free_vram_mb()
        log(f"GPU Monitor: Free VRAM = {free_mb} MiB / Required = {MIN_FREE_VRAM_MB} MiB")

        if free_mb >= MIN_FREE_VRAM_MB:
            consecutive_ready += 1
            log(f"GPU meets requirement! Ready check {consecutive_ready}/2...")
            if consecutive_ready >= 2:
                log("GPU memory persistently available! Starting VOC SAM-PCRE training NOW...")
                break
        else:
            consecutive_ready = 0

        time.sleep(CHECK_INTERVAL_SEC)

    # Launch Training Script
    cmd = [
        str(ROOT / "train_voc_sam_pcre.sh"),
        "--spg", "4",
        "--max_iters", "20000",
        "--eval_iters", "5000",
        "--save_ckpt"
    ]
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

    with open(ROOT / "experiments/voc_sam_pcre/live_run.log", "w") as run_log:
        for line in proc.stdout:
            run_log.write(line)
            run_log.flush()
            if "Iter:" in line or "Validating" in line or "m-IoU" in line:
                log(f"[Trainer] {line.strip()}")

    ret = proc.wait()
    duration = time.time() - start_time
    log(f"Training completed with exit code {ret} in {duration/3600:.2f} hours.")

if __name__ == "__main__":
    main()
