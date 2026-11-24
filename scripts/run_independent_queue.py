#!/usr/bin/env python3
"""Sequential Independent GPU Queue Supervisor for VOC SAM-PCRE Ablation Studies.

Runs the two Paradigm 2 methods strictly independently in sequence:
  Stage 1: SAT-CR alone (experiments/voc_sam_sat_cr)
  Stage 2: SBF-Distill alone (experiments/voc_sam_sbf_distill)
"""

import os
import sys
import time
import subprocess
import datetime
from pathlib import Path

ROOT = Path("/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE")
LOG_FILE = ROOT / "experiments/independent_ablation_queue.log"
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

def wait_for_gpu():
    consecutive_ready = 0
    while True:
        vram = get_free_vram_mb()
        if vram >= MIN_FREE_VRAM_MB:
            consecutive_ready += 1
            log(f"GPU check {consecutive_ready}/2: Free VRAM = {vram} MiB >= {MIN_FREE_VRAM_MB} MiB.")
            if consecutive_ready >= 2:
                log("GPU memory confirmed ready.")
                break
        else:
            if consecutive_ready > 0:
                log(f"GPU check reset: Free VRAM = {vram} MiB < {MIN_FREE_VRAM_MB} MiB.")
            consecutive_ready = 0
            log(f"Waiting for GPU to be ready (Current: {vram} MiB free, needed: {MIN_FREE_VRAM_MB} MiB). Next check in {CHECK_INTERVAL_SEC}s...")

        time.sleep(CHECK_INTERVAL_SEC)

def run_stage(script_name, work_dir_rel, stage_name):
    log("=" * 70)
    log(f"Starting {stage_name}")
    log(f"Script: {script_name}")
    log(f"Work directory: {work_dir_rel}")
    log("=" * 70)

    wait_for_gpu()

    cmd = ["bash", str(ROOT / script_name)]
    log(f"Executing: {' '.join(cmd)}")
    start_time = time.time()

    work_dir = ROOT / work_dir_rel
    work_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = work_dir / "queue_stdout.log"
    live_log = work_dir / "live_run.log"

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    with open(stdout_log, "w") as f_out, open(live_log, "w") as f_live:
        for line in proc.stdout:
            f_out.write(line)
            f_out.flush()
            f_live.write(line)
            f_live.flush()
            if "Iter:" in line or "m-IoU" in line or "Saved checkpoint" in line or "INFO:" in line:
                log(f"[{stage_name}] {line.strip()}")

    ret = proc.wait()
    duration = (time.time() - start_time) / 3600.0
    log(f"{stage_name} finished with exit code {ret} in {duration:.2f} hours.")
    if ret != 0:
        log(f"ERROR: {stage_name} exited with non-zero code {ret}! Halting queue.")
        sys.exit(ret)

def main():
    log("=" * 70)
    log("VOC SAM-PCRE Independent Ablation Queue Supervisor Started")
    log("Planned Stages:")
    log("  Stage 1: Method 1 (SAT-CR alone) -> experiments/voc_sam_sat_cr")
    log("  Stage 2: Method 2 (SBF-Distill alone) -> experiments/voc_sam_sbf_distill")
    log("=" * 70)

    # Stage 1: SAT-CR alone
    run_stage("train_voc_sam_sat_cr.sh", "experiments/voc_sam_sat_cr", "Stage 1: SAT-CR")

    # Cooldown between stages
    log("Cooling down 60 seconds before Stage 2...")
    time.sleep(60)

    # Stage 2: SBF-Distill alone
    run_stage("train_voc_sam_sbf_distill.sh", "experiments/voc_sam_sbf_distill", "Stage 2: SBF-Distill")

    log("=" * 70)
    log("ALL INDEPENDENT EXPERIMENTS COMPLETED SUCCESSFULLY!")
    log("=" * 70)

if __name__ == "__main__":
    main()
