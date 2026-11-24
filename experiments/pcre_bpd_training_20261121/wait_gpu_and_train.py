"""Wait for GPU free memory to reach threshold (e.g. >= 15GB), then automatically execute PCRE-BPD training and evaluation.
"""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent

def get_gpu_free_memory_mb():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        parts = res.strip().split("\n")[0].split(",")
        free_mb = int(parts[0].strip())
        total_mb = int(parts[1].strip())
        return free_mb, total_mb
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}", flush=True)
        return 0, 0

def main():
    parser = argparse.ArgumentParser(description="Wait for GPU memory and train PCRE-BPD")
    parser.add_argument("--min_free_mb", type=int, default=15000, help="Minimum free GPU memory in MB (default: 15000 = ~15GB)")
    parser.add_argument("--poll_interval", type=int, default=20, help="Polling interval in seconds")
    parser.add_argument("--max_iters", type=int, default=400, help="Training iterations")
    parser.add_argument("--spg", type=int, default=4, help="Samples per GPU")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--lr_decoder_mult", type=float, default=2.0, help="Decoder LR multiplier")
    parser.add_argument("--w_distill", type=float, default=0.25, help="Weight for BAPD Distill Loss")
    args = parser.parse_args()

    log_path = WORK_DIR / "wait_and_train.log"
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitoring GPU memory. Waiting for >= {args.min_free_mb} MB free memory...", flush=True)

    with open(log_path, "a") as log_f:
        log_f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitoring started. Threshold: {args.min_free_mb} MB free.\n")

    check_count = 0
    while True:
        free_mb, total_mb = get_gpu_free_memory_mb()
        used_mb = total_mb - free_mb
        check_count += 1

        msg = (f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Check #{check_count}: "
               f"Free: {free_mb} MB ({free_mb/1024:.2f} GB) / {total_mb} MB. "
               f"Used: {used_mb} MB. Target: >= {args.min_free_mb} MB ({args.min_free_mb/1024:.1f} GB)")
        print(msg, flush=True)

        if check_count % 3 == 0:
            with open(log_path, "a") as log_f:
                log_f.write(msg + "\n")

        if free_mb >= args.min_free_mb:
            print(f"\n>>> TARGET REACHED! Free memory: {free_mb} MB >= {args.min_free_mb} MB. Launching PCRE-BPD training now! <<<\n", flush=True)
            with open(log_path, "a") as log_f:
                log_f.write(f"\n>>> TARGET REACHED! Free memory: {free_mb} MB >= {args.min_free_mb} MB. Launching training! <<<\n")
            break

        time.sleep(args.poll_interval)

    # 1. Launch Training
    train_cmd = [
        sys.executable,
        str(WORK_DIR / "train_pcre_bpd.py"),
        "--max_iters", str(args.max_iters),
        "--log_iters", "20",
        "--save_iters", str(args.max_iters),
        "--eval_iters", str(args.max_iters),
        "--warmup_iters", "30",
        "--seg_warmup", "10",
        "--spg", str(args.spg),
        "--lr", str(args.lr),
        "--lr_decoder_mult", str(args.lr_decoder_mult),
        "--w_distill", str(args.w_distill),
        "--eval_limit", "100"
    ]
    print(f"Executing: {' '.join(train_cmd)}", flush=True)
    train_proc = subprocess.run(train_cmd, cwd=str(ROOT))
    if train_proc.returncode != 0:
        print(f"Training failed with return code {train_proc.returncode}!", flush=True)
        return

    # 2. Launch Full 1449-Image VOC Evaluation
    best_ckpt = WORK_DIR / "checkpoints" / f"model_iter_{args.max_iters:08d}.pth"
    if not best_ckpt.exists():
        ckpts = sorted((WORK_DIR / "checkpoints").glob("*.pth"))
        if ckpts:
            best_ckpt = ckpts[-1]

    if best_ckpt.exists():
        eval_cmd = [
            sys.executable,
            str(WORK_DIR / "evaluate_pcre_bpd.py"),
            "--checkpoint", str(best_ckpt),
            "--output_dir", str(WORK_DIR / "eval_results_final"),
            "--crf_workers", "4",
            "--limit", "0" # Full 1449 images
        ]
        print(f"\nExecuting Full Evaluation: {' '.join(eval_cmd)}", flush=True)
        eval_proc = subprocess.run(eval_cmd, cwd=str(ROOT))
        print(f"Full Evaluation finished with code {eval_proc.returncode}", flush=True)

if __name__ == "__main__":
    main()
