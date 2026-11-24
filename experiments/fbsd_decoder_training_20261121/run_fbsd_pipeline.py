"""
Pipeline runner for FBSD:
1. Trains Decoder with frozen ViT backbone on SAM pseudo-labels (train_fbsd.py)
2. Evaluates the resulting EMA checkpoint on all 1449 images of Pascal VOC 2012 Val (evaluate_fbsd.py)
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "experiments/fbsd_decoder_training_20261121"


def main():
    parser = argparse.ArgumentParser(description="FBSD Pipeline Supervisor")
    parser.add_argument("--steps", type=int, default=2500, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--anchor_weight", type=float, default=0.5, help="Anchor KL weight")
    parser.add_argument("--anchor_temp", type=float, default=2.0, help="Anchor KL temp")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    args = parser.parse_args()

    pipeline_log = WORK_DIR / "pipeline.log"
    with open(pipeline_log, "w") as f:
        f.write(f"FBSD Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Execute Training
    train_cmd = [
        sys.executable,
        str(WORK_DIR / "train_fbsd.py"),
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--anchor_weight", str(args.anchor_weight),
        "--anchor_temp", str(args.anchor_temp),
        "--ema_decay", str(args.ema_decay),
        "--log_interval", "50",
        "--save_interval", "500",
        "--work_dir", str(WORK_DIR.relative_to(ROOT))
    ]
    print(f"\n[1/2] Launching FBSD Training:\n{' '.join(train_cmd)}\n", flush=True)
    t0 = time.time()
    res = subprocess.run(train_cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"Training failed with return code {res.returncode}", flush=True)
        return

    train_elapsed = time.time() - t0
    print(f"\nTraining completed in {train_elapsed / 60:.2f} minutes.", flush=True)

    # 2. Execute Evaluation on the EMA checkpoint (primary)
    ema_ckpt = WORK_DIR / "checkpoints/model_fbsd_final_ema.pth"
    if not ema_ckpt.exists():
        ema_ckpt = sorted((WORK_DIR / "checkpoints").glob("*ema.pth"))[-1]

    eval_out_dir = WORK_DIR / "eval_results_ema"
    eval_cmd = [
        sys.executable,
        str(WORK_DIR / "evaluate_fbsd.py"),
        "--checkpoint", str(ema_ckpt),
        "--output_dir", str(eval_out_dir),
        "--crf_workers", "4",
        "--limit", "0"
    ]
    print(f"\n[2/2] Launching Evaluation on all 1449 VOC Val images:\n{' '.join(eval_cmd)}\n", flush=True)
    t1 = time.time()
    res_eval = subprocess.run(eval_cmd, cwd=str(ROOT))
    eval_elapsed = time.time() - t1
    print(f"\nEvaluation completed in {eval_elapsed / 60:.2f} minutes.", flush=True)

    print(f"\nFBSD Pipeline Finished. Total time: {(time.time() - t0)/60:.2f} minutes.", flush=True)


if __name__ == "__main__":
    main()
