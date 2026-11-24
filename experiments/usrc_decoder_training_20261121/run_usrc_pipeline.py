"""
End-to-end Pipeline Runner for USRC Experiment:
1. Trains Decoder with Unsupervised SAM-Reliability Clustered Pseudo-Labels (2500 steps, ~6 min)
2. Evaluates EMA model on full VOC 2012 Val set (1449 images) with Raw + DenseCRF (~12 min)
3. Evaluates Student model on full VOC 2012 Val set (1449 images) with Raw + DenseCRF (~12 min)
4. Generates comprehensive comparative summary against True Baseline and FBSD
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(ROOT))

EXP_DIR = ROOT / "experiments/usrc_decoder_training_20261121"
LOG_FILE = EXP_DIR / "pipeline.log"


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{t_str}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def run_cmd(cmd, desc):
    log(f"--- STARTING: {desc} ---")
    log(f"Command: {cmd}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, executable="/bin/bash")
    elapsed = time.time() - t0
    if res.returncode != 0:
        log(f"ERROR: {desc} failed with exit code {res.returncode} in {elapsed:.1f}s")
        sys.exit(res.returncode)
    log(f"--- COMPLETED: {desc} in {elapsed/60:.2f} minutes ---\n")


def main():
    log("======================================================================")
    log("USRC: Unsupervised SAM-Reliability Clustered Training & Evaluation")
    log("======================================================================")

    # Step 1: Train USRC
    train_cmd = (
        f"python3 experiments/usrc_decoder_training_20261121/train_usrc.py "
        f"--steps 2500 --batch_size 4 --lr 3e-5 --anchor_weight 0.5 "
        f"--work_dir experiments/usrc_decoder_training_20261121"
    )
    run_cmd(train_cmd, "USRC Decoder Training (2500 steps)")

    # Step 2: Evaluate EMA Model (Primary)
    ema_ckpt = EXP_DIR / "checkpoints/model_usrc_final_ema.pth"
    ema_eval_cmd = (
        f"python3 experiments/usrc_decoder_training_20261121/evaluate_usrc.py "
        f"--checkpoint {ema_ckpt} "
        f"--output_dir {EXP_DIR}/eval_results_ema "
        f"--crf_workers 4 --limit 0"
    )
    run_cmd(ema_eval_cmd, "Full VOC Val Evaluation: USRC EMA Model")

    # Step 3: Evaluate Student Model (Comparative)
    student_ckpt = EXP_DIR / "checkpoints/model_usrc_final_student.pth"
    student_eval_cmd = (
        f"python3 experiments/usrc_decoder_training_20261121/evaluate_usrc.py "
        f"--checkpoint {student_ckpt} "
        f"--output_dir {EXP_DIR}/eval_results_student "
        f"--crf_workers 4 --limit 0"
    )
    run_cmd(student_eval_cmd, "Full VOC Val Evaluation: USRC Student Model")

    # Step 4: Generate Summary
    log("Generating Comprehensive Comparison Report...")
    baseline_path = ROOT / "experiments/inference_voc_val_20261119/full/baseline/metrics.json"
    fbsd_path = ROOT / "experiments/decoder_selftrain_20261119/eval_results/metrics.json"
    ema_path = EXP_DIR / "eval_results_ema/metrics.json"
    stu_path = EXP_DIR / "eval_results_student/metrics.json"

    base_m = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    fbsd_m = json.loads(fbsd_path.read_text()) if fbsd_path.exists() else None
    ema_m = json.loads(ema_path.read_text()) if ema_path.exists() else None
    stu_m = json.loads(stu_path.read_text()) if stu_path.exists() else None

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_raw_miou": base_m["raw"]["miou"] * 100 if base_m else None,
        "baseline_crf_miou": base_m["crf"]["miou"] * 100 if base_m else None,
        "fbsd_raw_miou": fbsd_m["raw"]["miou"] * 100 if fbsd_m else None,
        "fbsd_crf_miou": fbsd_m["crf"]["miou"] * 100 if fbsd_m else None,
        "usrc_ema_raw_miou": ema_m["raw"]["miou"] * 100 if ema_m else None,
        "usrc_ema_crf_miou": ema_m["crf"]["miou"] * 100 if ema_m else None,
        "usrc_stu_raw_miou": stu_m["raw"]["miou"] * 100 if stu_m else None,
        "usrc_stu_crf_miou": stu_m["crf"]["miou"] * 100 if stu_m else None,
    }

    with open(EXP_DIR / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)
    log("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    log(f"Baseline CRF: {summary['baseline_crf_miou']:.2f}%")
    if summary['fbsd_crf_miou']:
        log(f"FBSD-v1  CRF: {summary['fbsd_crf_miou']:.2f}% (Delta: {summary['fbsd_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    log(f"USRC EMA CRF: {summary['usrc_ema_crf_miou']:.2f}% (Delta: {summary['usrc_ema_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    log(f"USRC Stu CRF: {summary['usrc_stu_crf_miou']:.2f}% (Delta: {summary['usrc_stu_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    log("=" * 70)


if __name__ == "__main__":
    main()
