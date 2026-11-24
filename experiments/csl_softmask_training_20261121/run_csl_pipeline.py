"""
Pipeline Runner for CSL-SD (Confidence Separable Soft-Distilled) Experiment:
1. Trains Decoder with CSL-SD Softmasking (2500 steps, ~6.5 min)
2. Evaluates EMA model on full VOC 2012 Val set (1449 images) with Raw + DenseCRF (~10 min)
3. Compares directly against Baseline, FBSD, and USRC
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(ROOT))

EXP_DIR = ROOT / "experiments/csl_softmask_training_20261121"
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
    log("CSL-SD: Confidence Separable Soft-Distilled Training & Evaluation")
    log("======================================================================")

    # Step 1: Train CSL Softmask
    train_cmd = (
        f"python3 experiments/csl_softmask_training_20261121/train_csl_softmask.py "
        f"--steps 2500 --batch_size 4 --lr 3e-5 --anchor_weight 0.5 --csl_alpha 8.0 "
        f"--work_dir experiments/csl_softmask_training_20261121"
    )
    run_cmd(train_cmd, "CSL-SD Softmask Decoder Training (2500 steps)")

    # Step 2: Evaluate EMA Model (Primary)
    ema_ckpt = EXP_DIR / "checkpoints/model_csl_final_ema.pth"
    ema_eval_cmd = (
        f"python3 experiments/usrc_decoder_training_20261121/evaluate_usrc.py "
        f"--checkpoint {ema_ckpt} "
        f"--output_dir {EXP_DIR}/eval_results_ema "
        f"--crf_workers 4 --limit 0"
    )
    run_cmd(ema_eval_cmd, "Full VOC Val Evaluation: CSL-SD EMA Model")

    # Step 3: Compile Report
    log("Compiling Multi-Method Comparison Report...")
    baseline_path = ROOT / "experiments/inference_voc_val_20261119/full/baseline/metrics.json"
    fbsd_path = ROOT / "experiments/fbsd_decoder_training_20261121/eval_results_ema/metrics.json"
    usrc_path = ROOT / "experiments/usrc_decoder_training_20261121/eval_results_ema/metrics.json"
    csl_path = EXP_DIR / "eval_results_ema/metrics.json"

    base_m = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    fbsd_m = json.loads(fbsd_path.read_text()) if fbsd_path.exists() else None
    usrc_m = json.loads(usrc_path.read_text()) if usrc_path.exists() else None
    csl_m = json.loads(csl_path.read_text()) if csl_path.exists() else None

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_crf_miou": base_m["crf"]["miou"] * 100 if base_m else None,
        "fbsd_crf_miou": fbsd_m["crf"]["miou"] * 100 if fbsd_m else None,
        "usrc_crf_miou": usrc_m["crf"]["miou"] * 100 if usrc_m else None,
        "csl_ema_raw_miou": csl_m["raw"]["miou"] * 100 if csl_m else None,
        "csl_ema_crf_miou": csl_m["crf"]["miou"] * 100 if csl_m else None,
    }

    with open(EXP_DIR / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)
    log("CSL-SD EXPERIMENT COMPLETED SUCCESSFULLY!")
    log(f"Baseline CRF: {summary['baseline_crf_miou']:.2f}%")
    if summary['fbsd_crf_miou']:
        log(f"FBSD     CRF: {summary['fbsd_crf_miou']:.2f}% (Delta: {summary['fbsd_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    if summary['usrc_crf_miou']:
        log(f"USRC     CRF: {summary['usrc_crf_miou']:.2f}% (Delta: {summary['usrc_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    log(f"CSL-SD   CRF: {summary['csl_ema_crf_miou']:.2f}% (Delta: {summary['csl_ema_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    log("=" * 70)


if __name__ == "__main__":
    main()
