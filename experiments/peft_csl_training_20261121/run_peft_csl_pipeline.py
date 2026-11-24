"""
End-to-End Pipeline Runner for PEFT-CSL Experiment:
1. Trains Decoder + LoRA ViT Attention (2500 steps, ~8 min)
2. Evaluates Merged EMA model on full VOC 2012 Val set (1449 images) with Raw + DenseCRF (~10 min)
3. Compares directly against Baseline, FBSD, USRC, and CSL-SD
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(ROOT))

EXP_DIR = ROOT / "experiments/peft_csl_training_20261121"
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
    log("PEFT-CSL: Parameter-Efficient LoRA ViT + CSL Soft-Distillation")
    log("======================================================================")

    # Step 1: Train PEFT-CSL
    train_cmd = (
        f"python3 experiments/peft_csl_training_20261121/train_peft_csl.py "
        f"--steps 2500 --batch_size 4 --lr_decoder 3e-5 --lr_lora 1e-4 --lora_rank 8 "
        f"--anchor_weight 0.5 --csl_alpha 8.0 --work_dir experiments/peft_csl_training_20261121"
    )
    run_cmd(train_cmd, "PEFT-CSL Training (2500 steps)")

    # Step 2: Evaluate Merged EMA Model (Primary)
    ema_ckpt = EXP_DIR / "checkpoints/model_peft_csl_final_ema.pth"
    ema_eval_cmd = (
        f"python3 experiments/usrc_decoder_training_20261121/evaluate_usrc.py "
        f"--checkpoint {ema_ckpt} "
        f"--output_dir {EXP_DIR}/eval_results_ema "
        f"--crf_workers 4 --limit 0"
    )
    run_cmd(ema_eval_cmd, "Full VOC Val Evaluation: PEFT-CSL EMA Model")

    # Step 3: Compile 5-Way Comparison Report
    log("Compiling Comprehensive 5-Way Comparison Report...")
    baseline_path = ROOT / "experiments/inference_voc_val_20261119/full/baseline/metrics.json"
    fbsd_path = ROOT / "experiments/fbsd_decoder_training_20261121/eval_results_ema/metrics.json"
    usrc_path = ROOT / "experiments/usrc_decoder_training_20261121/eval_results_ema/metrics.json"
    csl_path = ROOT / "experiments/csl_softmask_training_20261121/eval_results_ema/metrics.json"
    peft_path = EXP_DIR / "eval_results_ema/metrics.json"

    base_m = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    fbsd_m = json.loads(fbsd_path.read_text()) if fbsd_path.exists() else None
    usrc_m = json.loads(usrc_path.read_text()) if usrc_path.exists() else None
    csl_m = json.loads(csl_path.read_text()) if csl_path.exists() else None
    peft_m = json.loads(peft_path.read_text()) if peft_path.exists() else None

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_raw_miou": base_m["raw"]["miou"] * 100 if base_m else None,
        "baseline_crf_miou": base_m["crf"]["miou"] * 100 if base_m else None,
        "fbsd_crf_miou": fbsd_m["crf"]["miou"] * 100 if fbsd_m else None,
        "usrc_crf_miou": usrc_m["crf"]["miou"] * 100 if usrc_m else None,
        "csl_sd_raw_miou": csl_m["raw"]["miou"] * 100 if csl_m else None,
        "csl_sd_crf_miou": csl_m["crf"]["miou"] * 100 if csl_m else None,
        "peft_csl_raw_miou": peft_m["raw"]["miou"] * 100 if peft_m else None,
        "peft_csl_crf_miou": peft_m["crf"]["miou"] * 100 if peft_m else None,
    }

    with open(EXP_DIR / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)
    log("PEFT-CSL EXPERIMENT COMPLETED SUCCESSFULLY!")
    log(f"Baseline CRF: {summary['baseline_crf_miou']:.2f}%")
    if summary['fbsd_crf_miou']:
        log(f"FBSD     CRF: {summary['fbsd_crf_miou']:.2f}% (Delta: {summary['fbsd_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    if summary['csl_sd_crf_miou']:
        log(f"CSL-SD   CRF: {summary['csl_sd_crf_miou']:.2f}% (Delta: {summary['csl_sd_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    log(f"PEFT-CSL CRF: {summary['peft_csl_crf_miou']:.2f}% (Delta: {summary['peft_csl_crf_miou'] - summary['baseline_crf_miou']:+.2f}%)")
    log("=" * 70)


if __name__ == "__main__":
    main()
