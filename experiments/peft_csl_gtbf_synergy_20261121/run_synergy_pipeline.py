"""
End-to-End Synergy Pipeline:
Step 1: Export 1,449 DenseCRF masks from PEFT-CSL model
Step 2: Apply GTBF-CAMR Refinement on top of the PEFT-CSL masks
Step 3: Compile final comparative metrics
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(ROOT))

EXP_DIR = ROOT / "experiments/peft_csl_gtbf_synergy_20261121"
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
    log("PEFT-CSL + GTBF-CAMR Synergistic Refinement Pipeline")
    log("======================================================================")

    # Step 1: Export PEFT-CSL DenseCRF Masks
    export_cmd = f"python3 experiments/peft_csl_gtbf_synergy_20261121/export_peft_csl_crf.py"
    run_cmd(export_cmd, "Export PEFT-CSL DenseCRF Masks (1449 images)")

    # Step 2: Run GTBF-CAMR on PEFT-CSL Masks
    gtbf_cmd = f"python3 experiments/peft_csl_gtbf_synergy_20261121/run_peft_gtbf_synergy.py"
    run_cmd(gtbf_cmd, "GTBF-CAMR Synergy Refinement (1449 images)")

    log("=" * 70)
    log("SYNERGY PIPELINE COMPLETED SUCCESSFULLY!")
    metrics_path = EXP_DIR / "evaluation/metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        log(f"True Baseline CRF:       {m['true_baseline_crf_miou']:.2f}%")
        log(f"PEFT-CSL CRF:            {m['peft_csl_crf_miou']:.2f}%")
        log(f"Synergy (PEFT + GTBF):   {m['synergy_miou']:.2f}%")
        log(f"NET GAIN OVER BASELINE:  +{m['net_gain_vs_baseline']:.2f}%")
    log("=" * 70)


if __name__ == "__main__":
    main()
