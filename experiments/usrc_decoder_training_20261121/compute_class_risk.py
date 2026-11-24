"""
Stage 1 of USRC:
Unsupervised Class-Risk Profiling.
Computes spatial agreement IoU, area distortion factor, and composite risk score
for each of the 20 foreground classes between SAM pseudo-labels and Teacher predictions.
Applies Otsu optimal thresholding to automatically cluster classes into Low-Risk and High-Risk.
"""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.voc import class_list


def compute_otsu_threshold(values):
    """Computes Otsu optimal threshold to split 1D continuous values into 2 clusters."""
    sorted_vals = np.sort(values)
    best_thresh = sorted_vals[len(sorted_vals) // 2]
    best_var = -1.0

    # Try every midpoint as a potential threshold
    for i in range(1, len(sorted_vals)):
        thresh = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
        c1 = sorted_vals[sorted_vals <= thresh]
        c2 = sorted_vals[sorted_vals > thresh]
        if len(c1) == 0 or len(c2) == 0:
            continue
        w1 = len(c1) / float(len(sorted_vals))
        w2 = len(c2) / float(len(sorted_vals))
        m1 = np.mean(c1)
        m2 = np.mean(c2)
        # Inter-cluster variance
        var_between = w1 * w2 * ((m1 - m2) ** 2)
        if var_between > best_var:
            best_var = var_between
            best_thresh = thresh

    return float(best_thresh)


def main():
    sam_cache = ROOT / "experiments/representation_sam_20261120/cache/sam/labels"
    raw_cache = ROOT / "experiments/decoder_selftrain_20261119/teacher_cache/raw_teacher"
    train_txt = ROOT / "datasets/voc/train_aug.txt"
    output_json = ROOT / "experiments/usrc_decoder_training_20261121/class_risk_profile.json"

    names = [x.strip() for x in train_txt.read_text().split() if x.strip()]
    sample_size = min(3000, len(names))
    sample_names = names[:sample_size]
    print(f"Profiling class risk across {sample_size} training images...")

    sam_areas = {i: 0 for i in range(21)}
    raw_areas = {i: 0 for i in range(21)}
    intersections = {i: 0 for i in range(21)}

    for idx, name in enumerate(sample_names):
        sp = sam_cache / f"{name}.png"
        rp = raw_cache / f"{name}.png"
        if sp.exists() and rp.exists():
            s_arr = np.array(Image.open(sp))
            r_arr = np.array(Image.open(rp))
            for c in range(1, 21):
                s_m = (s_arr == c)
                r_m = (r_arr == c)
                if s_m.any() or r_m.any():
                    sam_areas[c] += int(s_m.sum())
                    raw_areas[c] += int(r_m.sum())
                    intersections[c] += int((s_m & r_m).sum())

        if (idx + 1) % 500 == 0 or (idx + 1) == sample_size:
            print(f"Processed {idx + 1}/{sample_size} images...")

    per_class_profile = {}
    risk_scores = []
    class_indices = []

    for c in range(1, 21):
        s_a = sam_areas[c]
        r_a = raw_areas[c]
        inter = intersections[c]
        union = s_a + r_a - inter
        iou = float(inter / union) if union > 0 else 1.0
        ratio = float(s_a / r_a) if r_a > 0 else 1.0
        risk = float((1.0 - iou) + 0.5 * abs(np.log(max(1e-3, ratio))))

        per_class_profile[class_list[c]] = {
            "class_id": c,
            "sam_area_pixels": s_a,
            "teacher_area_pixels": r_a,
            "agreement_iou": iou,
            "area_ratio": ratio,
            "risk_score": risk
        }
        risk_scores.append(risk)
        class_indices.append(c)

    # Compute optimal threshold via Otsu clustering
    tau_star = compute_otsu_threshold(risk_scores)
    print(f"\nOtsu Optimal Clustering Threshold: tau* = {tau_star:.4f}")

    low_risk = []
    high_risk = []
    for c in range(1, 21):
        c_name = class_list[c]
        r = per_class_profile[c_name]["risk_score"]
        if r <= tau_star:
            per_class_profile[c_name]["cluster"] = "low_risk"
            low_risk.append(c)
        else:
            per_class_profile[c_name]["cluster"] = "high_risk"
            high_risk.append(c)

    summary = {
        "threshold_tau": tau_star,
        "sample_size": sample_size,
        "low_risk_classes": [class_list[c] for c in low_risk],
        "low_risk_ids": low_risk,
        "high_risk_classes": [class_list[c] for c in high_risk],
        "high_risk_ids": high_risk,
        "per_class": per_class_profile
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved risk profile to {output_json}")
    print(f"Low-Risk Cluster ({len(low_risk)} classes): {summary['low_risk_classes']}")
    print(f"High-Risk Cluster ({len(high_risk)} classes): {summary['high_risk_classes']}")


if __name__ == "__main__":
    main()
