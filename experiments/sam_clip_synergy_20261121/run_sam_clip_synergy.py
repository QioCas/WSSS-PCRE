"""
Experiment 2: SAM + CLIP Zero-Shot Co-occurrence Verification (SAM-CLIP-Synergy)
Applied on Pascal VOC 2012 Val (1,449 images).

Key Technical Novelties:
1. Multi-Modal Semantic Verification:
   - SAM excels at spatial and boundary crispness but is class-agnostic.
   - CLIP provides powerful open-vocabulary visual-language semantic discriminability.
   - For every segmented component (especially notorious co-occurring confusion pairs like
     chair vs. diningtable, sofa vs. chair, person vs. bicycle/motorbike, bus vs. train),
     crops the localized image patch and verifies semantic identity via CLIP ViT-B/16 text embeddings.
2. Background Clutter Suppression:
   - Suppresses spurious false-positive fragments that have high cosine similarity with
     generic background context (floors, walls, grass, roads).
3. Evaluates all 1,449 images against True Baseline (74.22%), PEFT-CSL (74.36%), and GTBF-CAMR (75.88%).
"""
import os
import sys
import time
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import torch

ROOT = Path("/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE")
CLIP_DIR = Path("/home/jupyter-iec2024iot08/Workspace/.deps/pcre_clip")
CLIP_CKPT = Path("/home/jupyter-iec2024iot08/Workspace/.deps/pcre_clip_weights/ViT-B-16.pt")
DATA_DIR = Path("/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012")
INPUT_MASKS_DIR = ROOT / "experiments/sam_advanced_prompting_20261121/evaluation/masks"
FALLBACK_MASKS_DIR = ROOT / "experiments/peft_csl_gtbf_synergy_20261121/evaluation/masks"
EXP_DIR = ROOT / "experiments/sam_clip_synergy_20261121"
OUT_DIR = EXP_DIR / "evaluation"

sys.path.insert(0, str(CLIP_DIR))
import clip

VOC_CLASSES = [
    "_background_", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

VOC_PROMPT_NAMES = [
    "background, floor, wall, road or sky",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "dining table",
    "dog", "horse", "motorbike", "person", "potted plant",
    "sheep", "sofa", "train", "tv monitor"
]

# High-confusion co-occurrence sets
CONFUSION_PAIRS = {
    9: [9, 11, 18, 0],   # chair: check vs diningtable, sofa, bg
    11: [11, 9, 18, 0],  # diningtable: check vs chair, sofa, bg
    18: [18, 9, 11, 0],  # sofa: check vs chair, diningtable, bg
    2: [2, 14, 15, 0],   # bicycle: check vs motorbike, person, bg
    14: [14, 2, 15, 0],  # motorbike: check vs bicycle, person, bg
    19: [19, 6, 0],      # train: check vs bus, bg
    6: [6, 19, 0],       # bus: check vs train, bg
    10: [10, 13, 17, 0], # cow: check vs horse, sheep, bg
    13: [13, 10, 17, 0], # horse: check vs cow, sheep, bg
}


def palette():
    colors = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        value = i
        for j in range(8):
            for channel in range(3):
                colors[i, channel] |= ((value >> channel) & 1) << (7-j)
            value >>= 3
    colors[255] = (224, 224, 224)
    return colors


def compute_hist(pred, gt, nclasses=21):
    valid = (gt >= 0) & (gt < nclasses)
    return np.bincount(nclasses * gt[valid].astype(np.int64) + pred[valid],
                       minlength=nclasses * nclasses).reshape(nclasses, nclasses)


def compute_scores(hist):
    tp = np.diag(hist).astype(float)
    fn = hist.sum(1) - tp
    fp = hist.sum(0) - tp
    union = tp + fp + fn
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    present = union > 0
    miou = float(np.mean(iou[present]))
    return miou, iou, precision, recall


def main():
    print("=" * 70)
    print("=== STARTING SAM + CLIP ZERO-SHOT CO-OCCURRENCE VERIFICATION ===")
    print("=" * 70)
    start_time = time.time()

    val_list_file = ROOT / "datasets/voc/val.txt"
    val_names = val_list_file.read_text().split()
    total_images = len(val_names)
    print(f"Total VOC Val images: {total_images}")

    # Determine input mask source (use Experiment 1 if generated, else fallback to best Synergy)
    masks_source_dir = INPUT_MASKS_DIR if (INPUT_MASKS_DIR.exists() and len(list(INPUT_MASKS_DIR.glob('*.png'))) >= total_images) else FALLBACK_MASKS_DIR
    print(f"Using input masks from: {masks_source_dir}")

    print(f"Loading CLIP ViT-B/16 from {CLIP_CKPT}...")
    torch.cuda.set_device(0)
    clip_model, clip_preprocess = clip.load(str(CLIP_CKPT), device="cuda")
    clip_model.eval()

    # Pre-compute text feature bank
    templates = [
        "a photo of a {}.",
        "a clean picture of a {}.",
        "a crop showing a {}."
    ]
    with torch.no_grad():
        text_features_list = []
        for name in VOC_PROMPT_NAMES:
            prompts = [tpl.format(name) for tpl in templates]
            tokens = clip.tokenize(prompts).cuda()
            feats = clip_model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
            text_features_list.append(feats.mean(dim=0, keepdim=True))
        text_feature_bank = torch.cat(text_features_list, dim=0) # [21, 512]
        text_feature_bank /= text_feature_bank.norm(dim=-1, keepdim=True)
    print("CLIP Text Feature Bank encoded successfully.")

    colors = palette()
    hist_input = np.zeros((21, 21), dtype=np.int64)
    hist_clip = np.zeros((21, 21), dtype=np.int64)

    progress_file = EXP_DIR / "progress.json"
    status_file = EXP_DIR / "status.json"

    reassigned_count = 0
    suppressed_bg_count = 0
    total_components_checked = 0

    for idx, name in enumerate(val_names):
        img_path = DATA_DIR / "JPEGImages" / f"{name}.jpg"
        gt_path = DATA_DIR / "SegmentationClassAug" / f"{name}.png"
        mask_path = masks_source_dir / f"{name}.png"

        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)
        in_mask = np.array(Image.open(mask_path))
        gt = np.array(Image.open(gt_path))
        h, w, _ = img.shape

        hist_input += compute_hist(in_mask, gt)

        verified_mask = in_mask.copy()

        unique_classes = np.unique(in_mask)
        for c in unique_classes:
            if c == 0 or c == 255:
                continue
            class_mask = (in_mask == c)
            labeled, num_features = ndi.label(class_mask)
            for f_idx in range(1, num_features + 1):
                comp = (labeled == f_idx)
                area = comp.sum()
                if area < 200:
                    continue
                total_components_checked += 1

                # Check if class belongs to a known confusion set
                candidate_classes = CONFUSION_PAIRS.get(c, [c, 0])

                y_indices, x_indices = np.where(comp)
                pad_x = int(0.15 * (x_indices.max() - x_indices.min() + 1))
                pad_y = int(0.15 * (y_indices.max() - y_indices.min() + 1))
                x_min = max(0, x_indices.min() - pad_x)
                x_max = min(w, x_indices.max() + pad_x + 1)
                y_min = max(0, y_indices.min() - pad_y)
                y_max = min(h, y_indices.max() + pad_y + 1)

                crop_pil = pil_img.crop((x_min, y_min, x_max, y_max))
                crop_tensor = clip_preprocess(crop_pil).unsqueeze(0).cuda()

                with torch.no_grad():
                    img_feat = clip_model.encode_image(crop_tensor)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)
                    sims = (img_feat @ text_feature_bank.T).squeeze(0) # [21]

                c_curr_sim = sims[c].item()
                bg_sim = sims[0].item()

                # Rule 1: Background clutter suppression
                # If background similarity is much higher than object similarity for smallish components
                if bg_sim > (c_curr_sim + 0.08) and area < 800:
                    verified_mask[comp] = 0
                    suppressed_bg_count += 1
                    continue

                # Rule 2: High-confusion semantic rectification
                # E.g. diningtable vs chair, sofa vs chair
                best_comp_class = c
                best_comp_sim = c_curr_sim

                for alt_c in candidate_classes:
                    if alt_c == 0 or alt_c == c:
                        continue
                    alt_sim = sims[alt_c].item()
                    if alt_sim > best_comp_sim:
                        best_comp_sim = alt_sim
                        best_comp_class = alt_c

                # Reassign if alternate class is significantly more confident
                if best_comp_class != c and (best_comp_sim - c_curr_sim) > 0.05:
                    verified_mask[comp] = best_comp_class
                    reassigned_count += 1

        hist_clip += compute_hist(verified_mask, gt)

        # Save verified masks
        Image.fromarray(verified_mask.astype(np.uint8)).save(OUT_DIR / "masks" / f"{name}.png")
        Image.fromarray(colors[verified_mask]).save(OUT_DIR / "color" / f"{name}.png")

        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            elapsed = time.time() - start_time
            cur_miou_in, _, _, _ = compute_scores(hist_input)
            cur_miou_clip, _, _, _ = compute_scores(hist_clip)
            delta = (cur_miou_clip - cur_miou_in) * 100

            progress_data = {
                "processed": idx + 1,
                "total": total_images,
                "elapsed_seconds": elapsed,
                "input_mask_miou": cur_miou_in * 100,
                "clip_synergy_miou": cur_miou_clip * 100,
                "delta_pp": delta,
                "reassigned_count": reassigned_count,
                "suppressed_bg_count": suppressed_bg_count,
                "total_components": total_components_checked
            }
            with open(progress_file, "w") as pf:
                json.dump(progress_data, pf, indent=2)

            with open(status_file, "w") as sf:
                json.dump({
                    "stage": "running" if (idx + 1) < total_images else "completed",
                    "description": f"Processed {idx+1}/{total_images} images (Current CLIP mIoU: {cur_miou_clip*100:.2f}%)",
                    "progress": progress_data
                }, sf, indent=2)

            print(f"[{idx+1:4d}/{total_images}] Input: {cur_miou_in*100:.2f}% | CLIP-Synergy: {cur_miou_clip*100:.2f}% | Gain: {delta:+.2f}% | Reassigned: {reassigned_count} | Elapsed: {elapsed:.1f}s", flush=True)

    # Final scores
    final_miou_in, in_iou, in_prec, in_rec = compute_scores(hist_input)
    final_miou_clip, clip_iou, clip_prec, clip_rec = compute_scores(hist_clip)
    total_delta = (final_miou_clip - final_miou_in) * 100

    true_base_path = ROOT / "experiments/inference_voc_val_20261119/full/baseline/metrics.json"
    true_base_m = json.loads(true_base_path.read_text())["crf"]["miou"] * 100.0

    np.save(OUT_DIR / "clip_confusion.npy", hist_clip)
    np.save(OUT_DIR / "input_confusion.npy", hist_input)

    results = {
        "method": "SAM + CLIP Zero-Shot Co-occurrence Verification",
        "dataset": "Pascal VOC 2012 Val",
        "num_images": total_images,
        "true_baseline_crf_miou": true_base_m,
        "input_mask_miou": final_miou_in * 100,
        "clip_synergy_miou": final_miou_clip * 100,
        "net_gain_vs_baseline": (final_miou_clip * 100) - true_base_m,
        "gain_vs_input": total_delta,
        "reassigned_count": reassigned_count,
        "suppressed_bg_count": suppressed_bg_count,
        "per_class": {}
    }

    csv_rows = []
    for c_id, c_name in enumerate(VOC_CLASSES):
        i_iou = in_iou[c_id] * 100
        cl_iou = clip_iou[c_id] * 100
        d_iou = cl_iou - i_iou
        results["per_class"][c_name] = {
            "input_iou": i_iou,
            "clip_iou": cl_iou,
            "delta_vs_input": d_iou,
            "precision": clip_prec[c_id] * 100,
            "recall": clip_rec[c_id] * 100
        }
        csv_rows.append({
            "class": c_name,
            "input_iou": f"{i_iou:.2f}",
            "clip_iou": f"{cl_iou:.2f}",
            "delta_vs_input": f"{d_iou:+.2f}",
            "precision": f"{clip_prec[c_id]*100:.2f}",
            "recall": f"{clip_rec[c_id]*100:.2f}"
        })

    with open(OUT_DIR / "metrics.json", "w") as mf:
        json.dump(results, mf, indent=2)

    with open(EXP_DIR / "per_class_comparison.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["class", "input_iou", "clip_iou", "delta_vs_input", "precision", "recall"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 70)
    print("SAM + CLIP ZERO-SHOT EVALUATION COMPLETE (ALL 1449 IMAGES)")
    print(f"True Baseline CRF mIoU:     {true_base_m:.2f}%")
    print(f"Input Mask mIoU:            {final_miou_in*100:.2f}%")
    print(f"SAM + CLIP Synergy mIoU:    {final_miou_clip*100:.2f}%")
    print(f"NET GAIN OVER BASELINE:     +{(final_miou_clip*100) - true_base_m:.2f}%")
    print(f"Total Components Reassigned:{reassigned_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
