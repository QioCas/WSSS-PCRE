"""
Full evaluation of USRC checkpoint on VOC 2012 Val set (1449 images).
Evaluates both Raw Decoder prediction and DenseCRF post-processed prediction.
Directly compares with true baseline PCRE metrics from inference_voc_val_20261119.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT.parents[1] / ".deps/pcre_inference"
sys.path[:0] = [str(ROOT), str(DEPS)]

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from datasets.voc import VOC12SegDataset, class_list
from model.model_wsddn import network
from utils.evaluate import compute_final_scores

CRF_PARAMETERS = {
    "iter_max": 10,
    "pos_xy_std": 2,
    "pos_w": 1,
    "bi_xy_std": 140,
    "bi_rgb_std": 1,
    "bi_w": 7,
}


def hist_for(label, prediction, n_classes=21):
    valid = (label >= 0) & (label < n_classes)
    return np.bincount(
        n_classes * label[valid].astype(int) + prediction[valid].astype(int),
        minlength=n_classes ** 2,
    ).reshape(n_classes, n_classes)


def image_score(hist):
    union = hist.sum(0) + hist.sum(1) - hist.diagonal()
    present = hist.sum(1) > 0
    return float(np.mean(hist.diagonal()[present] / union[present]))


def crf_image(image_path, label_path, probabilities, name):
    image = np.array(Image.open(image_path).convert("RGB"))
    gt = np.array(Image.open(label_path))
    _, height, width = probabilities.shape
    model = dcrf.DenseCRF2D(width, height, probabilities.shape[0])
    model.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(probabilities)))
    model.addPairwiseGaussian(sxy=CRF_PARAMETERS["pos_xy_std"], compat=CRF_PARAMETERS["pos_w"])
    model.addPairwiseBilateral(sxy=CRF_PARAMETERS["bi_xy_std"], srgb=CRF_PARAMETERS["bi_rgb_std"],
                               rgbim=np.ascontiguousarray(image), compat=CRF_PARAMETERS["bi_w"])
    q = np.array(model.inference(CRF_PARAMETERS["iter_max"])).reshape(probabilities.shape)
    prediction = q.argmax(0).astype(np.uint8)
    hist = hist_for(gt, prediction)
    return name, hist, image_score(hist)


def main():
    parser = argparse.ArgumentParser(description="Evaluate USRC checkpoint on VOC Val")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--output_dir", type=str, default="experiments/usrc_decoder_training_20261121/eval_results")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.25, 1.5])
    parser.add_argument("--crf_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 for all 1449 images")
    parser.add_argument("--data_folder", type=str, default="/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012")
    parser.add_argument("--list_folder", type=str, default=str(ROOT / "datasets/voc"))
    opts = parser.parse_args()

    out_dir = Path(opts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load True Baseline Metrics
    true_base_path = ROOT / "experiments/inference_voc_val_20261119/full/baseline/metrics.json"
    true_baseline = json.loads(true_base_path.read_text())

    config_path = ROOT / "experiments/pcre_sat_20261117_200400_39e382/baseline/config.json"
    with open(config_path) as f:
        config = json.load(f)
    cfg = argparse.Namespace(**config)
    cfg.data_folder = opts.data_folder
    cfg.list_folder = opts.list_folder
    cfg.debug = False

    print(f"Loading checkpoint: {opts.checkpoint}")
    model = network(cfg, backbone=cfg.backbone, num_classes=cfg.num_classes, pretrained=False, aux_layer=cfg.aux_layer)
    model.encoder.after_load_state_dict()

    state = torch.load(opts.checkpoint, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    loaded = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint keys. Missing: {len(loaded.missing_keys)}, Unexpected: {len(loaded.unexpected_keys)}")

    model.eval().cuda()

    dataset = VOC12SegDataset(root_dir=cfg.data_folder, name_list_dir=cfg.list_folder,
                              split="val", stage="val", aug=False, ignore_index=cfg.ignore_index)
    names = list(dataset.name_list)
    total = min(opts.limit, len(names)) if opts.limit else len(names)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    raw_hist = np.zeros((21, 21), dtype=np.int64)
    crf_hist = np.zeros((21, 21), dtype=np.int64)
    per_image = {}
    pending = []

    def receive():
        nonlocal crf_hist
        if not pending:
            return
        fut = pending.pop(0)
        im_name, hist, score = fut.result()
        crf_hist += hist
        per_image[im_name]["crf_miou"] = score

    started = time.time()
    print(f"Starting evaluation on {total} images with scales {opts.scales} and DenseCRF...")
    with ProcessPoolExecutor(max_workers=opts.crf_workers, mp_context=mp.get_context("spawn")) as executor:
        with torch.no_grad():
            for index, (name, inputs, labels, _) in enumerate(loader):
                if opts.limit > 0 and index >= total:
                    break
                name = name[0]
                inputs = inputs.cuda(non_blocking=True)
                scaled = []
                for scale in opts.scales:
                    size = (int(448 * scale), int(448 * scale))
                    resized = F.interpolate(inputs, size=size, mode="bilinear", align_corners=False)
                    flipped = torch.cat([resized, resized.flip(-1)], dim=0)
                    segs = model(flipped, args=cfg)[1]
                    segs = F.interpolate(segs, size=labels.shape[1:], mode="bilinear", align_corners=False)
                    scaled.append(segs[:1] + segs[1:].flip(-1))

                stack = torch.stack(scaled, dim=0)
                seg = torch.maximum(stack[0], stack.mean(dim=0))
                prediction = seg.argmax(1)[0].cpu().numpy().astype(np.uint8)
                gt = labels[0].numpy()
                hist = hist_for(gt, prediction)
                raw_hist += hist

                per_image[name] = {
                    "raw_miou": image_score(hist),
                    "classes_in_gt": [int(c) for c in np.unique(gt) if c < 21]
                }
                probabilities = F.softmax(seg, dim=1)[0].cpu().numpy()

                image_path = str(Path(cfg.data_folder) / "JPEGImages" / f"{name}.jpg")
                label_path = str(Path(cfg.data_folder) / "SegmentationClassAug" / f"{name}.png")
                pending.append(executor.submit(crf_image, image_path, label_path, probabilities, name))

                if len(pending) >= 2 * opts.crf_workers:
                    receive()

                if (index + 1) % 100 == 0 or (index + 1) == total:
                    elapsed = time.time() - started
                    speed = (index + 1) / elapsed
                    eta = (total - (index + 1)) / speed if speed > 0 else 0
                    print(f"[{index+1:04d}/{total:04d}] Processed | Speed: {speed:.2f} img/s | ETA: {eta/60:.1f}m", flush=True)

            while pending:
                receive()

    raw_scores = compute_final_scores(raw_hist)
    crf_scores = compute_final_scores(crf_hist)
    elapsed_total = time.time() - started

    metrics = {
        "checkpoint": opts.checkpoint,
        "total_images": total,
        "elapsed_seconds": elapsed_total,
        "raw": raw_scores,
        "crf": crf_scores,
        "per_image": per_image,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-class comparison CSV against True Baseline
    csv_lines = ["Class,Baseline_CRF,USRC_Raw,USRC_CRF,Delta_CRF"]
    for i, cls_name in enumerate(class_list):
        b_score = true_baseline["crf"]["iou"].get(str(i), true_baseline["crf"]["iou"].get(i, 0.0)) * 100.0
        r_val = raw_scores["iou"].get(i, raw_scores["iou"].get(str(i), 0.0))
        c_val = crf_scores["iou"].get(i, crf_scores["iou"].get(str(i), 0.0))
        r_score = r_val * 100.0
        c_score = c_val * 100.0
        delta = c_score - b_score
        csv_lines.append(f"{cls_name},{b_score:.2f},{r_score:.2f},{c_score:.2f},{delta:+.2f}")

    b_mean = true_baseline["crf"]["miou"] * 100.0
    r_mean = raw_scores["miou"] * 100.0
    c_mean = crf_scores["miou"] * 100.0
    delta_mean = c_mean - b_mean
    csv_lines.append(f"Mean (mIoU),{b_mean:.2f},{r_mean:.2f},{c_mean:.2f},{delta_mean:+.2f}")

    with open(out_dir / "per_class_comparison.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    print("\n" + "=" * 60)
    print(f"USRC Evaluation Complete in {elapsed_total/60:.2f} minutes")
    print(f"True Baseline CRF: {b_mean:.2f}% | True Baseline Raw: {true_baseline['raw']['miou']*100:.2f}%")
    print(f"USRC Raw Decoder:  {r_mean:.2f}% (Delta Raw: {r_mean - true_baseline['raw']['miou']*100:+.2f}%)")
    print(f"USRC DenseCRF:     {c_mean:.2f}% (Delta CRF: {delta_mean:+.2f}%)")
    print("=" * 60)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
