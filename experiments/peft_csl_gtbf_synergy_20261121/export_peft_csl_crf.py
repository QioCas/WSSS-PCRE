"""
Step 1 of Synergy Pipeline:
Generates and saves the 1,449 DenseCRF predictions from the PEFT-CSL Final EMA model.
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


def crf_worker(image_path, label_path, probabilities, name, save_dir):
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

    # Save CRF mask as PNG
    if save_dir:
        Image.fromarray(prediction).save(os.path.join(save_dir, f"{name}.png"))

    hist = hist_for(gt, prediction)
    return name, hist, image_score(hist)


def main():
    parser = argparse.ArgumentParser(description="Export PEFT-CSL CRF Masks")
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/peft_csl_training_20261121/checkpoints/model_peft_csl_final_ema.pth")
    parser.add_argument("--save_dir", type=str,
                        default="experiments/peft_csl_gtbf_synergy_20261121/crf_masks")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.25, 1.5])
    parser.add_argument("--crf_workers", type=int, default=4)
    parser.add_argument("--data_folder", type=str,
                        default="/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012")
    parser.add_argument("--list_folder", type=str, default=str(ROOT / "datasets/voc"))
    opts = parser.parse_args()

    save_dir = Path(opts.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Loaded keys. Missing: {len(loaded.missing_keys)}, Unexpected: {len(loaded.unexpected_keys)}")
    model.eval().cuda()

    dataset = VOC12SegDataset(root_dir=cfg.data_folder, name_list_dir=cfg.list_folder,
                              split="val", stage="val", aug=False, ignore_index=cfg.ignore_index)
    total = len(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    crf_hist = np.zeros((21, 21), dtype=np.int64)
    pending = []

    def receive():
        nonlocal crf_hist
        if not pending:
            return
        fut = pending.pop(0)
        im_name, hist, score = fut.result()
        crf_hist += hist

    started = time.time()
    print(f"Exporting DenseCRF masks for {total} images to {save_dir}...")
    with ProcessPoolExecutor(max_workers=opts.crf_workers, mp_context=mp.get_context("spawn")) as executor:
        with torch.no_grad():
            for index, (name, inputs, labels, _) in enumerate(loader):
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
                probabilities = F.softmax(seg, dim=1)[0].cpu().numpy()

                image_path = str(Path(cfg.data_folder) / "JPEGImages" / f"{name}.jpg")
                label_path = str(Path(cfg.data_folder) / "SegmentationClassAug" / f"{name}.png")
                pending.append(executor.submit(crf_worker, image_path, label_path, probabilities, name, str(save_dir)))

                if len(pending) >= 2 * opts.crf_workers:
                    receive()

                if (index + 1) % 100 == 0 or (index + 1) == total:
                    elapsed = time.time() - started
                    speed = (index + 1) / elapsed
                    eta = (total - (index + 1)) / speed if speed > 0 else 0
                    print(f"[{index+1:04d}/{total:04d}] Processed | Speed: {speed:.2f} img/s | ETA: {eta/60:.1f}m", flush=True)

            while pending:
                receive()

    crf_scores = compute_final_scores(crf_hist)
    elapsed_total = time.time() - started
    print(f"CRF Mask Export Complete in {elapsed_total/60:.2f}m. CRF mIoU: {crf_scores['miou']*100:.2f}%")


if __name__ == "__main__":
    main()
