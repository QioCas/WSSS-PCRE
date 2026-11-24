"""
CSL-SD: Confidence Separable Soft-Distilled Decoder Training
Implements CSL (Confidence Separable Learning) soft-masking for WSSS Decoder Refinement:
- Estimates pixel reliability from Teacher confidence and residual dispersion via spectral relaxation
- Applies continuous Gaussian soft loss weights to mitigate overconfidence and boundary noise
- Smoothly attenuates gradient on disputed/porous boundary regions (e.g. chair legs, bicycle spokes)
- Preserves context continuity without hard pixel discarding or blurry teacher label injection
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.model_wsddn import network
from utils.decoder_selftraining import transform_pair, frozen_features, update_ema, refined_state


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )


class CslDataset:
    def __init__(self, data_folder, sam_cache_folder, name_list,
                 crop_size=448, short_range=(336, 672), jitter=0.2, seed=0):
        self.data_folder = Path(data_folder)
        self.sam_cache = Path(sam_cache_folder)
        self.names = name_list
        self.crop_size = crop_size
        self.short_range = short_range
        self.jitter = jitter
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        img_path = self.data_folder / "JPEGImages" / f"{name}.jpg"
        sam_path = self.sam_cache / f"{name}.png"

        img = Image.open(img_path).convert("RGB")
        lbl_sam = Image.open(sam_path)

        img_t, lbl_t = transform_pair(
            img, lbl_sam, self.rng,
            crop_size=self.crop_size,
            short_range=self.short_range,
            jitter=self.jitter
        )
        return img_t, lbl_t


@torch.no_grad()
def get_csl_soft_weights(probs_small, alpha=8.0, epsilon=1e-8):
    """
    Computes continuous reliability soft weights following CSL theory:
    - Maximum confidence p_n(k')
    - Residual dispersion v_n (variance among non-maximum classes)
    - Spectral relaxation via batched SVD for adaptive high-confidence center estimation
    - Gaussian weighting for smooth boundary attenuation
    """
    B, C, H, W = probs_small.shape
    max_conf, max_idx = torch.max(probs_small, dim=1)  # [B, H, W]
    one_hot = F.one_hot(max_idx, num_classes=C).permute(0, 3, 1, 2).float()
    rem = probs_small * (1.0 - one_hot)
    num_rem = max(1, C - 1)
    mean_rem = rem.sum(dim=1, keepdim=True) / num_rem
    diff_sq = ((rem - mean_rem) ** 2) * (1.0 - one_hot)
    res_var = diff_sq.sum(dim=1) / num_rem  # [B, H, W]

    # Flatten feature space: [B, 2, H*W]
    feat = torch.stack([max_conf.view(B, -1), res_var.view(B, -1)], dim=1)

    # Batched SVD: X: [B, 2, N]
    try:
        _, _, Vt = torch.linalg.svd(feat, full_matrices=False)  # Vt: [B, 2, N]
        assignments = torch.argmax(torch.abs(Vt), dim=1)  # [B, N]
    except Exception:
        # Fallback if SVD fails on rare singular edge cases
        return torch.ones((B, H, W), device=probs_small.device)

    mask0 = (assignments == 0).unsqueeze(1).float()  # [B, 1, N]
    mask1 = (assignments == 1).unsqueeze(1).float()  # [B, 1, N]

    n0 = mask0.sum(dim=-1).clamp(min=1.0)  # [B, 1]
    n1 = mask1.sum(dim=-1).clamp(min=1.0)  # [B, 1]

    mean0 = (feat * mask0).sum(dim=-1) / n0  # [B, 2]
    mean1 = (feat * mask1).sum(dim=-1) / n1  # [B, 2]

    pick1 = (mean1[:, 0] > mean0[:, 0]).unsqueeze(1)  # [B, 1]
    sel_mean = torch.where(pick1, mean1, mean0)  # [B, 2]
    sel_mask = torch.where(pick1.unsqueeze(-1), mask1, mask0)  # [B, 1, N]
    sel_n = torch.where(pick1, n1, n0)  # [B, 1]

    sel_var = ((feat - sel_mean.unsqueeze(-1)) ** 2 * sel_mask).sum(dim=-1) / sel_n.clamp(min=2.0)

    conf_mean = sel_mean[:, 0].view(B, 1, 1)
    res_mean = sel_mean[:, 1].view(B, 1, 1)
    conf_var = sel_var[:, 0].view(B, 1, 1)
    res_var_stat = sel_var[:, 1].view(B, 1, 1)

    conf_z = (max_conf - conf_mean) / torch.sqrt(conf_var + epsilon)
    res_z = (res_var - res_mean) / torch.sqrt(res_var_stat + epsilon)

    weight_conf = torch.exp(- (conf_z ** 2) / alpha)
    weight_res = torch.exp(- (res_z ** 2) / alpha)
    soft_w = weight_conf * weight_res

    confident_mask = (conf_z > 0) | (res_z > 0)
    weights = torch.where(confident_mask, torch.ones_like(soft_w), soft_w)
    return weights.clamp(min=0.1, max=1.0)


def soft_balanced_cross_entropy(logits, labels, weights):
    valid = (labels != 255)
    if not valid.any():
        return logits.sum() * 0.0

    ce = F.cross_entropy(logits, labels, ignore_index=255, reduction="none")
    weighted_ce = ce * weights

    bg_mask = valid & (labels == 0)
    fg_mask = valid & (labels > 0)

    terms = []
    if bg_mask.any():
        w_bg = weights[bg_mask].sum().clamp(min=1e-5)
        terms.append(weighted_ce[bg_mask].sum() / w_bg)
    if fg_mask.any():
        w_fg = weights[fg_mask].sum().clamp(min=1e-5)
        terms.append(weighted_ce[fg_mask].sum() / w_fg)

    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def anchor_kl(student_logits, teacher_logits, temperature=2.0):
    p_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    log_q_student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_q_student, p_teacher, reduction="none").sum(dim=1).mean() * (temperature ** 2)


def main():
    parser = argparse.ArgumentParser(description="CSL-SD Softmask Decoder Training")
    parser.add_argument("--steps", type=int, default=2500, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for ASPP decoder")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--anchor_weight", type=float, default=0.5, help="Weight for Teacher Anchor KL loss")
    parser.add_argument("--anchor_temp", type=float, default=2.0, help="Temperature for Anchor KL loss")
    parser.add_argument("--csl_alpha", type=float, default=8.0, help="CSL Gaussian smoothing scale parameter")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--log_interval", type=int, default=50, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--work_dir", type=str, default="experiments/csl_softmask_training_20261121", help="Output directory")
    args = parser.parse_args()

    work_dir = ROOT / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(work_dir / "train.log")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logging.info("=" * 65)
    logging.info("CSL-SD: Confidence Separable Soft-Distilled Decoder Training")
    logging.info(f"Target steps: {args.steps} | Batch size: {args.batch_size} | LR: {args.lr}")
    logging.info(f"CSL Alpha: {args.csl_alpha} | Anchor weight: {args.anchor_weight} | Anchor temp: {args.anchor_temp}")
    logging.info(f"Work dir: {work_dir}")
    logging.info("=" * 65)

    # 1. Load Baseline Model
    baseline_run = ROOT / "experiments/pcre_sat_20261117_200400_39e382/baseline"
    baseline_cfg = json.loads((baseline_run / "config.json").read_text())
    baseline_cfg["debug"] = False
    base_args = argparse.Namespace(**baseline_cfg)

    logging.info("Building model and loading baseline checkpoint...")
    model = network(
        base_args,
        backbone=base_args.backbone,
        num_classes=base_args.num_classes,
        pretrained=False,
        aux_layer=base_args.aux_layer
    )
    model.encoder.after_load_state_dict()
    ckpt_path = baseline_run / "checkpoints/model_iter_00020000.pth"
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    model.eval().cuda()

    # Freeze entire backbone
    model.requires_grad_(False)
    # Unfreeze ONLY decoder
    model.decoder.requires_grad_(True)
    teacher_decoder = copy.deepcopy(model.decoder).eval().requires_grad_(False)
    ema_decoder = copy.deepcopy(model.decoder).eval().requires_grad_(False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logging.info(f"Trainable parameters (Decoder only): {trainable_params:,}")
    logging.info(f"Frozen parameters (ViT Backbone etc.): {frozen_params:,}")

    # 2. Setup Dataset
    data_folder = Path("/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012")
    sam_cache_folder = ROOT / "experiments/representation_sam_20261120/cache/sam/labels"
    train_txt = ROOT / "datasets/voc/train_aug.txt"
    name_list = [x.strip() for x in train_txt.read_text().split() if x.strip()]
    logging.info(f"Loaded {len(name_list)} training image names from {train_txt}")

    dataset = CslDataset(
        data_folder=data_folder,
        sam_cache_folder=sam_cache_folder,
        name_list=name_list,
        crop_size=448,
        short_range=(336, 672),
        jitter=0.2,
        seed=args.seed
    )

    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # 4. Training Loop
    logging.info("Starting training loop...")
    order_rng = np.random.default_rng(args.seed)
    order = order_rng.permutation(len(dataset))
    cursor = 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = order_rng.permutation(len(dataset))
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size

        batch_examples = [dataset[int(i)] for i in indices]
        inputs = torch.stack([x[0] for x in batch_examples]).cuda(non_blocking=True)
        labels = torch.stack([x[1] for x in batch_examples]).cuda(non_blocking=True)

        # LR Schedule (Warmup + Cosine Decay)
        if step <= args.warmup_steps:
            cur_lr = args.lr * float(step) / float(args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
            cur_lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Forward Teacher (Frozen)
        with torch.no_grad():
            features = frozen_features(model, inputs)
            teacher_logits = teacher_decoder(features)  # [B, 21, 28, 28]

            # Compute CSL Soft Reliability Mask
            # 1. Teacher probabilities at 56x56 resolution for smooth spectral clustering
            t_probs_small = F.softmax(F.interpolate(teacher_logits, size=(56, 56), mode="bilinear", align_corners=False), dim=1)
            csl_w_small = get_csl_soft_weights(t_probs_small, alpha=args.csl_alpha)  # [B, 56, 56]
            csl_weights = F.interpolate(csl_w_small.unsqueeze(1), size=labels.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            # 2. Consensus Gating between SAM and Teacher
            t_pred_dense = F.interpolate(teacher_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
            # In disputed boundary regions where SAM and Teacher disagree on foreground classes:
            disagree = (labels != 255) & (labels > 0) & (t_pred_dense != labels)
            # Attenuate weight smoothly by CSL confidence to prevent memorizing boundary artifacts
            final_weights = torch.where(disagree, csl_weights * 0.5, csl_weights)

        # Forward Student (Trainable)
        with autocast():
            student_logits = model.decoder(features)
            dense_student = F.interpolate(student_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            ce_loss = soft_balanced_cross_entropy(dense_student, labels, final_weights)
            kl_loss = anchor_kl(student_logits, teacher_logits, temperature=args.anchor_temp)
            total_loss = ce_loss + args.anchor_weight * kl_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        update_ema(ema_decoder, model.decoder, decay=args.ema_decay)

        if step % args.log_interval == 0 or step == args.steps:
            elapsed = time.time() - t0
            speed = step / elapsed
            eta_sec = (args.steps - step) / speed if speed > 0 else 0
            mean_w = final_weights.mean().item()
            logging.info(
                f"[{step:04d}/{args.steps:04d}] "
                f"Loss: {total_loss.item():.4f} | "
                f"SoftCE: {ce_loss.item():.4f} | "
                f"Anchor KL: {kl_loss.item():.4f} | "
                f"AvgWeight: {mean_w:.3f} | "
                f"LR: {cur_lr:.2e} | "
                f"Speed: {speed:.2f} it/s | "
                f"ETA: {eta_sec / 60:.1f}m"
            )

        if step % args.save_interval == 0 or step == args.steps:
            student_sd = refined_state(model, model.decoder)
            ema_sd = refined_state(model, ema_decoder)
            torch.save(student_sd, ckpt_dir / f"model_csl_step_{step:08d}_student.pth")
            torch.save(ema_sd, ckpt_dir / f"model_csl_step_{step:08d}_ema.pth")
            logging.info(f"Saved checkpoints at step {step}")

    final_student = ckpt_dir / "model_csl_final_student.pth"
    final_ema = ckpt_dir / "model_csl_final_ema.pth"
    torch.save(refined_state(model, model.decoder), final_student)
    torch.save(refined_state(model, ema_decoder), final_ema)
    logging.info(f"Training Complete! Final models saved: {final_student} and {final_ema}")


if __name__ == "__main__":
    main()
