"""
USRC: Unsupervised SAM-Reliability Clustered Decoder Training
Implements Stage 2 of the USRC Framework:
Trains the ASPP segmentation decoder with frozen ViT backbone using:
- SAM pseudo-labels for Low-Risk classes (automatically determined by Otsu clustering)
- Teacher pseudo-labels for High-Risk classes (slender, porous, or composite categories)
- Balanced Cross-Entropy and Teacher Anchor KL divergence
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


class UsrcDataset:
    def __init__(self, data_folder, sam_cache_folder, teacher_cache_folder, low_risk_ids, high_risk_ids, name_list,
                 crop_size=448, short_range=(336, 672), jitter=0.2, seed=0):
        self.data_folder = Path(data_folder)
        self.sam_cache = Path(sam_cache_folder)
        self.teacher_cache = Path(teacher_cache_folder)
        self.low_risk_ids = list(low_risk_ids)
        self.high_risk_ids = list(high_risk_ids)
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
        teacher_path = self.teacher_cache / f"{name}.png"

        img = Image.open(img_path).convert("RGB")
        lbl_sam = np.array(Image.open(sam_path))

        if teacher_path.exists():
            lbl_teacher = np.array(Image.open(teacher_path))
            # Mathematically partitioned pseudo-label fusion:
            # 1. Low-risk classes: SAM foundation model provides superior geometric boundary refinement.
            # 2. High-risk classes (slender/porous/composite): Teacher predictions are substituted to preserve topological integrity.
            is_sam_low = np.isin(lbl_sam, self.low_risk_ids)
            is_tch_high = np.isin(lbl_teacher, self.high_risk_ids)
            lbl_fused = np.where(is_sam_low, lbl_sam, np.where(is_tch_high, lbl_teacher, lbl_sam)).astype(np.uint8)
        else:
            lbl_fused = lbl_sam

        img_t, lbl_t = transform_pair(
            img, Image.fromarray(lbl_fused), self.rng,
            crop_size=self.crop_size,
            short_range=self.short_range,
            jitter=self.jitter
        )
        return img_t, lbl_t


def balanced_cross_entropy(logits, labels):
    valid = (labels != 255)
    if not valid.any():
        return logits.sum() * 0.0

    ce_losses = F.cross_entropy(logits, labels, ignore_index=255, reduction="none")
    bg_mask = valid & (labels == 0)
    fg_mask = valid & (labels > 0)

    terms = []
    if bg_mask.any():
        terms.append(ce_losses[bg_mask].mean())
    if fg_mask.any():
        terms.append(ce_losses[fg_mask].mean())

    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def anchor_kl(student_logits, teacher_logits, temperature=2.0):
    p_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    log_q_student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_q_student, p_teacher, reduction="none").sum(dim=1).mean() * (temperature ** 2)


def main():
    parser = argparse.ArgumentParser(description="USRC Training")
    parser.add_argument("--steps", type=int, default=2500, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for ASPP decoder")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--anchor_weight", type=float, default=0.5, help="Weight for Teacher Anchor KL loss")
    parser.add_argument("--anchor_temp", type=float, default=2.0, help="Temperature for Anchor KL loss")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--log_interval", type=int, default=50, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--profile_path", type=str, default="experiments/usrc_decoder_training_20261121/class_risk_profile.json", help="Path to class risk profile JSON")
    parser.add_argument("--work_dir", type=str, default="experiments/usrc_decoder_training_20261121", help="Output directory")
    args = parser.parse_args()

    work_dir = ROOT / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(work_dir / "train.log")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 1. Load Class-Risk Profile
    profile_path = ROOT / args.profile_path if not os.path.isabs(args.profile_path) else Path(args.profile_path)
    if not profile_path.exists():
        logging.error(f"Class risk profile not found at {profile_path}! Run compute_class_risk.py first.")
        return
    profile = json.loads(profile_path.read_text())
    high_risk_ids = profile["high_risk_ids"]

    logging.info("=" * 60)
    logging.info("USRC: Unsupervised SAM-Reliability Clustered Decoder Training")
    logging.info(f"Target steps: {args.steps} | Batch size: {args.batch_size} | LR: {args.lr}")
    logging.info(f"Low-Risk SAM Classes ({len(profile['low_risk_classes'])}): {profile['low_risk_classes']}")
    logging.info(f"High-Risk Teacher Classes ({len(profile['high_risk_classes'])}): {profile['high_risk_classes']}")
    logging.info(f"Anchor weight: {args.anchor_weight} | Anchor temp: {args.anchor_temp} | EMA decay: {args.ema_decay}")
    logging.info("=" * 60)

    # 2. Load Baseline Model
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

    # 3. Setup Dataset
    data_folder = Path("/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012")
    sam_cache_folder = ROOT / "experiments/representation_sam_20261120/cache/sam/labels"
    teacher_cache_folder = ROOT / "experiments/decoder_selftrain_20261119/teacher_cache/raw_teacher"
    train_txt = ROOT / "datasets/voc/train_aug.txt"
    name_list = [x.strip() for x in train_txt.read_text().split() if x.strip()]
    logging.info(f"Loaded {len(name_list)} training image names from {train_txt}")

    dataset = UsrcDataset(
        data_folder=data_folder,
        sam_cache_folder=sam_cache_folder,
        teacher_cache_folder=teacher_cache_folder,
        low_risk_ids=profile["low_risk_ids"],
        high_risk_ids=high_risk_ids,
        name_list=name_list,
        crop_size=448,
        short_range=(336, 672),
        jitter=0.2,
        seed=args.seed
    )

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # 5. Training Loop
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

        # Forward
        with torch.no_grad():
            features = frozen_features(model, inputs)
            teacher_logits = teacher_decoder(features)

        with autocast():
            student_logits = model.decoder(features)
            dense_student = F.interpolate(student_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            ce_loss = balanced_cross_entropy(dense_student, labels)
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
            logging.info(
                f"[{step:04d}/{args.steps:04d}] "
                f"Loss: {total_loss.item():.4f} | "
                f"CE: {ce_loss.item():.4f} | "
                f"Anchor KL: {kl_loss.item():.4f} | "
                f"LR: {cur_lr:.2e} | "
                f"Speed: {speed:.2f} it/s | "
                f"ETA: {eta_sec / 60:.1f}m"
            )

        if step % args.save_interval == 0 or step == args.steps:
            student_sd = refined_state(model, model.decoder)
            ema_sd = refined_state(model, ema_decoder)
            torch.save(student_sd, ckpt_dir / f"model_usrc_step_{step:08d}_student.pth")
            torch.save(ema_sd, ckpt_dir / f"model_usrc_step_{step:08d}_ema.pth")
            logging.info(f"Saved checkpoints at step {step}")

    final_student = ckpt_dir / "model_usrc_final_student.pth"
    final_ema = ckpt_dir / "model_usrc_final_ema.pth"
    torch.save(refined_state(model, model.decoder), final_student)
    torch.save(refined_state(model, ema_decoder), final_ema)
    logging.info(f"Training Complete! Final models saved: {final_student} and {final_ema}")


if __name__ == "__main__":
    main()
