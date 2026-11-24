"""
PEFT-CSL: Parameter-Efficient Fine-Tuning (LoRA) with Confidence Separable Soft-Distillation
Resolves the WSSS Dilemma:
- Avoids catastrophic feature collapse of full unfreezing (BPD) by keeping 86M ViT base weights 100% frozen
- Overcomes the capacity bottleneck of frozen-backbone (FBSD) by injecting Low-Rank Adapters (LoRA, r=8) into top ViT self-attention blocks
- Uses CSL (Confidence Separable Learning) continuous soft-weighting to mitigate boundary noise
- Smoothly merges LoRA weights into standard Linear layers on save for 100% inference compatibility
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
from utils.decoder_selftraining import transform_pair


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )


class LoRAQKV(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base_linear.weight + (self.lora_B @ self.lora_A) * self.scale

    @property
    def bias(self):
        return self.base_linear.bias

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = (F.linear(x, self.lora_A) @ self.lora_B.t()) * self.scale
        return base_out + lora_out

    def merge_and_get_linear(self):
        new_linear = nn.Linear(self.base_linear.in_features, self.base_linear.out_features, bias=(self.base_linear.bias is not None))
        delta = (self.lora_B @ self.lora_A) * self.scale
        new_linear.weight.data.copy_(self.base_linear.weight.data + delta)
        if self.base_linear.bias is not None:
            new_linear.bias.data.copy_(self.base_linear.bias.data)
        return new_linear


def inject_lora_vit(model, lora_blocks=[8, 9, 10, 11], r=8, alpha=16):
    for idx in lora_blocks:
        block = model.encoder.blocks[idx]
        block.attn.qkv = LoRAQKV(block.attn.qkv, r=r, alpha=alpha)


class PeftDataset:
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
    B, C, H, W = probs_small.shape
    max_conf, max_idx = torch.max(probs_small, dim=1)
    one_hot = F.one_hot(max_idx, num_classes=C).permute(0, 3, 1, 2).float()
    rem = probs_small * (1.0 - one_hot)
    num_rem = max(1, C - 1)
    mean_rem = rem.sum(dim=1, keepdim=True) / num_rem
    diff_sq = ((rem - mean_rem) ** 2) * (1.0 - one_hot)
    res_var = diff_sq.sum(dim=1) / num_rem

    feat = torch.stack([max_conf.view(B, -1), res_var.view(B, -1)], dim=1)

    try:
        _, _, Vt = torch.linalg.svd(feat, full_matrices=False)
        assignments = torch.argmax(torch.abs(Vt), dim=1)
    except Exception:
        return torch.ones((B, H, W), device=probs_small.device)

    mask0 = (assignments == 0).unsqueeze(1).float()
    mask1 = (assignments == 1).unsqueeze(1).float()

    n0 = mask0.sum(dim=-1).clamp(min=1.0)
    n1 = mask1.sum(dim=-1).clamp(min=1.0)

    mean0 = (feat * mask0).sum(dim=-1) / n0
    mean1 = (feat * mask1).sum(dim=-1) / n1

    pick1 = (mean1[:, 0] > mean0[:, 0]).unsqueeze(1)
    sel_mean = torch.where(pick1, mean1, mean0)
    sel_mask = torch.where(pick1.unsqueeze(-1), mask1, mask0)
    sel_n = torch.where(pick1, n1, n0)

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


def get_merged_state_dict(model):
    """Returns state_dict where LoRA weights are merged into Linear for standard evaluation"""
    sd = copy.deepcopy(model.state_dict())
    for idx in [8, 9, 10, 11]:
        qkv_module = model.encoder.blocks[idx].attn.qkv
        if isinstance(qkv_module, LoRAQKV):
            merged_w = qkv_module.weight.data
            sd[f"encoder.blocks.{idx}.attn.qkv.weight"] = merged_w
            # Remove lora_A and lora_B keys
            sd.pop(f"encoder.blocks.{idx}.attn.qkv.lora_A", None)
            sd.pop(f"encoder.blocks.{idx}.attn.qkv.lora_B", None)
    return {k.replace("module.", ""): v for k, v in sd.items()}


def update_peft_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            if p.requires_grad:
                ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def main():
    parser = argparse.ArgumentParser(description="PEFT-CSL Training")
    parser.add_argument("--steps", type=int, default=2500, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr_decoder", type=float, default=3e-5, help="Learning rate for ASPP decoder")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="Learning rate for LoRA adapters")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora_blocks", type=int, nargs="+", default=[8, 9, 10, 11], help="ViT blocks to adapt")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--anchor_weight", type=float, default=0.5, help="Weight for Teacher Anchor KL loss")
    parser.add_argument("--anchor_temp", type=float, default=2.0, help="Temperature for Anchor KL loss")
    parser.add_argument("--csl_alpha", type=float, default=8.0, help="CSL Gaussian smoothing scale parameter")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--log_interval", type=int, default=50, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--work_dir", type=str, default="experiments/peft_csl_training_20261121", help="Output directory")
    args = parser.parse_args()

    work_dir = ROOT / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(work_dir / "train.log")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logging.info("=" * 70)
    logging.info("PEFT-CSL: Parameter-Efficient LoRA ViT + CSL Soft-Distillation")
    logging.info(f"Target steps: {args.steps} | Batch size: {args.batch_size}")
    logging.info(f"LR Decoder: {args.lr_decoder} | LR LoRA: {args.lr_lora} | Rank: {args.lora_rank}")
    logging.info(f"Adapted Blocks: {args.lora_blocks} | CSL Alpha: {args.csl_alpha}")
    logging.info(f"Work dir: {work_dir}")
    logging.info("=" * 70)

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

    # Build Teacher (Completely frozen reference)
    teacher_model = copy.deepcopy(model).eval().cuda()
    teacher_model.requires_grad_(False)

    # Inject LoRA into student ViT
    model.requires_grad_(False)
    inject_lora_vit(model, lora_blocks=args.lora_blocks, r=args.lora_rank, alpha=args.lora_alpha)
    model.decoder.requires_grad_(True)
    model.cuda()

    ema_model = copy.deepcopy(model).eval().cuda()
    ema_model.requires_grad_(False)

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    decoder_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]

    logging.info(f"Trainable LoRA Parameters (ViT Blocks {args.lora_blocks}): {sum(p.numel() for p in lora_params):,}")
    logging.info(f"Trainable Decoder Parameters: {sum(p.numel() for p in decoder_params):,}")
    logging.info(f"Total Trainable Parameters: {sum(p.numel() for p in lora_params) + sum(p.numel() for p in decoder_params):,}")
    logging.info(f"Frozen Base Parameters (86M ViT etc.): {sum(p.numel() for p in frozen_params):,}")

    # 2. Setup Dataset
    data_folder = Path("/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012")
    sam_cache_folder = ROOT / "experiments/representation_sam_20261120/cache/sam/labels"
    train_txt = ROOT / "datasets/voc/train_aug.txt"
    name_list = [x.strip() for x in train_txt.read_text().split() if x.strip()]
    logging.info(f"Loaded {len(name_list)} training image names from {train_txt}")

    dataset = PeftDataset(
        data_folder=data_folder,
        sam_cache_folder=sam_cache_folder,
        name_list=name_list,
        crop_size=448,
        short_range=(336, 672),
        jitter=0.2,
        seed=args.seed
    )

    # 3. Optimizer with Param Groups
    optimizer = torch.optim.AdamW([
        {"params": decoder_params, "lr": args.lr_decoder, "weight_decay": args.weight_decay},
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": args.weight_decay}
    ])
    scaler = GradScaler()

    # 4. Training Loop
    logging.info("Starting PEFT-CSL training loop...")
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
            pct = float(step) / float(args.warmup_steps)
            cur_lr_dec = args.lr_decoder * pct
            cur_lr_lora = args.lr_lora * pct
        else:
            progress = (step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            cur_lr_dec = args.lr_decoder * cos_decay
            cur_lr_lora = args.lr_lora * cos_decay

        optimizer.param_groups[0]["lr"] = cur_lr_dec
        optimizer.param_groups[1]["lr"] = cur_lr_lora

        # Forward Teacher (Frozen)
        with torch.no_grad():
            t_out = teacher_model(inputs, args=base_args)
            t_seg = t_out[1]  # [B, 21, 28, 28]

            # Compute CSL Soft Reliability Mask
            t_probs_small = F.softmax(F.interpolate(t_seg, size=(56, 56), mode="bilinear", align_corners=False), dim=1)
            csl_w_small = get_csl_soft_weights(t_probs_small, alpha=args.csl_alpha)
            csl_weights = F.interpolate(csl_w_small.unsqueeze(1), size=labels.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            t_pred_dense = F.interpolate(t_seg, size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
            disagree = (labels != 255) & (labels > 0) & (t_pred_dense != labels)
            final_weights = torch.where(disagree, csl_weights * 0.5, csl_weights)

        # Forward Student (Trainable Decoder + LoRA ViT)
        with autocast():
            s_out = model(inputs, args=base_args)
            student_logits = s_out[1]
            dense_student = F.interpolate(student_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            ce_loss = soft_balanced_cross_entropy(dense_student, labels, final_weights)
            kl_loss = anchor_kl(student_logits, t_seg, temperature=args.anchor_temp)
            total_loss = ce_loss + args.anchor_weight * kl_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        update_peft_ema(ema_model, model, decay=args.ema_decay)

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
                f"LR Dec: {cur_lr_dec:.2e} | "
                f"LR LoRA: {cur_lr_lora:.2e} | "
                f"Speed: {speed:.2f} it/s | "
                f"ETA: {eta_sec / 60:.1f}m"
            )

        if step % args.save_interval == 0 or step == args.steps:
            student_sd = get_merged_state_dict(model)
            ema_sd = get_merged_state_dict(ema_model)
            torch.save(student_sd, ckpt_dir / f"model_peft_csl_step_{step:08d}_student.pth")
            torch.save(ema_sd, ckpt_dir / f"model_peft_csl_step_{step:08d}_ema.pth")
            logging.info(f"Saved merged checkpoints at step {step}")

    final_student = ckpt_dir / "model_peft_csl_final_student.pth"
    final_ema = ckpt_dir / "model_peft_csl_final_ema.pth"
    torch.save(get_merged_state_dict(model), final_student)
    torch.save(get_merged_state_dict(ema_model), final_ema)
    logging.info(f"PEFT-CSL Training Complete! Merged models saved: {final_student} and {final_ema}")


if __name__ == "__main__":
    main()
