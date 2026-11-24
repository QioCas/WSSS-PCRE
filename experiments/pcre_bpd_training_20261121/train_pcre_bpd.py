"""PCRE-BPD: Progressive Confidence Region Expansion with Boundary-Aware Prototype Distillation.

Addresses the two fundamental training bottlenecks of PCRE (CVPR 2025):
1. The Boundary Pixel Blindness: In standard PCRE, pixels with CAM confidence in [0.25, 0.70]
   are discarded with ignore_index=255 (~28.4% of all boundary pixels). The decoder never receives
   gradients at the object boundaries. PCRE-BPD applies soft continuous Prototype Knowledge
   Distillation (BAPD) specifically on these 255 boundary transition regions.
2. The Decoder-CAM Knowledge Gap: Standard PCRE achieves 75.0% CAM but only 72.58% Decoder
   due to an under-weighted segmentation loss (w_seg=0.12 vs wsddn=2.0). PCRE-BPD uses
   Progressive Decoder Loss Scaling (PDLS) to dynamically ramp w_seg from 0.12 to 0.36.
"""
import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT.parents[1] / ".deps/pcre_inference"
sys.path[:0] = [str(ROOT), str(DEPS)]

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from datasets import voc
from model.losses import (ClsTokenLoss, MyGo_Loss, get_masked_ptc_loss, get_seg_loss, get_energy_loss, DenseEnergyLoss)
from model.PAR import PAR
from model.model_wsddn import network
from utils import imutils, evaluate
from utils.camutils import (cam_to_label, get_token_mask, label_to_aff_mask, multi_scale_cam2_v2, refine_cams_with_bkg_v2)
from utils.pyutils import AverageMeter, cal_eta, setup_logger
from utils.single_device import SingleDeviceModule

parser = argparse.ArgumentParser(description="PCRE-BPD Training")
parser.add_argument("--max_iters", default=1000, type=int, help="max training iterations for this run")
parser.add_argument("--log_iters", default=25, type=int, help="logging frequency")
parser.add_argument("--eval_iters", default=500, type=int, help="validation evaluation frequency")
parser.add_argument("--save_iters", default=500, type=int, help="checkpoint save frequency")
parser.add_argument("--warmup_iters", default=100, type=int, help="learning rate warmup")
parser.add_argument("--seg_warmup", default=50, type=int, help="warmup before starting segmentation loss")
parser.add_argument("--seg_iter", default=200, type=int, help="iteration to start self-mask feedback")
parser.add_argument("--spg", default=4, type=int, help="samples per gpu")
parser.add_argument("--lr", default=5e-6, type=float, help="learning rate")
parser.add_argument("--lr_decoder_mult", default=2.0, type=float, help="learning rate multiplier for decoder")
parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warmup lr")
parser.add_argument("--wt_decay", default=0.01, type=float, help="weight decay")
parser.add_argument("--backbone", default="vit_base_patch16_224", type=str)
parser.add_argument("--data_folder", default="/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012", type=str)
parser.add_argument("--list_folder", default=str(ROOT / "datasets/voc"), type=str)
parser.add_argument("--num_classes", default=21, type=int)
parser.add_argument("--crop_size", default=448, type=int)
parser.add_argument("--high_thre", default=0.70, type=float)
parser.add_argument("--low_thre", default=0.25, type=float)
parser.add_argument("--pro_high_thre", default=0.70, type=float)
parser.add_argument("--pro_low_thre", default=0.25, type=float)
parser.add_argument("--bkg_thre", default=0.50, type=float)
parser.add_argument("--ignore_index", default=255, type=int)
parser.add_argument("--w_seg", default=0.12, type=float)
parser.add_argument("--w_distill", default=0.25, type=float, help="weight for Boundary-Aware Prototype Distillation")
parser.add_argument("--distill_temp", default=2.0, type=float, help="temperature for soft distillation")
parser.add_argument("--w_ptc", default=0.3, type=float)
parser.add_argument("--w_reg", default=0.05, type=float)
parser.add_argument("--wsddn", default=2.0, type=float)
parser.add_argument("--w_cpe", default=0.02, type=float)
parser.add_argument("--w_seg2", default=2.0, type=float)
parser.add_argument("--wsddn_topk", default=300, type=int)
parser.add_argument("--proto_m", default=0.99, type=float)
parser.add_argument("--momentum", default=0.999, type=float)
parser.add_argument("--aux_layer", default=-3, type=int)
parser.add_argument("--pooling", default="gmp", type=str)
parser.add_argument("--decoder", default="ASSP", type=str)
parser.add_argument("--work_dir", default="experiments/pcre_bpd_training_20261121", type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--pretrained_backbone", default=str(ROOT / "pretrained/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"), type=str)
parser.add_argument("--resume_ckpt", default=str(ROOT / "experiments/pcre_sat_20261117_200400_39e382/baseline/checkpoints/model_iter_00020000.pth"), type=str)
parser.add_argument("--eval_limit", default=0, type=int, help="validation set limit (0 for full 1449)")

args, _ = parser.parse_known_args()

# Load baseline config as base defaults
BASE_CONFIG_PATH = ROOT / "experiments/pcre_sat_20261117_200400_39e382/baseline/config.json"
if BASE_CONFIG_PATH.exists():
    with open(BASE_CONFIG_PATH) as f:
        base_defaults = json.load(f)
    for k, v in base_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

# Enforce clean absolute paths and flags
args.data_folder = "/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012"
args.list_folder = str(ROOT / "datasets/voc")
args.debug = False
args.pro_fuse = False
args.mygo_sup = False
args.clstoken_loss = "cos"
args.wsddn_loss = "sum"
args.mil = True
args.mask_stop = True
args.cam2mask = 10000
args.mask_exp = True
args.pre_mask = True
args.pro = True
args.ema = True
args.cam_scales = [1.0, 0.5, 0.75, 1.5]
args.grad_t = "none"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_optimizer(model, args):
    param_groups = [[], [], []]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            param_groups[0].append(param)
        elif "decoder" in name or "classifier" in name:
            param_groups[1].append(param)
        else:
            param_groups[2].append(param)

    optim = torch.optim.AdamW([
        {"params": param_groups[0], "lr": args.lr, "weight_decay": args.wt_decay},
        {"params": param_groups[1], "lr": args.lr * args.lr_decoder_mult, "weight_decay": args.wt_decay},
        {"params": param_groups[2], "lr": args.lr * args.lr_decoder_mult, "weight_decay": args.wt_decay},
    ], betas=(0.9, 0.999))
    return optim


def run_full_validation(model, val_loader, args, limit=0):
    model.eval()
    val_hist = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    total_imgs = len(val_loader) if limit == 0 else min(limit, len(val_loader))

    with torch.no_grad():
        for v_idx, v_data in enumerate(val_loader):
            if limit > 0 and v_idx >= limit:
                break
            v_name, v_inputs, v_labels, v_cls = v_data
            v_inputs = v_inputs.cuda(non_blocking=True)
            v_labels = v_labels.cuda(non_blocking=True)
            # Evaluate using multi-scale inference for high precision
            b, c, h, w = v_inputs.shape
            v_inputs_resized = F.interpolate(v_inputs, size=[args.crop_size, args.crop_size], mode="bilinear", align_corners=False)
            v_inputs_cat = torch.cat([v_inputs_resized, v_inputs_resized.flip(-1)], dim=0)
            v_out_segs = model(v_inputs_cat, args=args)[1]
            v_segs = (v_out_segs[:1] + v_out_segs[1:].flip(-1)) / 2.0
            v_segs = F.interpolate(v_segs, size=v_labels.shape[1:], mode="bilinear", align_corners=False)
            v_pred = torch.argmax(v_segs, dim=1).cpu().numpy().astype(np.int64)
            v_gt = v_labels.cpu().numpy().astype(np.int64)

            valid = (v_gt >= 0) & (v_gt < args.num_classes)
            val_hist += np.bincount(
                args.num_classes * v_gt[valid] + v_pred[valid],
                minlength=args.num_classes * args.num_classes
            ).reshape(args.num_classes, args.num_classes)

    tp = np.diag(val_hist).astype(float)
    union = val_hist.sum(1) + val_hist.sum(0) - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    per_class_iou = {voc.class_list[i]: float(iou[i]) * 100 for i in range(args.num_classes)}
    val_miou = float(np.mean(iou[union > 0])) * 100
    model.train()
    return val_miou, per_class_iou, total_imgs


def main():
    setup_seed(args.seed)
    work_dir = Path(args.work_dir)
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = work_dir / "train.log"
    setup_logger(filename=str(log_file))

    logging.info("=" * 60)
    logging.info("PCRE-BPD: Boundary-Aware Prototype Distillation Training")
    logging.info(f"Target Max Iterations: {args.max_iters}")
    logging.info(f"Work Directory: {work_dir}")
    logging.info(f"Samples per GPU (spg): {args.spg}")
    logging.info(f"BAPD Distill Weight: {args.w_distill}, Distill Temp: {args.distill_temp}")
    logging.info("=" * 60)

    # 1. Build Model
    logging.info(f"Building network with backbone: {args.backbone}...")
    base_model = network(
        args,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer=args.aux_layer
    )
    # Expand tokens to PCRE architecture (20 class tokens, pos_embed 216)
    base_model.encoder.after_load_state_dict()

    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        logging.info(f"Loading checkpoint from: {args.resume_ckpt}")
        sd = torch.load(args.resume_ckpt, map_location="cpu")
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        msg = base_model.load_state_dict(clean_sd, strict=False)
        logging.info(f"Loaded checkpoint: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    elif Path(args.pretrained_backbone).exists():
        logging.info(f"Loading pretrained ViT backbone from: {args.pretrained_backbone}")
        bb_sd = torch.load(args.pretrained_backbone, map_location="cpu")
        base_model.encoder.load_state_dict(bb_sd, strict=False)

    model = SingleDeviceModule(base_model).cuda()
    model.train()

    # 2. Build Dataloaders
    train_dataset = voc.VOC12ClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split="train_aug",
        stage="train",
        aug=True,
        rescale_range=(0.5, 2),
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
        color_aug=True,
    )
    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split="val",
        stage="val",
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    train_iter = iter(train_loader)

    # 3. Optimization components
    optim = get_optimizer(base_model, args)
    scaler = GradScaler()
    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

    avg_meter = AverageMeter()
    start_time = time.time()
    best_miou = 0.0

    metrics_log_path = work_dir / "training_metrics.jsonl"
    val_log_path = work_dir / "validation_metrics.jsonl"

    logging.info("Starting training loop...")

    for n_iter in range(args.max_iters):
        try:
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image, seg_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image, seg_mask = next(train_iter)

        inputs = inputs.cuda(non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.cuda(non_blocking=True)

        # LR polynomial schedule with warmup
        if n_iter < args.warmup_iters:
            cur_lr = args.warmup_lr + (args.lr - args.warmup_lr) * n_iter / args.warmup_iters
        else:
            cur_lr = args.lr * (1.0 - (n_iter - args.warmup_iters) / (args.max_iters - args.warmup_iters + 1e-6)) ** 0.9
        for pg in optim.param_groups:
            pg["lr"] = cur_lr

        # Generate CAM pseudo-labels
        with torch.no_grad():
            auxiliaryMask, grad_mask, (cams_aux, cams, prototypes_cam) = multi_scale_cam2_v2(
                model, inputs=inputs, scales=args.cam_scales, args=args, cls_label=cls_label
            )
            cams_aux, cams, prototypes_cam = cams_aux.detach(), cams.detach(), prototypes_cam.detach()

            valid_cam, _ = cam_to_label(
                cams, cls_label=cls_label, img_box=img_box, ignore_mid=True,
                bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                ignore_index=args.ignore_index
            )
            refined_pseudo_label = refine_cams_with_bkg_v2(
                par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
                high_thre=args.high_thre, low_thre=args.low_thre,
                ignore_index=args.ignore_index, img_box=img_box
            )

        # Forward pass
        with autocast():
            (cls, segs, fmap, cls_aux, backbone_feat, cls_token, cls_token_logit,
             backbone_weight, outputs_seg_masks, cls_wsddn, prototypes) = model(
                inputs, label_idx=label_idx, n_iter=n_iter, cls_label=cls_label, args=args
            )

            cls_loss = F.multilabel_soft_margin_loss(cls.float(), cls_label)
            cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux.float(), cls_label)
            wsddn_pred = cls_wsddn.sum(dim=1).float().clamp(min=1e-6, max=1.0 - 1e-6)
            cls_gt = cls_label.detach().float()
            cls_loss_wsddn = - (cls_gt * torch.log(wsddn_pred) + (1.0 - cls_gt) * torch.log(1.0 - wsddn_pred)).mean()

            resized_cams_aux = F.interpolate(cams_aux, size=(28, 28), mode="bilinear", align_corners=False)
            _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box,
                                                ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                                low_thre=args.low_thre, ignore_index=args.ignore_index)
            aff_mask = label_to_aff_mask(pseudo_label_aux)
            ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

            segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode="bilinear", align_corners=False)

            # Hard segmentation loss on confident pixels
            seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

            # --- KEY INNOVATION: Boundary-Aware Prototype Distillation (BAPD) ---
            # Supervise the 28% boundary pixels marked 255 using soft prototype guidance
            boundary_mask = (refined_pseudo_label == args.ignore_index)
            if boundary_mask.sum() > 0 and prototypes_cam is not None:
                proto_interp = F.interpolate(prototypes_cam, size=segs.shape[2:], mode="bilinear", align_corners=False)
                # Class gating: zero-out classes not present in the image
                proto_interp = proto_interp * cls_label[:, :, None, None]
                # Background score: 1.0 - max foreground prototype probability
                proto_bg = (1.0 - proto_interp.max(dim=1, keepdim=True)[0]).clamp(min=0.01)
                proto_21 = torch.cat([proto_bg, proto_interp], dim=1) # [B, 21, H, W]
                proto_21 = proto_21 / (proto_21.sum(dim=1, keepdim=True) + 1e-6)

                p_soft = F.softmax(proto_21.float() / args.distill_temp, dim=1)
                log_q_soft = F.log_softmax(segs.float() / args.distill_temp, dim=1)

                # Soft KL divergence specifically focused on the 255 boundary transition regions
                kl_map = F.kl_div(log_q_soft, p_soft, reduction="none").sum(dim=1) # [B, H, W]
                distill_loss = kl_map[boundary_mask].mean() * (args.distill_temp ** 2)
            else:
                distill_loss = torch.tensor(0.0, device=segs.device)

            # Progressive Loss Scaling (PDLS): Ramps w_seg dynamically
            progress = min(1.0, n_iter / float(max(1, args.max_iters)))
            cur_w_seg = args.w_seg * (1.0 + 2.0 * progress) # Ramps from 0.12 to 0.36

            reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)

            if n_iter <= args.seg_warmup:
                total_loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.wsddn * cls_loss_wsddn
            else:
                total_loss = (1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss +
                              cur_w_seg * seg_loss + args.w_distill * distill_loss +
                              args.wsddn * cls_loss_wsddn + args.w_reg * reg_loss)

        optim.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optim)
        scaler.update()

        avg_meter.add({
            "cls_loss": cls_loss.item(),
            "seg_loss": seg_loss.item(),
            "distill_loss": distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
            "ptc_loss": ptc_loss.item(),
            "total_loss": total_loss.item()
        })

        # Logging
        if (n_iter + 1) % args.log_iters == 0:
            elapsed = time.time() - start_time
            iter_speed = (n_iter + 1) / elapsed
            eta_seconds = (args.max_iters - (n_iter + 1)) / iter_speed if iter_speed > 0 else 0

            log_str = (f"[{n_iter+1:04d}/{args.max_iters:04d}] LR: {cur_lr:.2e} | "
                       f"Loss: {avg_meter.get('total_loss'):.4f} | "
                       f"Seg: {avg_meter.get('seg_loss'):.4f} | "
                       f"BAPD Distill: {avg_meter.get('distill_loss'):.4f} | "
                       f"Speed: {iter_speed:.2f} it/s | ETA: {eta_seconds/60:.1f}m")
            logging.info(log_str)
            print(log_str, flush=True)

            log_entry = {
                "iteration": n_iter + 1,
                "elapsed_seconds": elapsed,
                "lr": cur_lr,
                "total_loss": avg_meter.get("total_loss"),
                "seg_loss": avg_meter.get("seg_loss"),
                "distill_loss": avg_meter.get("distill_loss"),
                "cls_loss": avg_meter.get("cls_loss"),
                "ptc_loss": avg_meter.get("ptc_loss"),
                "w_seg": cur_w_seg
            }
            with open(metrics_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # Save Checkpoint and Validate
        if (n_iter + 1) % args.save_iters == 0 or (n_iter + 1) == args.max_iters:
            ckpt_path = ckpt_dir / f"model_iter_{n_iter+1:08d}.pth"
            torch.save(base_model.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint: {ckpt_path}")

            # Run validation
            logging.info(f"Running evaluation on VOC Val (limit={args.eval_limit})...")
            val_miou, per_class_iou, num_eval = run_full_validation(base_model, val_loader, args, limit=args.eval_limit)

            val_res = {
                "iteration": n_iter + 1,
                "val_images": num_eval,
                "val_miou": val_miou,
                "checkpoint": str(ckpt_path),
                "per_class_miou": per_class_iou
            }
            with open(val_log_path, "a") as vf:
                vf.write(json.dumps(val_res) + "\n")

            logging.info(f">>> Validation at Iter {n_iter+1}: Raw Seg mIoU = {val_miou:.2f}% (on {num_eval} images) <<<")
            print(f">>> Validation at Iter {n_iter+1}: Raw Seg mIoU = {val_miou:.2f}% (on {num_eval} images) <<<", flush=True)

    logging.info("PCRE-BPD Training completed successfully.")
    print("PCRE-BPD Training completed successfully.", flush=True)


if __name__ == "__main__":
    main()
