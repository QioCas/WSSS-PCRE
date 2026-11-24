import argparse
import datetime
import logging
import os
import random
import sys
from matplotlib import pyplot as plt
import wandb

sys.path.append('.')
sys.path.append("")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import imageio.v2 as imageio
from PIL import Image

from datasets import voc
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.losses import (ClsTokenLoss, MyGo_Loss, MyGo_Loss2, MyGo_Loss3, get_masked_ptc_loss, get_seg_loss, get_energy_loss, DenseEnergyLoss)
from torch.nn.parallel import DistributedDataParallel
from model.PAR import PAR
from utils import imutils, evaluate
from utils.camutils import (cam_to_label, get_token_mask, label_to_aff_mask, multi_scale_cam2_v2, 
                            refine_cams_with_bkg_v2,)
from utils.vis import get_train_miou
from utils.wsddnutils import get_atten_mask
from utils.pyutils import AverageMeter, cal_eta, setup_logger
from engine import build_network_wsddn, build_optimizer, build_validation
from torch.cuda.amp import autocast, GradScaler
import scipy.ndimage as ndi

# Import Paradigm 2 Innovations
from utils.sam_innovations import (
    compute_sam_patch_affinity,
    compute_boundary_weight_map,
    get_boundary_focal_seg_loss
)

parser = argparse.ArgumentParser()
torch.hub.set_dir("./pretrained")

### loss weight
parser.add_argument("--w_ptc", default=0.3, type=float, help="weight for ptc loss")
parser.add_argument("--w_seg", default=0.12, type=float, help="weight for segmentation loss")
parser.add_argument("--w_reg", default=0.05, type=float, help="weight for regression loss")
parser.add_argument("--wsddn", default=2.0, type=float, help="weight for wsddn loss")

parser.add_argument("--bin_t", default=0.7, type=float, help="binary threshold")
parser.add_argument("--w_cpe", default=0.1, type=float, help="weight for cpe loss")
parser.add_argument("--w_seg2", default=0.06, type=float, help="weight for auxiliary segmentation loss")
parser.add_argument("--w_cls", default=1.0, type=float, help="weight for classification loss")

### lr and warmup
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="learning rate for warmup")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup iterations")
parser.add_argument("--backbone_lr", default=2e-5, type=float, help="learning rate for backbone")
parser.add_argument("--decoder_lr", default=1e-4, type=float, help="learning rate for decoder")
parser.add_argument("--min_lr", default=1e-6, type=float, help="minimum learning rate")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", action="store_true", help="save checkpoint")
parser.add_argument("--crop_size", default=448, type=int, help="crop size")
parser.add_argument("--spg", default=4, type=int, help="samples per gpu")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--work_dir", default="experiments/voc_sam_pcre_v2", type=str, help="work directory")

### data
parser.add_argument("--data_folder", default="./dataset/VOC2012", type=str, help="dataset folder")
parser.add_argument("--list_folder", default="datasets/voc", type=str, help="train/val list folder")

### model
parser.add_argument("--backbone", default="vit_base_patch16_224", type=str, help="vit backbone")
parser.add_argument("--scales", default=(0.5, 2.0), type=float, help="random rescale in data augmentation")
parser.add_argument("--cam_scales", default=(1.0, 0.75, 1.25), type=float, help="multi-scale cam")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--pretrained", default=True, type=bool, help="use pretrained backbone")

### train and eval iters
parser.add_argument("--max_iters", default=20000, type=int, help="max iterations")
parser.add_argument("--log_iters", default=100, type=int, help="log iterations")
parser.add_argument("--eval_iters", default=5000, type=int, help="eval iterations")
parser.add_argument("--checkpoint_step", default=5000, type=int, help="checkpoint save step")

### cam thresholds
parser.add_argument("--high_thre", default=0.7, type=float, help="high threshold for cam")
parser.add_argument("--low_thre", default=0.25, type=float, help="low threshold for cam")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="background threshold for cam")

parser.add_argument("--pro_high_thre", default=0.65, type=float, help="high threshold for prototype cam")
parser.add_argument("--pro_low_thre", default=0.25, type=float, help="low threshold for prototype cam")

### stage iters
parser.add_argument("--seg_warmup", default=1500, type=int, help="warmup iterations for segmentation")
parser.add_argument("--seg_iter", default=4000, type=int, help="iterations for segmentation")

parser.add_argument("--color_aug", default=True, type=bool, help="use color augmentation")
parser.add_argument("--mask_circle_r", default=1.0, type=float, help="radius of mask circle")
parser.add_argument("--wandb_name", default="train_voc_sam_pcre_v2", type=str, help="wandb run name")
parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
parser.add_argument("--resume", default=None, type=str, help="resume from checkpoint")
parser.add_argument("--save_path", default=None, type=str, help="path to save predictions")
parser.add_argument("--pro_reg", action="store_true", help="regularization for prototype")
parser.add_argument("--reg_aux", action="store_true", help="auxiliary regularization")
parser.add_argument("--seg_aux", action="store_true", help="auxiliary segmentation loss")
parser.add_argument("--pro_scale", default=2, type=float, help="scale for prototype")
parser.add_argument("--clip", default=1.0, type=float, help="gradient clipping")
parser.add_argument("--wsddn_topk", default=0.2, type=float, help="topk for wsddn")

### SAM offline cache
parser.add_argument("--sam_cache_dir", default='experiments/representation_sam_20261120/cache/sam/labels', type=str, help="SAM offline cache labels folder")

### ablation args
parser.add_argument("--no-ema", action="store_false", dest="ema", help="Disable auxiliary augmentation")
parser.add_argument("--no-pro", action="store_false", dest="pro", help="Disable auxiliary augmentation")
parser.add_argument("--no-pre-mask", action="store_false", dest="pre_mask", help="Disable auxiliary augmentation")
parser.add_argument("--no-mask_exp", action="store_false", dest="mask_exp", help="Disable auxiliary augmentation")
parser.add_argument("--no-ctmd", action="store_false", dest="ctmd", help="Disable auxiliary augmentation")
parser.add_argument("--mygo_sup", action="store_true", dest="mygo_sup", help="Disable auxiliary augmentation")
parser.add_argument("--no-mask-stop", action="store_false", dest="mask_stop", help="Disable auxiliary augmentation")
parser.add_argument("--pro_fuse", action="store_true", dest="pro_fuse", help="Disable auxiliary augmentation")
parser.add_argument("--wsddn_loss", type=str, default="sum", help="wsddn_loss")
parser.add_argument("--no-mil", action="store_false", dest="mil", help="Disable auxiliary augmentation")
parser.add_argument("--clstoken_loss", default='cos', type=str)

### Paradigm 2 Innovations
parser.add_argument("--enable_sam_shrinkage", default=True, type=lambda x: (str(x).lower() == 'true'), help="Enable SAM-guided boundary uncertainty shrinkage on 255 pixels")
parser.add_argument("--enable_sam_crme_barrier", default=True, type=lambda x: (str(x).lower() == 'true'), help="Enable SAM boundary barrier to prevent CRME background bleeding")
parser.add_argument("--enable_sat_cr", default=True, type=lambda x: (str(x).lower() == 'true'), help="Enable Method 1: SAM-Affinity Token Contrastive Regularization (SAT-CR)")
parser.add_argument("--sat_cr_purity", default=0.85, type=float, help="Purity threshold for SAT-CR patch homogeneity")
parser.add_argument("--enable_sbf_distill", default=True, type=lambda x: (str(x).lower() == 'true'), help="Enable Method 2: SAM-Guided Boundary-Focal Density Distillation (SBF-Distill)")
parser.add_argument("--boundary_weight_alpha", default=2.0, type=float, help="Weight scaling alpha for boundary focal loss in SBF-Distill")


# =========================================================================
# Custom Dataset with Synchronized SAM Offline Mask Augmentation
# =========================================================================
class VOC12ClsSamDataset(voc.VOC12ClsDataset):
    """VOC Dataset loading pre-cached offline SAM masks and applying identical transforms."""
    def __init__(self, *args, sam_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_dir = sam_dir

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_path = os.path.join(self.img_dir, _img_name + '.jpg')
        image = np.asarray(imageio.imread(img_path))
        
        # Load offline SAM mask if available
        sam_mask = None
        if self.sam_dir and os.path.exists(self.sam_dir):
            sam_path = os.path.join(self.sam_dir, _img_name + '.png')
            if os.path.exists(sam_path):
                try:
                    sam_mask = np.asarray(imageio.imread(sam_path))
                except Exception:
                    sam_mask = None
                    
        if sam_mask is None:
            sam_mask = np.full(image.shape[:2], 255, dtype=np.uint8)

        # Apply identical transformations (scaling, flipping, random cropping)
        pil_image = Image.fromarray(image)
        image, local_image, img_box, sam_mask_crop = self._VOC12ClsDataset__transforms(image=image, seg_mask=sam_mask)

        cls_label = self.label_list[_img_name]
        label_idx = np.zeros(1,)
        if cls_label.sum(-1) != 0:
            label_idx = np.where(cls_label.sum(-1) > 1, -1, np.argmax(cls_label, axis=-1) + 1)
        else:
            label_idx = np.array([-1])

        raw_image = image
        w_image = self.global_view2(pil_image)
        s_image = local_image

        return _img_name, image, cls_label, label_idx, img_box, raw_image, w_image, s_image, sam_mask_crop


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args=None):
    if args.local_rank == 0:
        try:
            wandb.init(
                project="wsss",
                name=args.wandb_name,
                config=args,
                mode="offline" if not args.use_wandb else "online"
            )
        except Exception:
            pass

    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    time_stamp_month = datetime.datetime.now().strftime("%Y-%m")
    args.work_dir = os.path.join(args.work_dir, time_stamp_month, f"train_voc_sam_pcre_v2_{timestamp[3:]}")
    os.makedirs(args.work_dir, exist_ok=True)
    setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
    logging.info(f"Initialized SAM-PCRE v2 with Innovations: SAT-CR={args.enable_sat_cr}, SBF-Distill={args.enable_sbf_distill}")
    logging.info(f"Arguments: {args}")

    setup_seed(args.seed)

    model, get_seg_crme_loss, get_cls_token_loss = build_network_wsddn(args)
    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)

    optimizer = build_optimizer(model, args)
    scaler = GradScaler()
    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24])
    par.to(device=args.local_rank)

    dali_dataset = VOC12ClsSamDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split='train_aug',
        stage='train',
        aug=True,
        resize_range=args.scales,
        rescale_range=[448, 448],
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
        color_aug=args.color_aug,
        sam_dir=args.sam_cache_dir
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split='val',
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    sampler = DistributedSampler(dali_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        dali_dataset,
        batch_size=args.spg,
        num_workers=4,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        sampler=val_sampler
    )

    device = torch.device(args.local_rank)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler = sampler
    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()
    n_iter = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        n_iter = checkpoint['n_iter']
        logging.info(f"Resumed from iteration {n_iter}")

    while n_iter <= args.max_iters:
        try:
            batch = next(train_loader_iter)
        except StopIteration:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        _img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image, sam_mask_crop = batch

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)
        label_idx = label_idx.to(device, non_blocking=True)
        sam_mask_crop = sam_mask_crop.to(device, non_blocking=True)

        cur_lr = optimizer.param_groups[0]['lr']

        with torch.no_grad():
            auxiliaryMask, grad_mask, (cams_aux, cams, prototypes_cam) = multi_scale_cam2_v2(
                model, inputs=inputs, scales=args.cam_scales, cls_label=cls_label, args=args
            )

            valid_cam, asd = cam_to_label(
                cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                ignore_index=args.ignore_index
            )

            valid_cam_aux, asd = cam_to_label(
                cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                ignore_index=args.ignore_index
            )

        # Baseline PAR refinement from PCRE
        refined_pseudo_label = refine_cams_with_bkg_v2(
            par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
            high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box
        )
        refined_pseudo_label_aux = refine_cams_with_bkg_v2(
            par, inputs_denorm, cams=valid_cam_aux, cls_labels=cls_label,
            high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box
        )

        # =========================================================================
        # INNOVATION 1: SAM-Guided Boundary Uncertainty Shrinkage
        # =========================================================================
        if args.enable_sam_shrinkage:
            uncertain = (refined_pseudo_label == args.ignore_index)
            if uncertain.any():
                for b in range(inputs.shape[0]):
                    b_unc = uncertain[b]
                    b_sam = sam_mask_crop[b]
                    refined_pseudo_label[b, b_unc & (b_sam == 0)] = 0
                    active_classes = torch.nonzero(cls_label[b]).squeeze(-1) + 1
                    for c in active_classes:
                        c_val = c.item()
                        refined_pseudo_label[b, b_unc & (b_sam == c_val)] = c_val

        # Token affinity from ToCo
        resized_cams_aux = F.interpolate(cams_aux, size=(28, 28), mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(
            resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
            bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
            ignore_index=args.ignore_index
        )

        # =========================================================================
        # INNOVATION 3 (NEW METHOD 1): SAT-CR (SAM-Affinity Token Contrastive Regularization)
        # =========================================================================
        if args.enable_sat_cr:
            aff_mask, _ = compute_sam_patch_affinity(
                sam_mask=sam_mask_crop,
                pseudo_label_aux=pseudo_label_aux,
                purity_thresh=args.sat_cr_purity,
                ignore_index=args.ignore_index
            )
        else:
            aff_mask = label_to_aff_mask(pseudo_label_aux, ignore_index=args.ignore_index)

        # Bảo vệ cls_token_mask: Luôn dùng pseudo_label_aux gốc từ CAM để đảm bảo cpe_loss học đúng
        cls_token_mask = get_token_mask(pseudo_label_aux, args.num_classes, cls_label.detach())
        atten_mask = None

        (
            cls, segs, fmap, cls_aux,
            backbone_feat, cls_token, cls_token_logit, backbone_weight, outputs_seg_masks, cls_wsddn, prototypes
        ) = model(
            inputs,
            label_idx=label_idx,
            n_iter=n_iter,
            atten_mask=atten_mask,
            cls_label=cls_label,
            aug_img=None,
            args=args
        )

        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        if not args.mil:
            cls_loss_wsddn = F.multilabel_soft_margin_loss(cls_wsddn.sum(dim=1), cls_label)
        elif args.wsddn_loss == "sum":
            cls_loss_wsddn = F.binary_cross_entropy(cls_wsddn.sum(dim=1), cls_label.detach())
        elif args.wsddn_loss == "max":
            cls_loss_wsddn = F.binary_cross_entropy(cls_wsddn.max(dim=1)[0], cls_label.detach())

        if args.clstoken_loss == "cos":
            cpe_loss = get_masked_ptc_loss(fmap.detach(), cls_token_mask.flatten(-2), cls_token)
        elif args.clstoken_loss == "linear":
            cpe_loss = get_cls_token_loss(cls_token, cls_label)
        elif args.clstoken_loss == "sum":
            cpe_loss = get_cls_token_loss(cls_token, cls_label, sum=True)

        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # =========================================================================
        # INNOVATION 2: SAM-Constrained CRME Boundary Barrier
        # =========================================================================
        if args.pro:
            valid_pro, asd = cam_to_label(
                prototypes_cam, cls_label=cls_label, img_box=img_box, ignore_mid=True,
                bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                ignore_index=args.ignore_index
            )
            refined_pro_label = refine_cams_with_bkg_v2(
                par, inputs_denorm, cams=valid_pro, cls_labels=cls_label,
                high_thre=args.pro_high_thre, low_thre=args.pro_low_thre,
                ignore_index=args.ignore_index, img_box=img_box
            )

            if args.enable_sam_crme_barrier:
                for b in range(inputs.shape[0]):
                    b_sam = sam_mask_crop[b]
                    leak = (refined_pro_label[b] > 0) & (refined_pro_label[b] != args.ignore_index) & (b_sam == 0)
                    refined_pro_label[b, leak] = 0
                    
                    pro_unc = (refined_pro_label[b] == args.ignore_index)
                    if pro_unc.any():
                        refined_pro_label[b, pro_unc & (b_sam == 0)] = 0
                        active_classes = torch.nonzero(cls_label[b]).squeeze(-1) + 1
                        for c in active_classes:
                            c_val = c.item()
                            refined_pro_label[b, pro_unc & (b_sam == c_val)] = c_val

            crme_loss = get_seg_crme_loss(
                outputs_seg_masks, cls_wsddn.detach(), cls_label, refined_pro_label.detach(),
                topk=args.wsddn_topk,
                mask_circle=((n_iter <= args.seg_iter and args.pre_mask) or args.mask_stop),
                mask_exp=args.mask_exp,
                fin_size=refined_pseudo_label.shape[-2:],
                mask_circle_r=1 + (1 - np.cos((n_iter - args.seg_warmup) / max(1, args.seg_iter - args.seg_warmup) * np.pi)) / 2 * 4
            )
            get_seg_crme_loss.update_expand_time(
                1 + (1 - np.cos((n_iter - args.seg_warmup) / max(1, args.seg_iter - args.seg_warmup) * np.pi)) / 2 * 9
            )
        else:
            crme_loss = get_seg_crme_loss(
                outputs_seg_masks, cls_wsddn.detach(), cls_label.detach(), refined_pseudo_label.detach(),
                topk=args.wsddn_topk, mask_circle=(n_iter <= args.seg_iter), fin_size=refined_pseudo_label.shape[-2:]
            )

        # =========================================================================
        # INNOVATION 4 (NEW METHOD 2): SBF-Distill (SAM-Guided Boundary-Focal Density Distillation)
        # =========================================================================
        target_seg_label = refined_pro_label.detach().type(torch.long) if args.mygo_sup else refined_pseudo_label.type(torch.long)

        if args.enable_sbf_distill:
            boundary_weights = compute_boundary_weight_map(
                sam_mask=sam_mask_crop,
                alpha=args.boundary_weight_alpha,
                kernel_size=9,
                ignore_index=args.ignore_index
            )
            seg_loss = get_boundary_focal_seg_loss(
                pred=segs,
                label=target_seg_label,
                weight_map=boundary_weights,
                ignore_index=args.ignore_index
            )
        else:
            seg_loss = get_seg_loss(segs, target_seg_label, ignore_index=args.ignore_index)

        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)

        if args.seg_aux:
            if args.enable_sbf_distill:
                seg_loss_aux = get_boundary_focal_seg_loss(
                    pred=segs,
                    label=refined_pseudo_label_aux.type(torch.long),
                    weight_map=boundary_weights,
                    ignore_index=args.ignore_index
                )
            else:
                seg_loss_aux = get_seg_loss(segs, refined_pseudo_label_aux.type(torch.long), ignore_index=args.ignore_index)
            seg_loss = (seg_loss + seg_loss_aux) / 2

        if args.reg_aux:
            reg_aux_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label_aux, img_box=img_box, loss_layer=loss_layer)
            reg_loss = (reg_loss + reg_aux_loss) / 2

        if n_iter <= args.seg_warmup:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.wsddn * cls_loss_wsddn + args.w_cpe * cpe_loss
        else:
            loss = (
                1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_cpe * cpe_loss + args.w_ptc * ptc_loss +
                args.w_seg * seg_loss + args.w_seg2 * crme_loss + args.wsddn * cls_loss_wsddn + args.w_reg * reg_loss
            )

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cpe_loss': cpe_loss.item(),
            'crme_loss': crme_loss.item() if isinstance(crme_loss, torch.Tensor) else crme_loss,
            'cls_score': float(cls_score),
            'cls_loss_wsddn': cls_loss_wsddn.item(),
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
        })

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()

        if n_iter % args.log_iters == 0 and args.local_rank == 0:
            delta, eta = cal_eta(start_time, n_iter, args.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            if cur_lr > 0:
                log_data = {
                    "Iter": n_iter,
                    "Elapsed": delta,
                    "ETA": eta,
                    "LR": f"{cur_lr:.3e}",
                    "cls_loss": avg_meter.pop('cls_loss'),
                    "cls_loss_aux": avg_meter.pop('cls_loss_aux'),
                    "ptc_loss": avg_meter.pop('ptc_loss'),
                    "seg_loss": avg_meter.pop('seg_loss'),
                    "cpe_loss": avg_meter.pop('cpe_loss'),
                    "crme_loss": avg_meter.pop('crme_loss'),
                    "cls_loss_wsddn": avg_meter.pop('cls_loss_wsddn'),
                }
                logging.info(
                    f"Iter: {log_data['Iter']}; Elapsed: {log_data['Elapsed']}; ETA: {log_data['ETA']}; LR: {log_data['LR']}; "
                    f"cls_loss: {log_data['cls_loss']:.4f}, "
                    f"cls_loss_aux: {log_data['cls_loss_aux']:.4f}, "
                    f"ptc_loss: {log_data['ptc_loss']:.4f}, "
                    f"seg_loss: {log_data['seg_loss']:.4f}, "
                    f"cpe_loss: {log_data['cpe_loss']:.4f}, "
                    f"crme_loss: {log_data['crme_loss']:.4f}, "
                    f"cls_loss_wsddn: {log_data['cls_loss_wsddn']:.4f}"
                )

        if n_iter % args.checkpoint_step == 0 and args.local_rank == 0 and args.save_ckpt and n_iter > 0:
            ckpt_dir = os.path.join(args.work_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "n_iter": n_iter
                },
                os.path.join(ckpt_dir, f"model_iter_{n_iter}.pth")
            )
            logging.info(f"Saved checkpoint to {os.path.join(ckpt_dir, f'model_iter_{n_iter}.pth')}")

        if (n_iter % args.eval_iters == 0 and n_iter > 0) or (n_iter == args.max_iters):
            if args.local_rank == 0:
                logging.info("Validating on Pascal VOC Val...")
            val_engine = build_validation(args)
            val_engine(
                model=model,
                val_loader=val_loader,
                n_iter=n_iter,
                args=args,
                save_path=os.path.join(args.work_dir, "predictions")
            )

        n_iter += 1


if __name__ == "__main__":
    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    start_time = datetime.datetime.now()
    train(args)
