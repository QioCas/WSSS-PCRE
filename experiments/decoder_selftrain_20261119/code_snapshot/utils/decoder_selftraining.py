"""Frozen-feature decoder refinement: no pixel ground truth in training inputs."""
import copy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F


def load_baseline(plan, device='cuda'):
    from argparse import Namespace
    from model.model_wsddn import network
    run=Path(plan['teacher_run'])
    cfg=json.loads((run/'config.json').read_text())
    cfg.update(data_folder=plan['data_folder'],list_folder=plan['list_folder'],debug=False)
    args=Namespace(**cfg)
    model=network(args,backbone=args.backbone,num_classes=args.num_classes,
                  pretrained=False,aux_layer=args.aux_layer)
    model.encoder.after_load_state_dict()
    state=torch.load(run/'checkpoints'/plan['teacher_checkpoint'],map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    model.eval().to(device)
    return model,args


def normalized_image(image):
    image=np.asarray(image,dtype=np.float32)
    mean=np.array([123.675,116.28,103.53],dtype=np.float32)
    std=np.array([58.395,57.12,57.375],dtype=np.float32)
    return torch.from_numpy(((image-mean)/std).transpose(2,0,1).copy())


@torch.no_grad()
def frozen_features(model, inputs):
    patches=model.encoder.forward_features(inputs)[1]
    return model.to_2D(patches,inputs.shape[-2]//model.encoder.patch_size,
                       inputs.shape[-1]//model.encoder.patch_size)


@torch.no_grad()
def fused_teacher_logits(model, inputs, size, scales):
    scaled=[]
    for scale in scales:
        resized=F.interpolate(inputs,size=(int(448*scale),int(448*scale)),
                              mode='bilinear',align_corners=False)
        augmented=torch.cat([resized,resized.flip(-1)])
        logits=model.decoder(frozen_features(model,augmented))
        logits=F.interpolate(logits,size=size,mode='bilinear',align_corners=False)
        scaled.append(logits[:1]+logits[1:].flip(-1))
    stack=torch.stack(scaled)
    return torch.maximum(stack[0],stack.mean(0))


def confident_labels(probabilities, valid_ids, confidence=.7, margin=.2):
    """Map subset-class posterior to full VOC IDs, rejecting uncertain pixels."""
    if probabilities.shape[0]!=len(valid_ids):
        raise ValueError('Class IDs must match probability channels')
    order=np.argsort(probabilities,axis=0)
    winner=order[-1]
    top=np.take_along_axis(probabilities,winner[None],axis=0)[0]
    runner=np.take_along_axis(probabilities,order[-2:-1],axis=0)[0] if len(valid_ids)>1 else np.zeros_like(top)
    label=np.asarray(valid_ids,dtype=np.uint8)[winner]
    label[(top<confidence)|((top-runner)<margin)]=255
    return label


def transform_pair(image,label,rng,crop_size=448,short_range=(336,672),jitter=.2):
    """One geometry for RGB and cached pseudo-labels, with RGB-only color jitter."""
    short=rng.randint(*short_range)
    ratio=short/min(image.size)
    size=tuple(max(1,round(x*ratio)) for x in image.size)
    image=image.resize(size,Image.Resampling.BILINEAR)
    label=label.resize(size,Image.Resampling.NEAREST)
    if rng.random()<.5:
        image=ImageOps.mirror(image);label=ImageOps.mirror(label)
    pad_w=max(0,crop_size-image.width);pad_h=max(0,crop_size-image.height)
    image=ImageOps.expand(image,(0,0,pad_w,pad_h),fill=(124,116,104))
    label=ImageOps.expand(label,(0,0,pad_w,pad_h),fill=255)
    left=rng.randint(0,image.width-crop_size);top=rng.randint(0,image.height-crop_size)
    box=(left,top,left+crop_size,top+crop_size)
    image=image.crop(box);label=label.crop(box)
    if jitter:
        operations=[ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]
        rng.shuffle(operations)
        for operation in operations:
            image=operation(image).enhance(rng.uniform(1-jitter,1+jitter))
    return normalized_image(image),torch.from_numpy(np.array(label,dtype=np.int64))


class PseudoLabelDataset:
    """Reads JPEGImages and cache only; never opens segmentation annotations."""
    def __init__(self,plan,cache,mode,rng,names=None):
        if mode not in ['raw_teacher','crf_teacher']:
            raise ValueError(mode)
        self.plan=plan;self.cache=Path(cache);self.mode=mode;self.rng=rng
        self.names=names or (Path(plan['list_folder'])/(plan['train_split']+'.txt')).read_text().split()

    def __getitem__(self,index):
        name=self.names[index]
        image=Image.open(Path(self.plan['data_folder'])/'JPEGImages'/f'{name}.jpg').convert('RGB')
        label=Image.open(self.cache/self.mode/f'{name}.png')
        return transform_pair(image,label,self.rng,self.plan['crop_size'],
                              self.plan['resize_short_range'],self.plan['color_jitter'])


def balanced_cross_entropy(logits,labels):
    valid=labels!=255
    losses=F.cross_entropy(logits,labels,ignore_index=255,reduction='none')
    terms=[]
    for mask in [valid&(labels==0),valid&(labels>0)]:
        if mask.any():
            terms.append(losses[mask].mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


def anchor_kl(student,teacher,temperature=2.):
    return F.kl_div(F.log_softmax(student/temperature,dim=1),
                    F.softmax(teacher.detach()/temperature,dim=1),
                    reduction='none').sum(1).mean()*(temperature**2)


@torch.no_grad()
def update_ema(ema,current,decay):
    for target,source in zip(ema.parameters(),current.parameters()):
        target.mul_(decay).add_(source,alpha=1-decay)
    for target,source in zip(ema.buffers(),current.buffers()):
        target.copy_(source)


def initialize_heads(model):
    model.eval().requires_grad_(False)
    teacher=copy.deepcopy(model.decoder).eval().requires_grad_(False)
    model.decoder.requires_grad_(True)
    ema=copy.deepcopy(model.decoder).eval().requires_grad_(False)
    return teacher,ema


def refined_state(model,head):
    state={'module.'+key:value.detach().cpu() for key,value in model.state_dict().items()}
    for key,value in head.state_dict().items():
        state['module.decoder.'+key]=value.detach().cpu()
    return state
