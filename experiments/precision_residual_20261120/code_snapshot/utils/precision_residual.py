"""Frozen PCRE with bounded residual correction and train-only rejection targets."""
from pathlib import Path
import random
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import torch
from torch import nn
import torch.nn.functional as F
from utils.decoder_selftraining import normalized_image,load_baseline


class ResidualHead(nn.Module):
    def __init__(self,channels=768,width=64,detail=False,limit=2.):
        super().__init__();self.detail=detail;self.limit=limit
        self.layers=nn.Sequential(nn.Conv2d(channels*(2 if detail else 1),width,1),nn.GELU(),
                                  nn.Conv2d(width,width,3,padding=1),nn.GELU(),nn.Conv2d(width,21,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,features,early=None):
        x=F.normalize(features,dim=1)
        if self.detail:
            if early is None:raise ValueError('Detail head requires early features')
            high=early-F.avg_pool2d(early,3,stride=1,padding=1)
            x=torch.cat([x,F.normalize(high,dim=1)],1)
        return self.limit*torch.tanh(self.layers(x)/self.limit)


def correction_gate(base_logits,confidence=.95):
    probability=base_logits.detach().softmax(1);certainty,winner=probability.max(1,keepdim=True)
    # Same tag-free gate in all arms: allow predicted foreground and uncertain
    # background. Confident background logits are exactly preserved before CRF.
    return ((winner>0)|(certainty<confidence)).to(base_logits.dtype)


class PrecisionResidual(nn.Module):
    def __init__(self,base,plan,detail=False):
        super().__init__();self.base=base.eval().requires_grad_(False)
        channels=base.decoder_embd_dim
        self.residual=ResidualHead(channels,plan['residual_width'],detail,plan['residual_limit'])
        self.confidence=plan['gate_confidence'];self.detail=detail

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    @torch.no_grad()
    def frozen_forward(self,inputs):
        captured=[];handle=None
        if self.detail:
            handle=self.base.encoder.blocks[3].register_forward_hook(lambda module,args,out:captured.append(out[0].detach()))
        try:patches=self.base.encoder.forward_features(inputs)[1]
        finally:
            if handle is not None:handle.remove()
        h,w=inputs.shape[-2]//self.base.encoder.patch_size,inputs.shape[-1]//self.base.encoder.patch_size
        features=self.base.to_2D(patches,h,w);early=None
        if captured:early=self.base.to_2D(captured[0][:,-h*w:],h,w)
        return features,self.base.decoder(features),early

    def components(self,inputs):
        features,base,early=self.frozen_forward(inputs)
        gate=correction_gate(base,self.confidence)
        return base+gate*self.residual(features,early),base,gate

    def forward(self,inputs,args=None):return None,self.components(inputs)[0]


def load_residual_checkpoint(cfg,checkpoint):
    from tools.infer_voc_checkpoint_pair import sha256
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['format']!='precision_residual_v1':raise ValueError('Unexpected adapter format')
    plan=payload['plan'];teacher=Path(plan['teacher_run'])/'checkpoints'/plan['teacher_checkpoint']
    if sha256(teacher)!=payload['teacher_sha256']:raise ValueError('Baseline checksum changed')
    base,_=load_baseline(plan,device='cpu')
    model=PrecisionResidual(base,plan,payload['arm']['detail'])
    model.residual.load_state_dict(payload['ema'],strict=True)
    model.provenance={'architecture':'frozen_pcre_bounded_residual','baseline_sha256':payload['teacher_sha256'],
                      'external_image_text_prior':payload['arm']['external_prior'],
                      'adapter_steps':payload['step'],'detail':payload['arm']['detail']}
    return model


def strict_keep_kl(logits,base,labels,valid):
    probability=base.detach().softmax(1);winner=probability.argmax(1)
    mask=valid&(labels!=255)&(winner==labels)
    divergence=F.kl_div(logits.log_softmax(1),probability,reduction='none').sum(1)
    return (divergence[mask].mean() if mask.any() else logits.sum()*0),mask


def complementary_loss(logits,rejected,valid):
    mask=valid&(rejected>0)&(rejected<21)
    index=rejected.clamp(0,20).unsqueeze(1)
    # logsumexp form stays differentiable even when softmax rounds p(c) to 1.
    # Clamping 1-p(c) would zero the gradient on the hardest false positives.
    alternatives=logits.scatter(1,index,float('-inf'))
    loss=torch.logsumexp(logits,dim=1)-torch.logsumexp(alternatives,dim=1)
    # No positive class is inferred from a veto.
    return (loss[mask].mean() if mask.any() else logits.sum()*0),mask


def veto_labels(labels,rejected):
    output=labels.copy();valid=(labels>0)&(labels<21)&(rejected==labels)
    output[valid]=255;negative=np.full_like(labels,255);negative[valid]=labels[valid]
    return output,negative


@torch.no_grad()
def negative_evidence(features,bank,margin):
    vector=F.normalize(features,dim=1).permute(0,2,3,1)
    positive=torch.einsum('bhwd,ckd->bhwck',vector,bank['positive']).amax(-1)
    negative=torch.einsum('bhwd,ckd->bhwck',vector,bank['negative']).amax(-1)
    valid=bank['positive_valid']&bank['negative_valid'];valid[0]=False
    return ((negative-positive)>margin)&valid[None,None,None,:]


@torch.no_grad()
def projected_features(features,projection):
    vector=F.normalize(features,dim=1).permute(0,2,3,1)
    return F.normalize(vector@projection,dim=-1)


@torch.no_grad()
def density_evidence(features,bank):
    x=projected_features(features,bank['projection'])
    difference=x[...,None,None,:]-bank['gmm_means'][None,None,None,:,:,:]
    variance=bank['gmm_variances'];weights=bank['gmm_weights']
    component=-.5*((difference.square()/variance).sum(-1)+variance.log().sum(-1)+x.shape[-1]*np.log(2*np.pi))
    likelihood=torch.logsumexp(component+weights.clamp_min(1e-12).log(),dim=-1)
    valid=bank['density_valid'].clone();valid[0]=False
    return (likelihood<bank['density_cutoff'])&valid[None,None,None,:]


class ResidualDataset:
    """RGB + cached pseudo-labels only; no VOC pixel annotation access."""
    def __init__(self,plan,cache,names,rng):
        self.plan=plan;self.cache=Path(cache);self.names=names;self.rng=rng

    def __getitem__(self,index):
        p=self.plan;rng=self.rng;name=self.names[index]
        image=Image.open(Path(p['data_folder'])/'JPEGImages'/f'{name}.jpg').convert('RGB')
        label=Image.open(self.cache/'labels'/f'{name}.png')
        path=self.cache/'negative'/f'{name}.png'
        negative=Image.open(path) if path.exists() else Image.new('L',image.size,255)
        short=rng.randint(*p['resize_short_range']);ratio=short/min(image.size)
        size=tuple(max(1,round(v*ratio)) for v in image.size)
        image=image.resize(size,Image.Resampling.BILINEAR)
        label=label.resize(size,Image.Resampling.NEAREST);negative=negative.resize(size,Image.Resampling.NEAREST)
        if rng.random()<.5:image=ImageOps.mirror(image);label=ImageOps.mirror(label);negative=ImageOps.mirror(negative)
        support=Image.new('L',image.size,1);crop=p['crop_size']
        pad=(0,0,max(0,crop-image.width),max(0,crop-image.height))
        image=ImageOps.expand(image,pad,fill=(124,116,104));label=ImageOps.expand(label,pad,fill=255)
        negative=ImageOps.expand(negative,pad,fill=255);support=ImageOps.expand(support,pad,fill=0)
        left=rng.randint(0,image.width-crop);top=rng.randint(0,image.height-crop);box=(left,top,left+crop,top+crop)
        image=image.crop(box);label=label.crop(box);negative=negative.crop(box);support=support.crop(box)
        for op in [ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]:
            image=op(image).enhance(rng.uniform(1-p['jitter'],1+p['jitter']))
        return normalized_image(image),torch.from_numpy(np.array(label,dtype=np.int64)),torch.from_numpy(np.array(negative,dtype=np.int64)),torch.from_numpy(np.array(support,dtype=bool))
