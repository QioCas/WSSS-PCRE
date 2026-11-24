"""Reliable multi-view targets and weak/strong consistency for partial ViT tuning."""
import json
from pathlib import Path
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
from utils.decoder_selftraining import normalized_image


def student_features(model,inputs):
    # No no_grad here: frozen prefix is graph-free automatically, selected final
    # blocks must retain their computation graph.
    patches=model.encoder.forward_features(inputs)[1]
    return model.to_2D(patches,inputs.shape[-2]//model.encoder.patch_size,
                       inputs.shape[-1]//model.encoder.patch_size)


def select_trainable(model,last_blocks):
    count=len(model.encoder.blocks)
    if last_blocks<0 or last_blocks>count:raise ValueError('Invalid final-block count')
    model.eval().requires_grad_(False);model.decoder.requires_grad_(True)
    if last_blocks:
        for block in model.encoder.blocks[-last_blocks:]:block.requires_grad_(True)
        model.encoder.norm.requires_grad_(True)
    return [name for name,p in model.named_parameters() if p.requires_grad]


def reliable_labels(mean_probability,votes,valid_ids,total_views,confidence,min_votes):
    winner=mean_probability.argmax(0)
    top=np.take_along_axis(mean_probability,winner[None],0)[0]
    count=np.take_along_axis(votes,winner[None],0)[0]
    result=np.asarray(valid_ids,dtype=np.uint8)[winner]
    result[(top<confidence)|(count<min_votes)]=255
    return result,top,count/total_views


@torch.no_grad()
def view_probabilities(model,inputs,size,valid_ids,scales):
    from utils.decoder_selftraining import frozen_features
    mean=None;votes=None;views=0
    for scale in scales:
        resized=F.interpolate(inputs,size=(int(448*scale),int(448*scale)),mode='bilinear',align_corners=False)
        augmented=torch.cat([resized,resized.flip(-1)])
        logits=model.decoder(frozen_features(model,augmented))
        logits=F.interpolate(logits,size=size,mode='bilinear',align_corners=False)
        for index in range(2):
            current=logits[index,valid_ids]
            if index:current=current.flip(-1)
            probability=F.softmax(current,dim=0)
            if mean is None:
                mean=torch.zeros_like(probability)
                votes=torch.zeros_like(probability,dtype=torch.int32)
            mean.add_(probability)
            winner=probability.argmax(0,keepdim=True)
            votes.scatter_add_(0,winner,torch.ones_like(winner,dtype=torch.int32))
            views+=1
    return (mean/views).cpu().numpy(),votes.cpu().numpy(),views


def weak_strong_pair(image,label,rng,plan):
    short=rng.randint(*plan['resize_short_range']);ratio=short/min(image.size)
    size=tuple(max(1,round(v*ratio)) for v in image.size)
    image=image.resize(size,Image.Resampling.BILINEAR);label=label.resize(size,Image.Resampling.NEAREST)
    support=Image.new('L',image.size,1)
    if rng.random()<.5:image=ImageOps.mirror(image);label=ImageOps.mirror(label);support=ImageOps.mirror(support)
    crop=plan['crop_size'];pad=(0,0,max(0,crop-image.width),max(0,crop-image.height))
    image=ImageOps.expand(image,pad,fill=(124,116,104));label=ImageOps.expand(label,pad,fill=255);support=ImageOps.expand(support,pad,fill=0)
    left=rng.randint(0,image.width-crop);top=rng.randint(0,image.height-crop);box=(left,top,left+crop,top+crop)
    weak=image.crop(box);target=np.array(label.crop(box),dtype=np.int64);valid=np.array(support.crop(box),dtype=bool)
    strong=weak.copy()
    operations=[ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color];rng.shuffle(operations)
    for operation in operations:strong=operation(strong).enhance(rng.uniform(1-plan['strong_jitter'],1+plan['strong_jitter']))
    if rng.random()<plan['grayscale_probability']:strong=ImageOps.grayscale(strong).convert('RGB')
    if rng.random()<plan['blur_probability']:strong=strong.filter(ImageFilter.GaussianBlur(rng.uniform(.1,1.5)))
    if rng.random()<plan['cutout_probability']:
        side=max(1,int(crop*plan['cutout_area']**.5));x=rng.randint(0,crop-side);y=rng.randint(0,crop-side)
        strong.paste((124,116,104),(x,y,x+side,y+side));target[y:y+side,x:x+side]=255;valid[y:y+side,x:x+side]=False
    return normalized_image(weak),normalized_image(strong),torch.from_numpy(target),torch.from_numpy(valid)


class ConsistencyDataset:
    """Training access is RGB + cached pseudo-labels + image-level tags only."""
    def __init__(self,plan,cache,names,rng):
        self.plan=plan;self.cache=Path(cache);self.names=names;self.rng=rng
        self.tags=np.load(Path(plan['list_folder'])/'cls_labels_onehot.npy',allow_pickle=True).item()

    def __getitem__(self,index):
        name=self.names[index]
        image=Image.open(Path(self.plan['data_folder'])/'JPEGImages'/f'{name}.jpg').convert('RGB')
        label=Image.open(self.cache/'labels'/f'{name}.png')
        return (*weak_strong_pair(image,label,self.rng,self.plan),torch.as_tensor(self.tags[name],dtype=torch.float32))


def consistency_loss(student,teacher,labels,valid,image_tags,temperature=2.,confidence=.7):
    allowed=torch.cat([torch.ones_like(image_tags[:,:1]),image_tags],1).bool()[:,:,None,None]
    teacher=teacher.detach().masked_fill(~allowed,-1e4)
    probability=F.softmax(teacher,dim=1);certainty,winner=probability.max(1)
    mask=valid&(certainty>=confidence)&((labels==255)|(winner==labels))
    divergence=F.kl_div(F.log_softmax(student/temperature,dim=1),F.softmax(teacher/temperature,dim=1),reduction='none').sum(1)*(temperature**2)
    loss=divergence[mask].mean() if mask.any() else student.sum()*0
    return loss,mask


@torch.no_grad()
def ema_parameters(ema,model,decay):
    values=dict(model.named_parameters())
    for name,value in ema.items():value.mul_(decay).add_(values[name].detach(),alpha=1-decay)


def export_state(model,ema):
    state={'module.'+k:v.detach().cpu() for k,v in model.state_dict().items()}
    for k,v in ema.items():state['module.'+k]=v.detach().cpu()
    return state
