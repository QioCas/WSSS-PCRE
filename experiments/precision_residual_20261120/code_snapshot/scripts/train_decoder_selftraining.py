"""Refine only the baseline ASPP decoder using cached teacher targets and KL anchoring."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json,sha256
from utils.pcre_experiment_state import atomic_save,append_json


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from utils.decoder_selftraining import (load_baseline,frozen_features,initialize_heads,
        PseudoLabelDataset,balanced_cross_entropy,anchor_kl,update_ema,refined_state)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--mode',choices=['raw_teacher','crf_teacher'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    opts=ap.parse_args();plan=json.loads(opts.plan.read_text());out=opts.output.resolve()
    out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    manifest=json.loads((opts.cache/'manifest.json').read_text())
    assert not manifest['partial_cache'] or opts.smoke,'Partial cache is smoke-only'
    assert sha256(Path(plan['teacher_run'])/'checkpoints'/plan['teacher_checkpoint'])==manifest['teacher_sha256']
    steps=4 if opts.smoke else plan['refinement_steps']
    torch.manual_seed(plan['seed']);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,base_args=load_baseline(plan)
    teacher,ema=initialize_heads(model)
    optimizer=torch.optim.AdamW(model.decoder.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable==sum(p.numel() for p in model.decoder.parameters())
    frozen_before={k:sha256_tensor(v) for k,v in model.state_dict().items() if not k.startswith('decoder.')}
    image_rng=random.Random(plan['seed']);order_rng=np.random.default_rng(plan['seed'])
    dataset=PseudoLabelDataset(plan,opts.cache,opts.mode,image_rng,names=manifest['names'])
    order=order_rng.permutation(len(dataset.names));cursor=0;start=time.monotonic()
    config=vars(base_args).copy()
    config.update(experiment_mode='decoder_'+opts.mode,refinement_steps=steps,
                  teacher_checkpoint=plan['teacher_checkpoint'],teacher_run=plan['teacher_run'],
                  training_pixel_gt_read=False,trainable_parameters=trainable,backbone_frozen=True,
                  data_folder=plan['data_folder'],list_folder=plan['list_folder'])
    atomic_json(out/'config.json',config);atomic_json(out/'refinement_plan.json',plan)
    for step in range(1,steps+1):
        if cursor+plan['batch_size']>len(order):order=order_rng.permutation(len(order));cursor=0
        indices=order[cursor:cursor+plan['batch_size']];cursor+=plan['batch_size']
        examples=[dataset[int(i)] for i in indices]
        inputs=torch.stack([x[0] for x in examples]).cuda()
        labels=torch.stack([x[1] for x in examples]).cuda()
        model.eval();teacher.eval();ema.eval()
        with torch.no_grad():
            features=frozen_features(model,inputs)
            reference=teacher(features)
        logits=model.decoder(features)
        dense=F.interpolate(logits,size=labels.shape[-2:],mode='bilinear',align_corners=False)
        ce=balanced_cross_entropy(dense,labels)
        kl=anchor_kl(logits,reference,plan['anchor_temperature'])
        loss=ce+plan['anchor_weight']*kl
        if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at step {step}')
        lr=plan['lr']*min(1.,step/max(1,plan['warmup_steps']))*.5*(1+math.cos(math.pi*(step-1)/max(1,steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step();update_ema(ema,model.decoder,plan['ema_decay'])
        if step<=4 or step%plan['log_interval']==0 or step==steps:
            row={'step':step,'total':steps,'ce':float(ce.detach()),'anchor_kl':float(kl.detach()),
                 'loss':float(loss.detach()),'lr':lr,'seconds':time.monotonic()-start,
                 'labeled_fraction':float((labels!=255).float().mean()),
                 'trainable_parameters':trainable,'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3}
            append_json(out/'training_metrics.jsonl',row);atomic_json(out/'progress.json',row)
            print(json.dumps(row),flush=True)
        if step%plan['checkpoint_interval']==0 or step==steps:
            payload={'format':'decoder_selftrain_v1','step':step,'decoder':model.decoder.state_dict(),
                     'ema_decoder':ema.state_dict(),'optimizer':optimizer.state_dict(),
                     'order':order,'cursor':cursor,'order_rng':order_rng.bit_generator.state,
                     'augmentation_rng':image_rng.getstate(),'torch_rng':torch.get_rng_state(),
                     'cuda_rng':torch.cuda.get_rng_state_all(),'plan':plan,
                     'teacher_sha256':manifest['teacher_sha256']}
            atomic_save(payload,out/'checkpoints'/f'training_state_step_{step:08d}.pth')
        del inputs,labels,features,reference,logits,dense,loss,ce,kl
    frozen_after={k:sha256_tensor(v) for k,v in model.state_dict().items() if not k.startswith('decoder.')}
    assert frozen_before==frozen_after,'Frozen baseline weights changed'
    checkpoints={}
    export_heads=[('ema',ema)] if opts.smoke else [('student',model.decoder),('ema',ema)]
    for label,head in export_heads:
        name=f'model_refined_step_{steps:08d}_{label}.pth'
        atomic_save(refined_state(model,head),out/'checkpoints'/name)
        checkpoints[label]=name
    summary={'completed_steps':steps,'partial_training':opts.smoke,'frozen_weights_unchanged':True,
             'trainable_parameters':trainable,'backbone_frozen':True,'checkpoints':checkpoints,
             'training_pixel_gt_read':False,'seconds':time.monotonic()-start}
    atomic_json(out/'training_summary.json',summary)
    atomic_json(out/'progress.json',{'stage':'complete',**summary})
    print(json.dumps(summary,indent=2),flush=True)


def sha256_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

if __name__=='__main__':main()
