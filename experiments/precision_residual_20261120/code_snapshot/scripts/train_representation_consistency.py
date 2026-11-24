"""Matched decoder/last-two-block trials with frozen weak-view teacher guidance."""
import argparse,json,math,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json,sha256
from utils.pcre_experiment_state import atomic_save,append_json


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from utils.decoder_selftraining import load_baseline,frozen_features,balanced_cross_entropy
    from utils.representation_training import (student_features,select_trainable,ConsistencyDataset,
        consistency_loss,ema_parameters,export_state)
    from scripts.train_decoder_selftraining import sha256_tensor
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--mode',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    opts=ap.parse_args();plan=json.loads(opts.plan.read_text());arm=next(a for a in plan['arms'] if a['name']==opts.mode)
    out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    manifest=json.loads((opts.cache/'manifest.json').read_text())
    assert not manifest['partial_cache'] or opts.smoke
    assert manifest['external_segmentation_prior']==arm['external_segmentation_prior']
    assert manifest['teacher_sha256']==sha256(Path(plan['teacher_run'])/'checkpoints'/plan['teacher_checkpoint'])
    torch.manual_seed(plan['seed']);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    student,args=load_baseline(plan);teacher,_=load_baseline(plan);teacher.requires_grad_(False)
    selected=select_trainable(student,arm['last_blocks']);selected_set=set(selected)
    parameters=dict(student.named_parameters())
    head=[v for k,v in parameters.items() if k in selected_set and k.startswith('decoder.')]
    backbone=[v for k,v in parameters.items() if k in selected_set and not k.startswith('decoder.')]
    groups=[{'params':head,'lr':plan['lr'],'base_lr':plan['lr']}]
    if backbone:groups.append({'params':backbone,'lr':plan['backbone_lr'],'base_lr':plan['backbone_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=plan['weight_decay'])
    ema={k:parameters[k].detach().clone() for k in selected}
    frozen_before={k:sha256_tensor(v) for k,v in student.state_dict().items() if k not in selected_set}
    teacher_before={k:sha256_tensor(v) for k,v in teacher.state_dict().items()}
    config=vars(args).copy();config.update(experiment_mode=opts.mode,train_last_blocks=arm['last_blocks'],
        external_segmentation_prior=arm['external_segmentation_prior'],training_pixel_gt_read=False,
        trainable_parameters=sum(v.numel() for v in parameters.values() if v.requires_grad),
        data_folder=plan['data_folder'],list_folder=plan['list_folder'])
    atomic_json(out/'config.json',config);atomic_json(out/'refinement_plan.json',plan)
    atomic_json(out/'trainable_parameters.json',{'names':selected,'count':config['trainable_parameters']})
    image_rng=random.Random(plan['seed']);order_rng=np.random.default_rng(plan['seed'])
    dataset=ConsistencyDataset(plan,opts.cache,manifest['names'],image_rng)
    order=order_rng.permutation(len(manifest['names']));cursor=0
    steps=4 if opts.smoke else plan['refinement_steps'];start=time.monotonic()
    for step in range(1,steps+1):
        if cursor+plan['batch_size']>len(order):order=order_rng.permutation(len(order));cursor=0
        indices=order[cursor:cursor+plan['batch_size']];cursor+=plan['batch_size']
        samples=[dataset[int(i)] for i in indices]
        weak,strong,labels,valid,tags=[torch.stack([sample[i] for sample in samples]).cuda() for i in range(5)]
        student.eval();teacher.eval()
        with torch.no_grad():
            reference=teacher.decoder(frozen_features(teacher,weak))
            reference=F.interpolate(reference,size=labels.shape[-2:],mode='bilinear',align_corners=False)
        logits=student.decoder(student_features(student,strong))
        logits=F.interpolate(logits,size=labels.shape[-2:],mode='bilinear',align_corners=False)
        ce=balanced_cross_entropy(logits,labels)
        kl,kd_mask=consistency_loss(logits,reference,labels,valid,tags,plan['anchor_temperature'],plan['anchor_confidence'])
        loss=ce+plan['anchor_weight']*kl
        if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at {step}')
        multiplier=min(1.,step/max(1,plan['warmup_steps']))*.5*(1+math.cos(math.pi*(step-1)/max(1,steps)))
        for group in optimizer.param_groups:group['lr']=group['base_lr']*multiplier
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None for name,p in student.named_parameters() if name not in selected_set):raise AssertionError('Frozen parameter received a gradient')
        if backbone and step==1 and not any(p.grad is not None and p.grad.abs().sum()>0 for p in backbone):raise AssertionError('Selected backbone blocks did not receive gradients')
        norms=[float(torch.nn.utils.clip_grad_norm_(g['params'],1.,error_if_nonfinite=True)) for g in optimizer.param_groups]
        optimizer.step();ema_parameters(ema,student,plan['ema_decay'])
        if step<=4 or step%plan['log_interval']==0 or step==steps:
            row={'step':step,'total':steps,'ce':float(ce.detach()),'consistency':float(kl.detach()),'loss':float(loss.detach()),
                 'labeled_fraction':float((labels!=255).float().mean()),'consistency_fraction':float(kd_mask.float().mean()),
                 'head_lr':groups[0]['lr'],'backbone_lr':groups[1]['lr'] if backbone else 0.,'gradient_norms':norms,
                 'seconds':time.monotonic()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3}
            append_json(out/'training_metrics.jsonl',row);atomic_json(out/'progress.json',row);print(json.dumps(row),flush=True)
        if step%plan['checkpoint_interval']==0 or step==steps:
            atomic_save({'format':'representation_consistency_v1','step':step,
                         'selected_parameters':{k:parameters[k].detach() for k in selected},'ema':ema,
                         'optimizer':optimizer.state_dict(),'order':order,'cursor':cursor,
                         'order_rng':order_rng.bit_generator.state,'augmentation_rng':image_rng.getstate(),
                         'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                         'plan':plan,'teacher_sha256':manifest['teacher_sha256']},
                        out/'checkpoints'/f'training_state_step_{step:08d}.pth')
        del weak,strong,labels,valid,tags,reference,logits,loss,ce,kl,kd_mask,samples
    frozen_after={k:sha256_tensor(v) for k,v in student.state_dict().items() if k not in selected_set}
    assert frozen_before==frozen_after,'Frozen student weights changed'
    assert teacher_before=={k:sha256_tensor(v) for k,v in teacher.state_dict().items()},'Frozen teacher changed'
    checkpoint=f'model_refined_step_{steps:08d}_ema.pth'
    atomic_save(export_state(student,ema),out/'checkpoints'/checkpoint)
    summary={'completed_steps':steps,'partial_training':opts.smoke,'frozen_weights_unchanged':True,
             'teacher_unchanged':True,'train_last_blocks':arm['last_blocks'],
             'trainable_parameters':config['trainable_parameters'],'checkpoints':{'ema':checkpoint},
             'training_pixel_gt_read':False,'external_segmentation_prior':arm['external_segmentation_prior'],
             'seconds':time.monotonic()-start}
    atomic_json(out/'training_summary.json',summary);atomic_json(out/'progress.json',{'stage':'complete',**summary})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
