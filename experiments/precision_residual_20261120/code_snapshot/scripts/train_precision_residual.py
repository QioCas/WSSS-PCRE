"""Matched, frozen-baseline residual trials with strictly masked distillation."""
import argparse,json,math,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json,sha256
from utils.pcre_experiment_state import atomic_save,append_json


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from utils.decoder_selftraining import load_baseline,balanced_cross_entropy
    from utils.precision_residual import PrecisionResidual,ResidualDataset,strict_keep_kl,complementary_loss
    from scripts.train_decoder_selftraining import sha256_tensor
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--mode',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    opts=ap.parse_args();p=json.loads(opts.plan.read_text());arm=next(a for a in p['arms'] if a['name']==opts.mode)
    out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    manifest=json.loads((opts.cache/'manifest.json').read_text())
    assert not manifest['partial_cache'] or opts.smoke
    assert bool(manifest.get('external_prior',False))==arm['external_prior']
    assert manifest['kind']==arm['target']
    teacher_hash=sha256(Path(p['teacher_run'])/'checkpoints'/p['teacher_checkpoint'])
    assert teacher_hash==manifest['teacher_sha256']
    torch.manual_seed(p['seed']);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(p['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,args=load_baseline(p);model=PrecisionResidual(base,p,arm['detail']).cuda().train()
    parameters=list(model.residual.parameters());optimizer=torch.optim.AdamW(parameters,lr=p['lr'],weight_decay=p['weight_decay'])
    ema={k:v.detach().clone() for k,v in model.residual.state_dict().items()}
    frozen_before={k:sha256_tensor(v) for k,v in base.state_dict().items()}
    config=vars(args).copy();config.update(data_folder=p['data_folder'],list_folder=p['list_folder'],
        experiment_mode=opts.mode,training_pixel_gt_read=False,external_image_text_prior=arm['external_prior'])
    atomic_json(out/'config.json',config);atomic_json(out/'plan.json',p)
    image_rng=random.Random(p['seed']);order_rng=np.random.default_rng(p['seed'])
    dataset=ResidualDataset(p,opts.cache,manifest['names'],image_rng)
    order=order_rng.permutation(len(manifest['names']));cursor=0
    steps=4 if opts.smoke else p['refinement_steps'];start=time.monotonic()
    parity=None
    for step in range(1,steps+1):
        if cursor+p['batch_size']>len(order):order=order_rng.permutation(len(order));cursor=0
        batch=[dataset[int(i)] for i in order[cursor:cursor+p['batch_size']]];cursor+=p['batch_size']
        rgb,labels,rejected,valid=[torch.stack([sample[i] for sample in batch]).cuda() for i in range(4)]
        logits,reference,gate=model.components(rgb)
        if step==1:
            assert torch.equal(logits,reference),'Residual is not exactly zero at initialization'
            with torch.no_grad():full=base(rgb,args=args)[1]
            maximum=float((full-reference).abs().max());assert maximum<=1e-6
            parity={'zero_residual_logits_bitwise_equal':True,'base_segmentation_max_abs_error':maximum,
                    'initial_residual_parameters':sum(v.numel() for v in parameters)}
            atomic_json(out/'initial_parity.json',parity)
        logits=F.interpolate(logits,size=labels.shape[-2:],mode='bilinear',align_corners=False)
        reference=F.interpolate(reference,size=labels.shape[-2:],mode='bilinear',align_corners=False)
        ce=balanced_cross_entropy(logits,labels);keep,keep_mask=strict_keep_kl(logits,reference,labels,valid)
        negative,negative_mask=complementary_loss(logits,rejected,valid)
        loss=ce+p['keep_weight']*keep+(p['negative_weight']*negative if arm['negative_loss'] else 0.)
        if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at {step}')
        schedule=min(1.,step/max(1,p['warmup_steps']))*.5*(1+math.cos(math.pi*(step-1)/steps))
        for group in optimizer.param_groups:group['lr']=p['lr']*schedule
        optimizer.zero_grad(set_to_none=True);loss.backward()
        assert all(v.grad is None for v in base.parameters()),'Frozen baseline got gradients'
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step()
        with torch.no_grad():
            for k,value in model.residual.state_dict().items():ema[k].mul_(p['ema_decay']).add_(value,alpha=1-p['ema_decay'])
        if step<=4 or step%p['log_interval']==0 or step==steps:
            row={'step':step,'total':steps,'ce':float(ce.detach()),'keep_kl':float(keep.detach()),
                 'negative_loss':float(negative.detach()),'total_loss':float(loss.detach()),
                 'labeled_fraction':float((labels!=255).float().mean()),'keep_fraction':float(keep_mask.float().mean()),
                 'negative_fraction':float(negative_mask.float().mean()),'gate_fraction':float(gate.mean()),
                 'ignored_pixels_in_keep_kl':int((keep_mask&(labels==255)).sum()),
                 'lr':optimizer.param_groups[0]['lr'],'gradient_norm':norm,'seconds':time.monotonic()-start,
                 'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3}
            append_json(out/'training_metrics.jsonl',row);atomic_json(out/'progress.json',row);print(json.dumps(row),flush=True)
        if step%p['checkpoint_interval']==0 or step==steps:
            atomic_save({'step':step,'residual':model.residual.state_dict(),'ema':ema,'optimizer':optimizer.state_dict(),
                         'order':order,'cursor':cursor,'order_rng':order_rng.bit_generator.state,
                         'augmentation_rng':image_rng.getstate(),'torch_rng':torch.get_rng_state(),
                         'cuda_rng':torch.cuda.get_rng_state_all(),'teacher_sha256':teacher_hash,'plan':p,'arm':arm},
                         out/'checkpoints'/f'training_state_step_{step:08d}.pth')
        del rgb,labels,rejected,valid,logits,reference,gate,ce,keep,negative,loss,keep_mask,negative_mask,batch
    assert frozen_before=={k:sha256_tensor(v) for k,v in base.state_dict().items()},'Frozen baseline changed'
    filename=f'adapter_step_{steps:08d}_ema.pth'
    atomic_save({'format':'precision_residual_v1','step':steps,'plan':p,'arm':arm,'teacher_sha256':teacher_hash,
                 'ema':{k:v.detach().cpu() for k,v in ema.items()}},out/'checkpoints'/filename)
    changed=float(sum((v.detach()-torch.zeros_like(v)).square().sum() for v in model.residual.layers[-1].parameters()))
    assert changed>0,'Output layer did not learn'
    summary={'completed_steps':steps,'partial_training':opts.smoke,'frozen_weights_unchanged':True,
             'initial_parity':parity,'checkpoints':{'ema':filename},'trainable_parameters':sum(v.numel() for v in parameters),
             'training_pixel_gt_read':False,'external_image_text_prior':arm['external_prior'],
             'target_manifest_sha256':sha256(opts.cache/'manifest.json'),'teacher_sha256':teacher_hash,
             'seconds':time.monotonic()-start}
    atomic_json(out/'training_summary.json',summary);atomic_json(out/'progress.json',{'stage':'complete',**summary})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
