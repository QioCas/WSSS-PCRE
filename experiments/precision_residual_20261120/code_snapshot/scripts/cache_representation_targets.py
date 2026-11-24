"""Six-view reliability cache or externally SAM-refined teacher targets; no VOC mask GT."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json,sha256


def main():
    import numpy as np
    from PIL import Image
    import torch
    from utils.decoder_selftraining import load_baseline,normalized_image
    from utils.representation_training import view_probabilities,reliable_labels
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--kind',choices=['reliable','sam'],required=True)
    ap.add_argument('--reliable',type=Path);ap.add_argument('--limit',type=int,default=0)
    opts=ap.parse_args();plan=json.loads(opts.plan.read_text());out=opts.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    for folder in ['labels','stats']:(out/folder).mkdir(exist_ok=True)
    names=(Path(plan['list_folder'])/(plan['train_split']+'.txt')).read_text().split()
    val=(Path(plan['list_folder'])/(plan['val_split']+'.txt')).read_text().split()
    assert len(names)==len(set(names))==10582 and not set(names)&set(val)
    if opts.limit:names=names[:opts.limit]
    tags=np.load(Path(plan['list_folder'])/'cls_labels_onehot.npy',allow_pickle=True).item()
    teacher=Path(plan['teacher_run'])/'checkpoints'/plan['teacher_checkpoint']
    manifest={'names':names,'kind':opts.kind,'partial_cache':bool(opts.limit),'training_pixel_gt_read':False,
              'teacher_sha256':sha256(teacher),'tags_sha256':sha256(Path(plan['list_folder'])/'cls_labels_onehot.npy'),
              'external_segmentation_prior':opts.kind=='sam','confidence':plan['cache_confidence'],
              'minimum_view_votes':plan['minimum_view_votes'],'scales':plan['teacher_scales']}
    if opts.kind=='sam':
        if opts.reliable is None:raise ValueError('SAM needs a reliability cache')
        source=json.loads((opts.reliable/'manifest.json').read_text())
        old=json.loads((Path(plan['old_teacher_cache'])/'manifest.json').read_text())
        assert source['names']==names and source['teacher_sha256']==manifest['teacher_sha256']==old['teacher_sha256']
        assert source['tags_sha256']==manifest['tags_sha256']==old['tags_sha256']
        assert set(names)<=set(old['names']) and old['scales']==plan['teacher_scales']
        manifest.update(sam_sha256=sha256(plan['sam_checkpoint']),reliable_manifest_sha256=sha256(opts.reliable/'manifest.json'),
                        old_reference_manifest_sha256=sha256(Path(plan['old_teacher_cache'])/'manifest.json'),
                        sam_thresholds={k:v for k,v in plan.items() if k.startswith('sam_') and isinstance(v,(float,int))})
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest,'Incompatible cache reuse'
    else:atomic_json(out/'manifest.json',manifest)
    torch.manual_seed(plan['seed']);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if opts.kind=='reliable':model,_=load_baseline(plan);model.requires_grad_(False)
    else:
        sys.path.insert(0,plan['sam_module_dir'])
        from segment_anything import SamPredictor,sam_model_registry
        from utils.sam_targets import propose_from_sam,compose_targets
        model=sam_model_registry['vit_b'](checkpoint=plan['sam_checkpoint']).eval().requires_grad_(False).cuda()
        predictor=SamPredictor(model)
    start=time.monotonic()
    for index,name in enumerate(names,1):
        target=out/'labels'/f'{name}.png';statfile=out/'stats'/f'{name}.json'
        if not (target.exists() and statfile.exists()):
            image=Image.open(Path(plan['data_folder'])/'JPEGImages'/f'{name}.jpg').convert('RGB')
            if opts.kind=='reliable':
                valid_ids=np.r_[0,np.flatnonzero(np.asarray(tags[name])>0)+1]
                inputs=normalized_image(image)[None].cuda()
                mean,votes,views=view_probabilities(model,inputs,(image.height,image.width),valid_ids,plan['teacher_scales'])
                label,confidence,agreement=reliable_labels(mean,votes,valid_ids,views,plan['cache_confidence'],plan['minimum_view_votes'])
                stats={'confidence_sum':float(confidence.sum()),'agreement_sum':float(agreement.sum()),
                       'vote_histogram':np.bincount(np.rint(agreement*views).astype(int).ravel(),minlength=views+1).tolist()}
                del inputs,mean,votes
            else:
                reliable=np.array(Image.open(opts.reliable/'labels'/f'{name}.png'))
                reference=np.array(Image.open(Path(plan['old_teacher_cache'])/'crf_teacher'/f'{name}.png'))
                assert reliable.shape==reference.shape==(image.height,image.width)
                predictor.set_image(np.array(image))
                proposals,attempted=propose_from_sam(predictor,reliable,reference,np.asarray(tags[name]),plan)
                label,stats=compose_targets(reliable,proposals,plan)
                stats.update(attempted_proposals=attempted,accepted_proposals=len(proposals),
                             sam_quality_scores=[p['sam_predicted_quality'] for p in proposals])
            ids=set(np.unique(label).tolist());assert ids<=({0,255}|set((np.flatnonzero(np.asarray(tags[name])>0)+1).tolist()))
            stats.update(image=name,pixels=label.size,foreground=int(((label>0)&(label<21)).sum()),
                         background=int((label==0).sum()),ignore=int((label==255).sum()))
            temporary=target.with_suffix('.partial.png');Image.fromarray(label).save(temporary);temporary.replace(target)
            atomic_json(statfile,stats)
        if index%100==0 or index==len(names):
            progress={'stage':opts.kind+'_cache','completed':index,'total':len(names),'seconds':time.monotonic()-start}
            atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    stats=[json.loads((out/'stats'/f'{name}.json').read_text()) for name in names]
    pixels=sum(s['pixels'] for s in stats)
    summary={'images':len(names),'pixels':pixels,'partial_cache':bool(opts.limit),'training_pixel_gt_read':False,
             'external_segmentation_prior':opts.kind=='sam',
             **{k:sum(s[k] for s in stats)/pixels for k in ['foreground','background','ignore']}}
    if opts.kind=='sam':
        summary.update({k:sum(s[k] for s in stats) for k in ['attempted_proposals','accepted_proposals','changed_pixels','added_foreground','trimmed_foreground','foreground_class_reassignments']})
        summary['changed_fraction']=summary['changed_pixels']/pixels
    else:summary.update(confidence_mean=sum(s['confidence_sum'] for s in stats)/pixels,agreement_mean=sum(s['agreement_sum'] for s in stats)/pixels)
    assert summary['foreground']>.001 and summary['ignore']<.98,'Insufficient training supervision'
    atomic_json(out/'summary.json',summary);atomic_json(out/'progress.json',{'stage':'complete','completed':len(names),'total':len(names)})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
