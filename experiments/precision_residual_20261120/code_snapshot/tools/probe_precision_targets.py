"""Development-val label-quality diagnostic, isolated from training data access."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json,sha256,serializable


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from utils.decoder_selftraining import load_baseline,normalized_image,frozen_features
    from utils.representation_training import view_probabilities,reliable_labels
    from utils.precision_residual import negative_evidence,density_evidence,veto_labels
    from scripts.cache_precision_targets import load_clip_teacher,clip_rejections
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=0);opts=ap.parse_args();p=json.loads(opts.plan.read_text())
    out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False)
    val=(Path(p['list_folder'])/'val.txt').read_text().split();train=(Path(p['list_folder'])/'train_aug.txt').read_text().split()
    assert len(val)==1449 and not set(val)&set(train)
    count=opts.limit or p['probe_images'];assert 0<count<=len(val)
    chosen=sorted(np.random.default_rng(p['seed']).choice(len(val),count,replace=False));names=[val[i] for i in chosen]
    metadata={'names':names,'images':len(names),'purpose':'Development-val pseudo-label precision/recall diagnostic only',
              'image_labels_used_for_pseudo_labels':True,'used_to_fit_bank':False,'used_for_training':False,
              'thresholds_changed_from_probe':False,'bank_sha256':sha256(opts.bank/'bank.pth'),
              'qualification':'Not full-val model inference. Val already informs development; this is not an independent test.'}
    atomic_json(out/'manifest.json',metadata)
    torch.set_num_threads(2);torch.manual_seed(p['seed'])
    torch.cuda.set_per_process_memory_fraction(p['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=load_baseline(p);model.requires_grad_(False)
    clip_model,preprocess,text=load_clip_teacher(p)
    bank={k:v.cuda() for k,v in torch.load(opts.bank/'bank.pth',weights_only=True,map_location='cpu')['tensors'].items()}
    tags=np.load(Path(p['list_folder'])/'cls_labels_onehot.npy',allow_pickle=True).item()
    modes=['reliable','negative','density','clip'];totals={m:{k:np.zeros(21,np.int64) for k in ['tp','predicted','ground_truth']} for m in modes}
    changes={m:{'removed_correct':0,'removed_wrong':0} for m in modes[1:]};start=time.monotonic()
    for mode in modes:(out/mode).mkdir()
    with torch.no_grad():
        for index,name in enumerate(names,1):
            image=Image.open(Path(p['data_folder'])/'JPEGImages'/f'{name}.jpg').convert('RGB')
            rgb=normalized_image(image)[None].cuda();ids=np.r_[0,np.flatnonzero(np.asarray(tags[name])>0)+1]
            mean,votes,views=view_probabilities(model,rgb,(image.height,image.width),ids,p['teacher_scales'])
            reliable,_,_=reliable_labels(mean,votes,ids,views,.8,5)
            features=frozen_features(model,F.interpolate(rgb,size=(448,448),mode='bilinear',align_corners=False))
            labels={'reliable':reliable}
            target=torch.as_tensor(reliable.astype(np.int64),device='cuda')
            for kind,evidence in [('negative',negative_evidence(features,bank,p['negative_margin'])),('density',density_evidence(features,bank))]:
                evidence=F.interpolate(evidence.permute(0,3,1,2).float(),size=reliable.shape,mode='nearest')[0].bool()
                mask=(evidence.gather(0,target.clamp(0,20)[None])[0]&(target>0)&(target<21)).cpu().numpy()
                rejected=np.full_like(reliable,255);rejected[mask]=reliable[mask];labels[kind]=veto_labels(reliable,rejected)[0]
            rejected,_=clip_rejections(image,reliable,np.asarray(tags[name]),clip_model,preprocess,text,p)
            labels['clip']=veto_labels(reliable,rejected)[0]
            # This tool alone reads val GT, after pseudo-label generation. Train
            # programs never import or consume its masks/ground-truth statistics.
            gt=np.array(Image.open(Path(p['data_folder'])/'SegmentationClassAug'/f'{name}.png'));valid=gt<21
            for kind,label in labels.items():
                retained=valid&(label<21);correct=retained&(label==gt)
                totals[kind]['tp']+=np.bincount(gt[correct],minlength=21)
                totals[kind]['predicted']+=np.bincount(label[retained],minlength=21)
                totals[kind]['ground_truth']+=np.bincount(gt[valid],minlength=21)
                if kind!='reliable':
                    removed=valid&(reliable<21)&(label==255)
                    changes[kind]['removed_correct']+=int((removed&(reliable==gt)).sum())
                    changes[kind]['removed_wrong']+=int((removed&(reliable!=gt)).sum())
                Image.fromarray(label).save(out/kind/f'{name}.png')
            if index%20==0 or index==len(names):
                row={'stage':'label_probe','completed':index,'total':len(names),'seconds':time.monotonic()-start}
                atomic_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    results={}
    for kind,value in totals.items():
        tp=value['tp'];pred=value['predicted'];gt=value['ground_truth']
        precision=np.divide(tp,pred,out=np.full(21,np.nan),where=pred>0)
        recall=np.divide(tp,gt,out=np.full(21,np.nan),where=gt>0)
        results[kind]={'foreground_micro_precision':float(tp[1:].sum()/max(1,pred[1:].sum())),
                       'foreground_micro_recall':float(tp[1:].sum()/max(1,gt[1:].sum())),
                       'per_class_precision':precision,'per_class_recall':recall,
                       **value,**changes.get(kind,{})}
    atomic_json(out/'summary.json',serializable({'metadata':metadata,'results':results}))
    atomic_json(out/'progress.json',{'stage':'complete','completed':len(names),'total':len(names)})
    print(json.dumps({k:{n:v[n] for n in ['foreground_micro_precision','foreground_micro_recall']} for k,v in results.items()}),flush=True)


if __name__=='__main__':main()
