"""Veto uncertain foreground using train-only negative/density/CLIP evidence."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json,sha256


def load_clip_teacher(plan):
    import torch
    sys.path.insert(0,plan['clip_module_dir']);sys.path.insert(0,plan['clip_dependency_dir'])
    import clip
    from datasets.voc import class_list
    model,preprocess=clip.load(plan['clip_model_name'],device='cuda',jit=False,download_root=plan['clip_download_dir'])
    model.eval().requires_grad_(False)
    aliases={'aeroplane':'airplane','diningtable':'dining table','motorbike':'motorcycle',
             'pottedplant':'potted plant','tvmonitor':'television monitor'}
    prompts=['a photo of background.']+[f'a photo of a {aliases.get(c,c)}.' for c in class_list[1:]]
    with torch.no_grad():
        text=model.encode_text(clip.tokenize(prompts).cuda()).float()
        text=text/text.norm(dim=-1,keepdim=True)
    return model,preprocess,text


def clip_rejections(image,label,tags,model,preprocess,text,plan):
    import numpy as np
    import torch
    from scipy import ndimage
    proposals=[];stats={'regions':0,'vetoed_regions':0}
    ids=[0]+(np.flatnonzero(tags>0)+1).tolist()
    for c in ids[1:]:
        regions,n=ndimage.label(label==c);sizes=np.bincount(regions.ravel());sizes[0]=0
        for j in np.argsort(sizes)[::-1][:2]:
            if j==0 or sizes[j]<plan['clip_min_component_pixels']:continue
            mask=regions==j;y,x=np.where(mask);px=max(4,int((x.max()-x.min()+1)*.1));py=max(4,int((y.max()-y.min()+1)*.1))
            box=(max(0,x.min()-px),max(0,y.min()-py),min(image.width,x.max()+px+1),min(image.height,y.max()+py+1))
            proposals.append((c,mask,preprocess(image.crop(box))))
    rejected=np.full_like(label,255)
    for start in range(0,len(proposals),8):
        batch=proposals[start:start+8]
        with torch.no_grad():
            embedding=model.encode_image(torch.stack([p[2] for p in batch]).cuda()).float()
            embedding=embedding/embedding.norm(dim=-1,keepdim=True)
            similarity=(embedding@text[ids].T).cpu().numpy()
        for (c,mask,_),scores in zip(batch,similarity):
            index=ids.index(c);own=scores[index];other=np.delete(scores,index).max()
            stats['regions']+=1
            if other-own>plan['clip_margin']:rejected[mask]=c;stats['vetoed_regions']+=1
    return rejected,stats


def main():
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from utils.decoder_selftraining import load_baseline,normalized_image,frozen_features
    from utils.precision_residual import negative_evidence,density_evidence,veto_labels
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--kind',choices=['negative','density','clip'],required=True);ap.add_argument('--limit',type=int,default=0)
    opts=ap.parse_args();p=json.loads(opts.plan.read_text());out=opts.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for folder in ['labels','negative','stats']:(out/folder).mkdir(exist_ok=True)
    source=Path(p['reliable_cache']);original=json.loads((source/'manifest.json').read_text())
    train=(Path(p['list_folder'])/'train_aug.txt').read_text().split();val=(Path(p['list_folder'])/'val.txt').read_text().split()
    assert train==original['names'] and len(train)==10582 and not set(train)&set(val)
    names=train[:opts.limit] if opts.limit else train;bank_meta=json.loads((opts.bank/'manifest.json').read_text())
    assert opts.limit or not bank_meta['partial_bank']
    teacher_hash=sha256(Path(p['teacher_run'])/'checkpoints'/p['teacher_checkpoint'])
    assert teacher_hash==original['teacher_sha256']==bank_meta['teacher_sha256']
    tags=np.load(Path(p['list_folder'])/'cls_labels_onehot.npy',allow_pickle=True).item()
    assert sha256(Path(p['list_folder'])/'cls_labels_onehot.npy')==original['tags_sha256']==bank_meta['tags_sha256']
    metadata={'names':names,'kind':opts.kind,'partial_cache':bool(opts.limit),'training_pixel_gt_read':False,
              'teacher_sha256':teacher_hash,'tags_sha256':original['tags_sha256'],'source_manifest_sha256':sha256(source/'manifest.json'),
              'bank_sha256':sha256(opts.bank/'bank.pth'),'external_prior':opts.kind=='clip',
              'negative_margin':p['negative_margin'],'gmm_lower_quantile':p['gmm_lower_quantile'],'clip_margin':p['clip_margin']}
    if opts.kind=='clip':metadata['clip_sha256']=sha256(Path(p['clip_download_dir'])/'ViT-B-16.pt')
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==metadata,'Incompatible cache'
    else:atomic_json(out/'manifest.json',metadata)
    torch.manual_seed(p['seed']);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(p['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if opts.kind=='clip':
        model,preprocess,text=load_clip_teacher(p)
    else:
        model,_=load_baseline(p);model.requires_grad_(False)
        bank={k:v.cuda() for k,v in torch.load(opts.bank/'bank.pth',weights_only=True,map_location='cpu')['tensors'].items()}
    start=time.monotonic()
    for index,name in enumerate(names,1):
        statfile=out/'stats'/f'{name}.json';label_file=out/'labels'/f'{name}.png';negative_file=out/'negative'/f'{name}.png'
        if not (statfile.exists() and label_file.exists() and negative_file.exists()):
            image=Image.open(Path(p['data_folder'])/'JPEGImages'/f'{name}.jpg').convert('RGB')
            label=np.array(Image.open(source/'labels'/f'{name}.png'))
            if opts.kind=='clip':rejected,stats=clip_rejections(image,label,np.asarray(tags[name]),model,preprocess,text,p)
            else:
                with torch.no_grad():
                    rgb=F.interpolate(normalized_image(image)[None].cuda(),size=(448,448),mode='bilinear',align_corners=False)
                    features=frozen_features(model,rgb)
                    evidence=negative_evidence(features,bank,p['negative_margin']) if opts.kind=='negative' else density_evidence(features,bank)
                    evidence=F.interpolate(evidence.permute(0,3,1,2).float(),size=label.shape,mode='nearest')[0].bool()
                    target=torch.as_tensor(label.astype(np.int64),device='cuda')
                    veto=evidence.gather(0,target.clamp(0,20)[None])[0]&(target>0)&(target<21)
                    rejected=np.full_like(label,255);rejected[veto.cpu().numpy()]=label[veto.cpu().numpy()]
                stats={}
            new,negative=veto_labels(label,rejected)
            assert np.all((new==label)|(new==255)) and not ((new==0)&(label!=0)).any()
            stats.update(pixels=int(label.size),original_foreground=int(((label>0)&(label<21)).sum()),
                removed_pixels=int((new!=label).sum()),foreground=int(((new>0)&(new<21)).sum()),
                ignore=int((new==255).sum()),removed_per_class=np.bincount(negative[negative<21],minlength=21).tolist())
            for array,path in [(new,label_file),(negative,negative_file)]:
                temp=path.with_suffix('.partial.png');Image.fromarray(array).save(temp);temp.replace(path)
            atomic_json(statfile,stats)
        if index%100==0 or index==len(names):
            row={'stage':opts.kind+'_targets','completed':index,'total':len(names),'seconds':time.monotonic()-start}
            atomic_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    rows=[json.loads((out/'stats'/f'{n}.json').read_text()) for n in names]
    total=sum(r['pixels'] for r in rows);removed=sum(r['removed_pixels'] for r in rows)
    fg=sum(r['original_foreground'] for r in rows)
    summary={'images':len(names),'partial_cache':bool(opts.limit),'pixels':total,'removed_pixels':removed,
             'removed_fraction_total':removed/total,'removed_fraction_foreground':removed/max(1,fg),
             'foreground':sum(r['foreground'] for r in rows)/total,'ignore':sum(r['ignore'] for r in rows)/total,
             'training_pixel_gt_read':False,'external_prior':opts.kind=='clip',
             'removed_per_class':np.sum([r['removed_per_class'] for r in rows],axis=0).tolist()}
    assert summary['foreground']>.001,'All usable foreground rejected'
    atomic_json(out/'summary.json',summary);atomic_json(out/'progress.json',{'stage':'complete','completed':len(names),'total':len(names)})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
