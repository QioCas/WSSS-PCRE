"""Mine train-only positive/absent-class negatives and fit a small density model."""
import argparse,heapq,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json,sha256
from utils.pcre_experiment_state import atomic_save


def main():
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    import torch
    import torch.nn.functional as F
    from utils.decoder_selftraining import load_baseline,normalized_image,frozen_features
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int,default=0)
    opts=ap.parse_args();p=json.loads(opts.plan.read_text());out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(p['reliable_cache']);manifest=json.loads((source/'manifest.json').read_text())
    train=(Path(p['list_folder'])/'train_aug.txt').read_text().split();val=(Path(p['list_folder'])/'val.txt').read_text().split()
    assert len(train)==10582 and len(set(train))==10582 and not set(train)&set(val)
    assert manifest['names']==train and not manifest['partial_cache']
    names=train[:opts.limit] if opts.limit else train
    tags=np.load(Path(p['list_folder'])/'cls_labels_onehot.npy',allow_pickle=True).item()
    teacher=Path(p['teacher_run'])/'checkpoints'/p['teacher_checkpoint'];teacher_hash=sha256(teacher)
    assert teacher_hash==manifest['teacher_sha256']
    assert sha256(Path(p['list_folder'])/'cls_labels_onehot.npy')==manifest['tags_sha256']
    torch.manual_seed(p['seed']);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(p['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=load_baseline(p);model.requires_grad_(False);rng=np.random.default_rng(p['seed'])
    positives=[[] for _ in range(21)];negative_heap=[[] for _ in range(21)];seen=np.zeros(21,np.int64)
    serial=0;start=time.monotonic();counts=np.zeros(21,np.int64)
    with torch.no_grad():
        for number,name in enumerate(names,1):
            image=Image.open(Path(p['data_folder'])/'JPEGImages'/f'{name}.jpg').convert('RGB')
            rgb=F.interpolate(normalized_image(image)[None].cuda(),size=(448,448),mode='bilinear',align_corners=False)
            features=frozen_features(model,rgb);h,w=features.shape[-2:]
            vector=F.normalize(features,dim=1)[0].flatten(1).T.cpu().numpy()
            label=np.array(Image.open(source/'labels'/f'{name}.png').resize((w,h),Image.Resampling.NEAREST))
            for c in range(21):
                core=ndimage.binary_erosion(label==c,iterations=1)
                candidates=np.flatnonzero(core.ravel())
                if not len(candidates):continue
                chosen=rng.choice(candidates,min(len(candidates),p['bank_positive_per_image']),replace=False)
                for index in chosen:
                    seen[c]+=1;counts[c]+=1;value=vector[index].copy()
                    if len(positives[c])<p['bank_positive_samples']:positives[c].append(value)
                    else:
                        slot=int(rng.integers(seen[c]))
                        if slot<len(positives[c]):positives[c][slot]=value
            base=model.decoder(features).softmax(1)[0]
            # Do not mine another confidently segmented foreground object's parts.
            background=(base[0]>=p['bank_background_confidence']).flatten()
            cam=F.conv2d(features,model.classifier.weight)[0].sigmoid().flatten(1)
            for c in np.flatnonzero(np.asarray(tags[name])==0)+1:
                score=cam[c-1].masked_fill(~background,-1)
                k=min(p['bank_negative_per_image'],int(background.sum()))
                if not k:continue
                values,indices=score.topk(k)
                for value,index in zip(values.cpu().tolist(),indices.cpu().tolist()):
                    serial+=1;entry=(value,serial,number,vector[index].copy());heap=negative_heap[c]
                    if len(heap)<p['bank_negative_samples']:heapq.heappush(heap,entry)
                    elif value>heap[0][0]:heapq.heapreplace(heap,entry)
            if number%100==0 or number==len(names):
                row={'stage':'bank','completed':number,'total':len(names),'seconds':time.monotonic()-start}
                atomic_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    dimension=vector.shape[1];q,_=np.linalg.qr(rng.normal(size=(dimension,p['projection_dim'])))
    projection=q.astype(np.float32);dim=projection.shape[1]
    bank={'positive':np.zeros((21,p['bank_positive_prototypes'],dimension),np.float32),
          'negative':np.zeros((21,p['bank_negative_prototypes'],dimension),np.float32),
          'positive_valid':np.zeros(21,bool),'negative_valid':np.zeros(21,bool),'density_valid':np.zeros(21,bool),
          'projection':projection,'gmm_means':np.zeros((21,p['gmm_components'],dim),np.float32),
          'gmm_variances':np.ones((21,p['gmm_components'],dim),np.float32),
          'gmm_weights':np.full((21,p['gmm_components']),1/p['gmm_components'],np.float32),
          'density_cutoff':np.full(21,-1e6,np.float32)}
    diagnostics=[]
    for c in range(21):
        positive=np.asarray(positives[c],dtype=np.float32)
        row={'class_id':c,'positive_candidates':int(counts[c]),'positive_samples':len(positive),
             'negative_candidates_retained':len(negative_heap[c])}
        if len(positive)>=max(16,p['bank_positive_prototypes']):
            estimator=KMeans(n_clusters=p['bank_positive_prototypes'],n_init=3,random_state=p['seed']).fit(positive)
            centers=estimator.cluster_centers_;centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
            bank['positive'][c]=centers;bank['positive_valid'][c]=True
            projected=positive@projection;projected/=np.maximum(np.linalg.norm(projected,axis=1,keepdims=True),1e-8)
            density=GaussianMixture(n_components=p['gmm_components'],covariance_type='diag',reg_covar=.005,
                                    n_init=1,max_iter=100,random_state=p['seed']).fit(projected)
            bank['gmm_means'][c]=density.means_;bank['gmm_variances'][c]=density.covariances_
            bank['gmm_weights'][c]=density.weights_;bank['density_cutoff'][c]=np.quantile(density.score_samples(projected),p['gmm_lower_quantile'])
            bank['density_valid'][c]=True;row['gmm_converged']=bool(density.converged_)
        ordered=sorted(negative_heap[c],key=lambda e:e[0],reverse=True)
        if len(ordered)>=p['bank_negative_prototypes']:
            # Pick distinct hard-negative directions, retaining the hardest first.
            pool=np.stack([v[3] for v in ordered]);selected=[0]
            for _ in range(1,p['bank_negative_prototypes']):
                distance=1-(pool@pool[selected].T).max(1);distance[selected]=-1
                selected.append(int(distance.argmax()))
            bank['negative'][c]=pool[selected];bank['negative_valid'][c]=True
            row['negative_source_images']=len({ordered[i][2] for i in selected})
        diagnostics.append(row)
    tensors={k:torch.from_numpy(v) for k,v in bank.items()}
    if not opts.limit:
        assert bank['positive_valid'].all() and bank['density_valid'].all(),'Missing positive class support'
        assert bank['negative_valid'][1:].all(),'Missing negative class support'
    atomic_save({'format':'precision_bank_v1','tensors':tensors},out/'bank.pth')
    provenance={'format':'precision_bank_v1','images':len(names),'partial_bank':bool(opts.limit),
                'teacher_sha256':teacher_hash,'tags_sha256':manifest['tags_sha256'],
                'reliable_manifest_sha256':sha256(source/'manifest.json'),'bank_sha256':sha256(out/'bank.pth'),
                'training_pixel_gt_read':False,'negative_mining_before_tag_filter':True,
                'negative_requires_teacher_background':p['bank_background_confidence'],
                'class_diagnostics':diagnostics,'parameters':{k:v for k,v in p.items() if k.startswith(('bank_','gmm_','projection_'))}}
    atomic_json(out/'manifest.json',provenance);atomic_json(out/'progress.json',{'stage':'complete','completed':len(names),'total':len(names)})
    print(json.dumps({'completed':len(names),'valid_positive_classes':int(bank['positive_valid'].sum()),
                      'valid_negative_classes':int(bank['negative_valid'].sum())}),flush=True)


if __name__=='__main__':main()
