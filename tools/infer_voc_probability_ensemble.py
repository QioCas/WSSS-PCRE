"""Fixed 50:50 probability ensemble; no class-wise weights or val-fitted parameters.

Each checkpoint first receives exactly the existing multi-scale/flip recipe.
Only then are the two normalized probabilities averaged before the same CRF.
Ground-truth masks are consumed exclusively for scoring.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import (RUNS,CRF_PARAMETERS,atomic_json,serializable,
    save_mask,hist_for,image_score,crf_image,sha256)

def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from argparse import Namespace
    from torch.utils.data import DataLoader
    from datasets.voc import VOC12SegDataset,class_list
    from model.model_wsddn import network
    from utils.evaluate import compute_final_scores
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--crf-workers',type=int,default=4)
    opts=ap.parse_args();os.chdir(ROOT);output=opts.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models=[];configs=[];sources=[]
    for mode,run_dir in RUNS.items():
        cfg=Namespace(**json.loads((run_dir/'config.json').read_text()));cfg.debug=False
        model=network(cfg,backbone=cfg.backbone,num_classes=cfg.num_classes,pretrained=False,aux_layer=cfg.aux_layer)
        model.encoder.after_load_state_dict()
        checkpoint=run_dir/'checkpoints/model_iter_00020000.pth'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        del state
        model.eval().cuda();models.append(model);configs.append(cfg)
        sources.append({'mode':mode,'checkpoint':str(checkpoint),'sha256':sha256(checkpoint)})
    cfg=configs[0]
    dataset=VOC12SegDataset(root_dir=cfg.data_folder,name_list_dir=cfg.list_folder,split='val',stage='val',aug=False,ignore_index=255)
    total=min(opts.limit,len(dataset)) if opts.limit else len(dataset)
    for stage in ['raw','crf']:
        for folder in ['masks','color']:(output/stage/folder).mkdir(parents=True)
    metadata={'method':'fixed_equal_probability_ensemble','weights':[0.5,0.5],
              'sources':sources,'scales':[1.,1.25,1.5],'horizontal_flip':True,
              'validation_images':total,'partial_validation':bool(opts.limit),'crf':CRF_PARAMETERS,
              'val_fitted_parameters':False,'image_labels_used_for_prediction':False,
              'precision':'float32','tf32':False,'runner_sha256':sha256(__file__)}
    atomic_json(output/'inference_config.json',metadata)
    histograms={s:np.zeros((21,21),dtype=np.int64) for s in ['raw','crf']}
    per_image={};pending=deque();start=time.monotonic()
    def receive():
        name,hist,score=pending.popleft().result();histograms['crf'][:]+=hist
        per_image[name]['crf_miou']=score
    with ProcessPoolExecutor(max_workers=opts.crf_workers,mp_context=mp.get_context('spawn')) as pool:
        with torch.no_grad():
            for index,(names,inputs,labels,_) in enumerate(DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)):
                if index>=total:break
                name=names[0];inputs=inputs.cuda();probabilities=[]
                for model,config in zip(models,configs):
                    scaled=[]
                    for scale in metadata['scales']:
                        size=(int(448*scale),int(448*scale))
                        resized=F.interpolate(inputs,size=size,mode='bilinear',align_corners=False)
                        augmented=torch.cat([resized,resized.flip(-1)])
                        segs=model(augmented,args=config)[1]
                        segs=F.interpolate(segs,size=labels.shape[1:],mode='bilinear',align_corners=False)
                        scaled.append(segs[:1]+segs[1:].flip(-1))
                    stack=torch.stack(scaled)
                    fused=torch.maximum(stack[0],stack.mean(0))
                    probabilities.append(F.softmax(fused,dim=1))
                mean_prob=torch.stack(probabilities).mean(0)
                pred=mean_prob.argmax(1)[0].cpu().numpy().astype(np.uint8)
                gt=labels[0].numpy();hist=hist_for(gt,pred);histograms['raw']+=hist
                save_mask(pred,output,'raw',name)
                per_image[name]={'raw_miou':image_score(hist),'classes_in_gt':[int(c) for c in np.unique(gt) if c<21]}
                pending.append(pool.submit(crf_image,str(Path(cfg.data_folder)/'JPEGImages'/f'{name}.jpg'),
                    str(Path(cfg.data_folder)/'SegmentationClassAug'/f'{name}.png'),mean_prob[0].cpu().numpy(),str(output),name))
                if len(pending)>=2*opts.crf_workers:receive()
                del inputs,probabilities,mean_prob,stack,scaled,fused,segs,resized,augmented
                if (index+1)%25==0 or index+1==total:
                    progress={'processed':index+1,'total':total,'seconds':time.monotonic()-start,
                              'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3}
                    atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
            while pending:receive()
    assert len(per_image)==total
    result={s:compute_final_scores(h) for s,h in histograms.items()}
    result.update(metadata=metadata,class_names=class_list,elapsed_seconds=time.monotonic()-start)
    for stage,hist in histograms.items():
        assert len(list((output/stage/'masks').glob('*.png')))==total
        np.save(output/f'{stage}_confusion.npy',hist)
    atomic_json(output/'metrics.json',serializable(result));atomic_json(output/'per_image.json',per_image)
    reference=ROOT/'experiments/inference_voc_val_20261119/full'
    summary={} if opts.limit else {mode:{s:json.loads((reference/mode/'metrics.json').read_text())[s]['miou']*100 for s in ['raw','crf']} for mode in RUNS}
    summary['ensemble']={s:result[s]['miou']*100 for s in ['raw','crf']}
    if not opts.limit:
        summary['ensemble_minus_baseline_pp']={s:summary['ensemble'][s]-summary['baseline'][s] for s in ['raw','crf']}
    summary['partial_validation']=bool(opts.limit)
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'progress.json',{'stage':'complete','summary':summary})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
