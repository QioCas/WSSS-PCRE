"""Cache train_aug pseudo-labels from the frozen baseline; pixel GT is never read."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json,sha256,CRF_PARAMETERS


def write_cache_item(name,image_path,probabilities,valid_ids,cache,plan):
    import numpy as np
    from PIL import Image
    sys.path.insert(0,plan['deps_folder'])
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    from utils.decoder_selftraining import confident_labels
    cache=Path(cache)
    image=np.array(Image.open(image_path).convert('RGB'))
    count,height,width=probabilities.shape
    crf=dcrf.DenseCRF2D(width,height,count)
    crf.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(probabilities)))
    crf.addPairwiseGaussian(sxy=CRF_PARAMETERS['pos_xy_std'],compat=CRF_PARAMETERS['pos_w'])
    crf.addPairwiseBilateral(sxy=CRF_PARAMETERS['bi_xy_std'],srgb=CRF_PARAMETERS['bi_rgb_std'],
                             rgbim=np.ascontiguousarray(image),compat=CRF_PARAMETERS['bi_w'])
    refined=np.array(crf.inference(CRF_PARAMETERS['iter_max'])).reshape(probabilities.shape)
    raw=confident_labels(probabilities,valid_ids,plan['confidence'],plan['margin'])
    smooth=confident_labels(refined,valid_ids,plan['confidence'],plan['margin'])
    stats={'image':name,'pixels':height*width,'present_ids':list(map(int,valid_ids)),
           'raw_crf_disagreement_pixels':int((raw!=smooth).sum())}
    for mode,label in [('raw_teacher',raw),('crf_teacher',smooth)]:
        target=cache/mode/(name+'.png');temp=target.with_suffix('.partial.png')
        Image.fromarray(label).save(temp);temp.replace(target)
        stats[mode]={'foreground':int(((label>0)&(label<21)).sum()),
                     'background':int((label==0).sum()),'ignore':int((label==255).sum())}
    atomic_json(cache/'stats'/(name+'.json'),stats)
    return stats


def main():
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from utils.decoder_selftraining import load_baseline,normalized_image,fused_teacher_logits,frozen_features
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=0)
    opts=ap.parse_args();plan=json.loads(opts.plan.read_text());out=opts.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    for folder in ['raw_teacher','crf_teacher','stats']:(out/folder).mkdir(exist_ok=True)
    names=(Path(plan['list_folder'])/(plan['train_split']+'.txt')).read_text().split()
    val=(Path(plan['list_folder'])/(plan['val_split']+'.txt')).read_text().split()
    assert len(names)==len(set(names))==10582 and not set(names)&set(val)
    if opts.limit:names=names[:opts.limit]
    tags=np.load(Path(plan['list_folder'])/'cls_labels_onehot.npy',allow_pickle=True).item()
    checkpoint=Path(plan['teacher_run'])/'checkpoints'/plan['teacher_checkpoint']
    manifest={'teacher_checkpoint':str(checkpoint),'teacher_sha256':sha256(checkpoint),
              'names':names,'train_images':len(names),'partial_cache':bool(opts.limit),
              'confidence':plan['confidence'],'margin':plan['margin'],'scales':plan['teacher_scales'],
              'crf':CRF_PARAMETERS,'training_pixel_gt_read':False,'val_images_cached':0,
              'tags_sha256':sha256(Path(plan['list_folder'])/'cls_labels_onehot.npy')}
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text())==manifest,'Incompatible cache reuse'
    else:atomic_json(out/'manifest.json',manifest)
    torch.manual_seed(plan['seed']);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,args=load_baseline(plan);model.requires_grad_(False)
    first=normalized_image(Image.open(Path(plan['data_folder'])/'JPEGImages'/f'{names[0]}.jpg').convert('RGB'))[None].cuda()
    resized=F.interpolate(first,size=(448,448),mode='bilinear',align_corners=False)
    with torch.no_grad():
        reference=model(resized,args=args)[1]
        shortcut=model.decoder(frozen_features(model,resized))
        torch.testing.assert_close(reference,shortcut,rtol=1e-5,atol=1e-5)
    atomic_json(out/'segmentation_path_parity.json',{'passed':True,'max_abs_error':float((reference-shortcut).abs().max()),'rtol':1e-5,'atol':1e-5})
    del reference,shortcut,first,resized
    completed={name for name in names if (out/'stats'/f'{name}.json').exists() and all((out/m/f'{name}.png').exists() for m in plan['modes'])}
    pending=deque();start=time.monotonic()
    def receive():
        stat=pending.popleft().result();completed.add(stat['image'])
    with ProcessPoolExecutor(max_workers=plan['crf_workers'],mp_context=mp.get_context('spawn')) as pool:
        for name in names:
            if name in completed:continue
            image_path=Path(plan['data_folder'])/'JPEGImages'/f'{name}.jpg'
            image=Image.open(image_path).convert('RGB')
            inputs=normalized_image(image)[None].cuda()
            valid_ids=np.r_[0,np.flatnonzero(np.asarray(tags[name])>0)+1]
            logits=fused_teacher_logits(model,inputs,(image.height,image.width),plan['teacher_scales'])
            probability=F.softmax(logits[:,valid_ids],dim=1)[0].cpu().numpy()
            pending.append(pool.submit(write_cache_item,name,str(image_path),probability,valid_ids,str(out),plan))
            if len(pending)>=2*plan['crf_workers']:receive()
            del inputs,logits,probability
            if completed and len(completed)%100==0:
                progress={'stage':'cache','completed':len(completed),'total':len(names),'seconds':time.monotonic()-start}
                atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        while pending:receive()
    assert len(completed)==len(names)
    stats=[json.loads((out/'stats'/f'{name}.json').read_text()) for name in names]
    pixels=sum(s['pixels'] for s in stats)
    summary={mode:{key:sum(s[mode][key] for s in stats)/pixels for key in ['foreground','background','ignore']} for mode in plan['modes']}
    for mode in plan['modes']:
        assert summary[mode]['foreground']>.001,'No useful foreground supervision'
        assert summary[mode]['ignore']<.98,'Pseudo-label coverage is too small'
    summary.update(images=len(names),pixels=pixels,partial_cache=bool(opts.limit),
                   training_pixel_gt_read=False,elapsed_seconds=time.monotonic()-start)
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'progress.json',{'stage':'complete','completed':len(names),'total':len(names)})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
