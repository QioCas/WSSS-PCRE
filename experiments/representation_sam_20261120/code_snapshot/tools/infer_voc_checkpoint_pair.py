"""Evaluate the saved PCRE baseline/adaptive with the repository's VOC inference recipe.

Loads each run's config and strict model-only checkpoint. Exports raw and CRF
masks, confusion matrices, per-image/per-class metrics and resumable progress.
CRF is streamed through a bounded CPU pool so full float logits need not be
stored on disk. Full eval uses all 1449 VOC val images; --limit is smoke-only.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parents[1] / '.deps/pcre_inference'
sys.path[:0] = [str(ROOT), str(DEPS)]
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')

RUNS = {
    'baseline': ROOT/'experiments/pcre_sat_20261117_200400_39e382/baseline',
    'adaptive': ROOT/'experiments/pcre_sat_20261117_130748_9e5320/adaptive',
}
CRF_PARAMETERS = dict(iter_max=10, pos_xy_std=2, pos_w=3,
                      bi_xy_std=140, bi_rgb_std=1, bi_w=7)


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def serializable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): serializable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return serializable(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def palette():
    import numpy as np
    colors = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        value = i
        for j in range(8):
            for channel in range(3):
                colors[i, channel] |= ((value >> channel) & 1) << (7-j)
            value >>= 3
    colors[255] = (224, 224, 224)
    return colors


def save_mask(mask, output, stage, name):
    import numpy as np
    from PIL import Image
    colors = palette()
    Image.fromarray(mask.astype(np.uint8)).save(output/stage/'masks'/f'{name}.png')
    Image.fromarray(colors[mask]).save(output/stage/'color'/f'{name}.png')


def hist_for(gt, pred, nclasses=21):
    import numpy as np
    valid = (gt >= 0) & (gt < nclasses)
    return np.bincount(nclasses*gt[valid].astype(np.int64)+pred[valid],
                       minlength=nclasses*nclasses).reshape(nclasses, nclasses)


def image_score(hist):
    import numpy as np
    union = hist.sum(0) + hist.sum(1) - hist.diagonal()
    present = hist.sum(1) > 0
    return float(np.mean(hist.diagonal()[present] / union[present]))


def crf_image(image_path, label_path, probabilities, output, name):
    import numpy as np
    from PIL import Image
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    image = np.array(Image.open(image_path).convert('RGB'))
    gt = np.array(Image.open(label_path))
    _, height, width = probabilities.shape
    model = dcrf.DenseCRF2D(width, height, probabilities.shape[0])
    model.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(probabilities)))
    model.addPairwiseGaussian(sxy=CRF_PARAMETERS['pos_xy_std'], compat=CRF_PARAMETERS['pos_w'])
    model.addPairwiseBilateral(sxy=CRF_PARAMETERS['bi_xy_std'], srgb=CRF_PARAMETERS['bi_rgb_std'],
                               rgbim=np.ascontiguousarray(image), compat=CRF_PARAMETERS['bi_w'])
    q = np.array(model.inference(CRF_PARAMETERS['iter_max'])).reshape(probabilities.shape)
    prediction = q.argmax(0).astype(np.uint8)
    save_mask(prediction, Path(output), 'crf', name)
    hist = hist_for(gt, prediction)
    return name, hist, image_score(hist)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluate_one(mode, run_dir, out, opts):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from argparse import Namespace
    from datasets.voc import VOC12SegDataset, class_list
    from model.model_wsddn import network
    from torch.utils.data import DataLoader
    from utils.evaluate import compute_final_scores

    output = out/mode
    if (output/'metrics.json').exists():
        raise FileExistsError(f'Completed results already exist: {output}')
    output.mkdir(parents=True, exist_ok=True)
    for stage in ['raw', 'crf']:
        for folder in ['masks','color']:
            (output/stage/folder).mkdir(parents=True,exist_ok=True)
    config = json.loads((run_dir/'config.json').read_text())
    cfg = Namespace(**config)
    cfg.debug = False
    checkpoint_name = getattr(opts, 'checkpoint_name', 'model_iter_00020000.pth')
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError('checkpoint_name must be a filename within checkpoints/')
    checkpoint = run_dir/'checkpoints'/checkpoint_name
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(opts.gpu_memory_gib*1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = network(cfg, backbone=cfg.backbone, num_classes=cfg.num_classes,
                    pretrained=False, aux_layer=cfg.aux_layer)
    # Training expands the ImageNet encoder from one token to 20 class tokens.
    model.encoder.after_load_state_dict()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    del state
    model.eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    dataset = VOC12SegDataset(root_dir=cfg.data_folder, name_list_dir=cfg.list_folder,
                               split='val', stage='val', aug=False, ignore_index=cfg.ignore_index)
    names = list(dataset.name_list)
    total = min(opts.limit,len(names)) if opts.limit else len(names)
    assert len(names)==1449 and len(set(names))==1449
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metadata = {'mode':mode,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
                'training_run':str(run_dir),'validation_images':total,'partial_validation':bool(opts.limit),
                'scales':opts.scales,'horizontal_flip':True,'input_base_size':448,
                'precision':'float32','tf32':False,'crf':CRF_PARAMETERS,
                'fusion':'max(first_scale_flip_sum, mean(all_scale_flip_sums))',
                'strict_load':True,'image_labels_used_for_prediction':False,
                'torch_version':torch.__version__,'densecrf_distribution':'pydensecrf2==1.1'}
    atomic_json(output/'inference_config.json', metadata)
    atomic_json(output/'training_config.json', config)
    raw_hist=np.zeros((21,21),dtype=np.int64)
    crf_hist=np.zeros((21,21),dtype=np.int64)
    per_image={}
    started=time.monotonic()
    pending=deque()
    def receive():
        name,hist,score = pending.popleft().result()
        crf_hist[:] += hist
        per_image[name]['crf_miou']=score
    with ProcessPoolExecutor(max_workers=opts.crf_workers, mp_context=mp.get_context('spawn')) as executor:
        with torch.no_grad():
            for index,(name,inputs,labels,_) in enumerate(loader):
                if index>=total:
                    break
                name=name[0]
                inputs=inputs.cuda()
                scaled=[]
                for scale in opts.scales:
                    size=(int(448*scale),int(448*scale))
                    resized=F.interpolate(inputs,size=size,mode='bilinear',align_corners=False)
                    flipped=torch.cat([resized,resized.flip(-1)],dim=0)
                    segs=model(flipped,args=cfg)[1]
                    segs=F.interpolate(segs,size=labels.shape[1:],mode='bilinear',align_corners=False)
                    scaled.append(segs[:1]+segs[1:].flip(-1))
                stack=torch.stack(scaled,dim=0)
                seg=torch.maximum(stack[0],stack.mean(dim=0))
                prediction=seg.argmax(1)[0].cpu().numpy().astype(np.uint8)
                gt=labels[0].numpy()
                hist=hist_for(gt,prediction)
                raw_hist += hist
                save_mask(prediction,output,'raw',name)
                per_image[name]={'raw_miou':image_score(hist),'classes_in_gt':[int(c) for c in np.unique(gt) if c<21]}
                probabilities=F.softmax(seg,dim=1)[0].cpu().numpy()
                pending.append(executor.submit(crf_image,
                    str(Path(cfg.data_folder)/'JPEGImages'/f'{name}.jpg'),
                    str(Path(cfg.data_folder)/'SegmentationClassAug'/f'{name}.png'),
                    probabilities,str(output),name))
                if len(pending)>=2*opts.crf_workers:
                    receive()
                del segs, seg, stack, scaled, inputs, resized, flipped, probabilities
                if (index+1)%25==0 or index+1==total:
                    elapsed=time.monotonic()-started
                    progress={'stage':mode,'processed':index+1,'total':total,'crf_completed':sum('crf_miou' in r for r in per_image.values()),
                              'seconds':elapsed,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
                    atomic_json(out/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
            while pending:
                receive()
    metrics={'raw':compute_final_scores(raw_hist),'crf':compute_final_scores(crf_hist),
             'class_names':class_list,'metadata':metadata,'elapsed_seconds':time.monotonic()-started,
             'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    assert len(per_image)==total and all('crf_miou' in r for r in per_image.values())
    for stage,hist in [('raw',raw_hist),('crf',crf_hist)]:
        assert len(list((output/stage/'masks').glob('*.png')))==total
        np.save(output/f'{stage}_confusion.npy',hist)
    atomic_json(output/'metrics.json',serializable(metrics))
    atomic_json(output/'per_image.json',serializable(per_image))
    print(f'FINISHED {mode}: raw={metrics["raw"]["miou"]*100:.6f}% CRF={metrics["crf"]["miou"]*100:.6f}%',flush=True)
    del model
    torch.cuda.empty_cache()
    return serializable(metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--scales',type=float,nargs='+',default=[1.0,1.25,1.5])
    parser.add_argument('--crf-workers',type=int,default=4)
    parser.add_argument('--gpu-memory-gib',type=float,default=8.0)
    opts=parser.parse_args()
    if opts.limit<0 or opts.crf_workers<1:
        parser.error('Invalid limit or worker count')
    os.chdir(ROOT)
    opts.output_dir=opts.output_dir.resolve()
    opts.output_dir.mkdir(parents=True,exist_ok=True)
    results={}
    for mode,run_dir in RUNS.items():
        print(f'START {mode}: {run_dir}',flush=True)
        results[mode]=evaluate_one(mode,run_dir,opts.output_dir,opts)
    summary={mode:{stage:results[mode][stage]['miou']*100 for stage in ['raw','crf']} for mode in RUNS}
    summary['delta_adaptive_minus_baseline_pp']={stage:summary['adaptive'][stage]-summary['baseline'][stage] for stage in ['raw','crf']}
    atomic_json(opts.output_dir/'summary.json',summary)
    atomic_json(opts.output_dir/'progress.json',{'stage':'complete','partial_validation':bool(opts.limit),'summary':summary})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
