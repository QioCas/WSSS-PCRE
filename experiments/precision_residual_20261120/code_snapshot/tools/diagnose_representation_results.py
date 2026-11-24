"""Post-hoc CPU diagnostics on saved VOC val predictions; never trains a model."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
from PIL import Image


def scores(hist):
    tp=np.diagonal(hist,axis1=-2,axis2=-1)
    return tp/(hist.sum(-1)+hist.sum(-2)-tp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--resamples',type=int,default=1000)
    args=parser.parse_args();job=args.job.resolve();plan=json.loads((job/'plan.json').read_text())
    output=job/'diagnostics';output.mkdir(exist_ok=True)
    baseline=Path(plan['reference_baseline_metrics']).parent
    folders={'baseline':baseline,**{a['name']:job/'evaluation'/a['name'] for a in plan['arms']}}
    metrics={k:json.loads((v/'metrics.json').read_text()) for k,v in folders.items()}
    names=(Path(plan['list_folder'])/'val.txt').read_text().split()
    assert len(names)==len(set(names))==1449
    train=(Path(plan['list_folder'])/'train_aug.txt').read_text().split();assert not set(names)&set(train)
    classes=metrics['baseline']['class_names'];nclasses=len(classes)
    hist={k:np.load(v/'crf_confusion.npy') for k,v in folders.items()}
    for k,v in hist.items():
        assert metrics[k]['metadata']['validation_images']==1449 and not metrics[k]['metadata']['partial_validation']
        assert np.array_equal(v.sum(1),hist['baseline'].sum(1))
        assert abs(scores(v).mean()-metrics[k]['crf']['miou'])<1e-12
    rows=[];decomposition={};logs={};b=metrics['baseline'];bh=hist['baseline']
    for name in list(folders)[1:]:
        value=metrics[name];h=hist[name]
        d=(scores(h)-scores(bh))*100
        decomposition[name]={'positive_class_contribution_pp':float(d[d>0].sum()/21),
            'negative_class_contribution_pp':float(d[d<0].sum()/21),
            'total_delta_pp':float(d.mean()),'improved_classes':int((d>0).sum()),
            'predicted_foreground_area_relative_change_percent':float((h.sum(0)[1:].sum()/bh.sum(0)[1:].sum()-1)*100)}
        for i,c in enumerate(classes):
            rows.append({'method':name,'class':c,'baseline_crf_iou_percent':scores(bh)[i]*100,
                'crf_iou_percent':scores(h)[i]*100,'iou_delta_pp':d[i],
                'precision_delta_pp':(value['crf']['precision'][str(i)]-b['crf']['precision'][str(i)])*100,
                'recall_delta_pp':(value['crf']['recall'][str(i)]-b['crf']['recall'][str(i)])*100,
                'predicted_area_relative_change_percent':(h.sum(0)[i]/bh.sum(0)[i]-1)*100,
                'tp':int(h[i,i]),'fp':int(h.sum(0)[i]-h[i,i]),'fn':int(h.sum(1)[i]-h[i,i])})
        sampled=[json.loads(line) for line in (job/'train'/name/'training_metrics.jsonl').read_text().splitlines()]
        logs[name]={'logged_batches':len(sampled),
            'logged_batch_means':{k:float(np.mean([r[k] for r in sampled]))
                                  for k in ['ce','consistency','labeled_fraction','consistency_fraction']},
            'mean_lower_bound_KL_on_cache_ignored_pixels_fraction':float(np.mean([
                max(0,r['consistency_fraction']-r['labeled_fraction']) for r in sampled])),
            'qualification':'Logged batches only, not every training step; KL-on-ignore value is a lower bound, not measured overlap.'}
    with (output/'class_precision_recall.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'decomposition':decomposition,'training_masks':logs},indent=2),flush=True)
    # Reconstruct image-level matrices for paired bootstrap of the same global
    # dataset mIoU, not a different mean-of-image-IoU metric.
    image_hist={k:np.zeros((len(names),21,21),np.int64) for k in folders}
    corrections={k:{'baseline_errors_corrected':0,'baseline_correct_pixels_broken':0} for k in list(folders)[1:]}
    for index,name in enumerate(names):
        gt=np.array(Image.open(Path(plan['data_folder'])/'SegmentationClassAug'/f'{name}.png'))
        valid=gt<21;truth=gt[valid].astype(np.int64);predictions={}
        for mode,folder in folders.items():
            pred=np.array(Image.open(folder/'crf/masks'/f'{name}.png'))
            assert pred.shape==gt.shape and pred.max()<21
            pred=pred[valid].astype(np.int64);predictions[mode]=pred
            image_hist[mode][index]=np.bincount(21*truth+pred,minlength=441).reshape(21,21)
        correct=predictions['baseline']==truth
        for mode,counts in corrections.items():
            counts['baseline_errors_corrected']+=int((~correct&(predictions[mode]==truth)).sum())
            counts['baseline_correct_pixels_broken']+=int((correct&(predictions[mode]!=truth)).sum())
        if (index+1)%300==0:print('val_masks_read',index+1,'/',len(names),flush=True)
    for mode in folders:assert np.array_equal(image_hist[mode].sum(0),hist[mode]),mode
    rng=np.random.default_rng(0);bootstrap={k:[] for k in list(folders)[1:]+['last2_minus_decoder','sam_minus_last2']}
    for start in range(0,args.resamples,50):
        size=min(50,args.resamples-start)
        indices=rng.integers(0,len(names),size=(size,len(names)))
        sampled={k:scores(v[indices].sum(1)).mean(-1)*100 for k,v in image_hist.items()}
        for mode in list(folders)[1:]:bootstrap[mode].extend((sampled[mode]-sampled['baseline']).tolist())
        bootstrap['last2_minus_decoder'].extend((sampled['last2_consistency']-sampled['decoder_control']).tolist())
        bootstrap['sam_minus_last2'].extend((sampled['last2_sam']-sampled['last2_consistency']).tolist())
    intervals={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in bootstrap.items()}
    result={'metric':'Global dataset CRF mIoU, 21 classes including background',
        'validation_images':len(names),'decomposition':decomposition,'training_masks':logs,
        'paired_pixel_corrections':corrections,'bootstrap_95_percentile_ci_pp':intervals,
        'bootstrap_resamples':args.resamples,'bootstrap_seed':0,'saved_confusions_reproduced_from_masks':True,
        'bootstrap_qualification':'Fixed-model whole-image resampling on development val; excludes training-seed variation and does not establish causal effects.',
        'source_metrics':{k:str(v/'metrics.json') for k,v in folders.items()},
        'training_pixel_gt_read':False,'validation_pixel_gt_read':True}
    (output/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'bootstrap':intervals,'pixel_corrections':corrections},indent=2),flush=True)


if __name__=='__main__':main()
