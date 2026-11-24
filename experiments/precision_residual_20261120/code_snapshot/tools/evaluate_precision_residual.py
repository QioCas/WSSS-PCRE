"""Evaluate a bounded residual adapter using the unchanged VOC baseline recipe."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import evaluate_one,atomic_json
from utils.precision_residual import load_residual_checkpoint


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--limit',type=int,default=0)
    opts=ap.parse_args();p=json.loads(opts.plan.read_text());os.chdir(ROOT)
    training=json.loads((opts.run_dir/'training_summary.json').read_text())
    assert not training['partial_training'] or opts.limit,'No smoke checkpoint in full evaluation'
    opts.checkpoint_name=training['checkpoints']['ema'];opts.scales=p['teacher_scales']
    opts.crf_workers=p['crf_workers'];opts.gpu_memory_gib=p['gpu_memory_gib'];opts.model_loader=load_residual_checkpoint
    opts.output.mkdir(parents=True,exist_ok=True);v=evaluate_one(opts.tag,opts.run_dir,opts.output,opts)
    summary={'mode':opts.tag,'partial_evaluation':bool(opts.limit),'raw':v['raw']['miou']*100,'crf':v['crf']['miou']*100}
    if not opts.limit:
        baseline=json.loads(Path(p['reference_baseline_metrics']).read_text())
        summary['delta_vs_baseline_pp']={k:(v[k]['miou']-baseline[k]['miou'])*100 for k in ['raw','crf']}
    atomic_json(opts.output/opts.tag/'comparison.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
