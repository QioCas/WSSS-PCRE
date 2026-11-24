"""Evaluate a declared refinement checkpoint with the existing full VOC recipe."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import evaluate_one,atomic_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tag',required=True)
    ap.add_argument('--kind',choices=['student','ema'],required=True)
    ap.add_argument('--limit',type=int,default=0)
    opts=ap.parse_args();plan=json.loads(opts.plan.read_text());os.chdir(ROOT)
    training=json.loads((opts.run_dir/'training_summary.json').read_text())
    if training['partial_training'] and not opts.limit:
        raise ValueError('Smoke checkpoints must never enter full evaluation results')
    opts.checkpoint_name=training['checkpoints'][opts.kind]
    opts.scales=plan['teacher_scales'];opts.crf_workers=plan['crf_workers'];opts.gpu_memory_gib=plan['gpu_memory_gib']
    opts.output.mkdir(parents=True,exist_ok=True)
    result=evaluate_one(opts.tag,opts.run_dir,opts.output,opts)
    summary={'mode':opts.tag,'partial_evaluation':bool(opts.limit),
             'refinement_steps':training['completed_steps'],
             'raw':result['raw']['miou']*100,'crf':result['crf']['miou']*100}
    if not opts.limit:
        baseline=json.loads(Path(plan['reference_baseline_metrics']).read_text())
        summary['delta_vs_baseline_pp']={s:result[s]['miou']*100-baseline[s]['miou']*100 for s in ['raw','crf']}
    atomic_json(opts.output/opts.tag/'comparison.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
