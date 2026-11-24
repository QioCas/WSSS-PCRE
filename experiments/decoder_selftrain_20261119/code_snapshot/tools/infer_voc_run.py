"""Evaluate a new PCRE run using the exact paired-checkpoint inference engine."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import evaluate_one,atomic_json

def main():
    import json,os
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--mode',default='adaptive_consensus')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--scales',type=float,nargs='+',default=[1.,1.25,1.5])
    ap.add_argument('--crf-workers',type=int,default=4)
    ap.add_argument('--gpu-memory-gib',type=float,default=8.)
    opts=ap.parse_args();os.chdir(ROOT)
    opts.output_dir=opts.output_dir.resolve();opts.output_dir.mkdir(parents=True,exist_ok=True)
    result=evaluate_one(opts.mode,opts.run_dir.resolve(),opts.output_dir,opts)
    summary={'candidate':{s:result[s]['miou']*100 for s in ['raw','crf']},'partial_validation':bool(opts.limit)}
    if not opts.limit:
        baseline=json.loads((ROOT/'experiments/inference_voc_val_20261119/full/baseline/metrics.json').read_text())
        summary['baseline']={s:baseline[s]['miou']*100 for s in ['raw','crf']}
        summary['delta_vs_baseline_pp']={s:summary['candidate'][s]-summary['baseline'][s] for s in ['raw','crf']}
        summary['better_on_voc_val']=summary['delta_vs_baseline_pp']['crf']>0
    atomic_json(opts.output_dir/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
