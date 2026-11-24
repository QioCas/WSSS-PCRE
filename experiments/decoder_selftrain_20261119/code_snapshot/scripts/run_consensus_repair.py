"""Detached, source-checked repair pipeline: preflight -> smoke -> 20k -> inference.

It waits for available GPU memory, preserves other GPU jobs, and records failure
instead of retrying invalid runs or adopting a candidate that scores lower.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_pcre_experiments import preflight,wait_gpu,run_training,event

def hashes():
    paths=[]
    for directory in ['scripts','utils','model','engine','datasets','tools']:
        paths.extend((ROOT/directory).rglob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def worker(directory):
    os.chdir(ROOT)
    cfg=json.loads((directory/'queue_config.json').read_text())
    expected=json.loads((directory/'launch_source_hashes.json').read_text())
    try:
        report=preflight(cfg)
        (directory/'preflight.json').write_text(json.dumps(report,indent=2))
        event(directory,'preflight_completed',preflight=report)
        for smoke in [True,False]:
            wait_gpu(cfg,directory)
            if hashes()!=expected:
                raise RuntimeError('Source changed since queue launch; refusing an unrecorded experiment')
            run_training(cfg,directory,'adaptive_consensus',smoke=smoke)
        event(directory,'inference_running')
        command=[sys.executable,'-u','tools/infer_voc_run.py',
                 '--run-dir',str(directory/'adaptive_consensus'),
                 '--output-dir',str(directory/'inference')]
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
        with (directory/'inference.console.log').open('x') as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((directory/'inference/summary.json').read_text())
        decision={'status':'completed','summary':summary,
                  'single_model_choice':'adaptive_consensus' if summary['better_on_voc_val'] else 'baseline',
                  'qualification':'VOC val development result, one seed; no independent-test or multi-seed claim.'}
        (directory/'decision.json').write_text(json.dumps(decision,indent=2))
        event(directory,'completed',**decision)
    except Exception as error:
        event(directory,'failed',error=str(error))
        raise

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job-dir',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=Path('configs/pcre_consensus_repair.json'))
    ap.add_argument('--worker',action='store_true')
    opts=ap.parse_args();os.chdir(ROOT);directory=opts.job_dir.resolve()
    if opts.worker:
        worker(directory);return
    directory.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(opts.config.read_text())
    (directory/'queue_config.json').write_text(json.dumps(cfg,indent=2))
    (directory/'launch_source_hashes.json').write_text(json.dumps(hashes(),indent=2))
    with (directory/'supervisor.log').open('x') as stream:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--job-dir',str(directory)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (directory/'supervisor.pid').write_text(str(proc.pid)+'\n')
    print(json.dumps({'job_dir':str(directory),'pid':proc.pid}),flush=True)

if __name__=='__main__':main()
