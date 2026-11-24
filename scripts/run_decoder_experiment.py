"""Detached source-frozen decoder self-training experiment and live report."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import atomic_json
from utils.pcre_experiment_state import append_json
from utils.decoder_experiment_report import render_report
ACTIVE=None


def stamp():return datetime.now(timezone.utc).isoformat()


def status(job,stage,description,**fields):
    value={'time':stamp(),'stage':stage,'description':description,**fields}
    atomic_json(job/'status.json',value);append_json(job/'events.jsonl',value)
    render_report(job)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def resource_wait(job,plan,description):
    start=time.monotonic()
    while True:
        if (job/'STOP').exists():raise KeyboardInterrupt('STOP file requested')
        free=shutil.disk_usage(job).free/1024**3
        if free<plan['minimum_disk_free_gib']:raise RuntimeError(f'Only {free:.2f} GiB free disk')
        result=subprocess.run(['nvidia-smi','-i',str(plan['gpu']),'--query-gpu=memory.free',
                               '--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
        available=int(result.stdout.strip())
        if available>=plan['min_free_gpu_mb']:return
        status(job,'waiting_for_gpu',description,gpu_free_mb=available,disk_free_gib=free)
        if time.monotonic()-start>plan['max_gpu_wait_hours']*3600:raise TimeoutError('GPU wait budget exceeded')
        time.sleep(20)


def run_stage(job,plan,stage,description,command,progress=None,gpu=True):
    global ACTIVE
    if gpu:resource_wait(job,plan,description)
    logfile=job/'logs'/(stage+'.log')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(plan['gpu']),OMP_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled',
             PYTHONPATH=str(ROOT)+os.pathsep+plan['deps_folder'])
    with logfile.open('x') as stream:
        ACTIVE=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status(job,stage,description,pid=ACTIVE.pid,command=command,log=str(logfile))
        previous=None
        while ACTIVE.poll() is None:
            if (job/'STOP').exists():
                os.killpg(ACTIVE.pid,signal.SIGTERM);ACTIVE.wait();raise KeyboardInterrupt('STOP file requested')
            if shutil.disk_usage(job).free<3*1024**3:
                os.killpg(ACTIVE.pid,signal.SIGTERM);ACTIVE.wait();raise RuntimeError('Stopping before disk exhaustion')
            if progress and Path(progress).exists():
                try:current=json.loads(Path(progress).read_text())
                except (json.JSONDecodeError,OSError):current=None
                if stage.startswith('eval_') and current and current.get('stage')!=stage.removeprefix('eval_'):
                    current=None
                if current is not None and current!=previous:
                    completed=current.get('completed',current.get('step',current.get('processed')))
                    total=current.get('total')
                    text=description+(f' ({completed}/{total})' if completed is not None and total else '')
                    atomic_json(job/'status.json',{'time':stamp(),'stage':stage,'description':text,
                                                  'pid':ACTIVE.pid,'progress':current})
                    render_report(job);previous=current
            time.sleep(5)
        code=ACTIVE.returncode;ACTIVE=None
    if code:raise RuntimeError(f'{stage} exited {code}; see {logfile}')
    append_json(job/'events.jsonl',{'time':stamp(),'stage':stage+'_completed','log':str(logfile)})


def freeze_sources(job):
    snapshot=job/'code_snapshot';snapshot.mkdir(exist_ok=False)
    copied={}
    for directory in ['scripts','tools','utils','model','engine','datasets','tests']:
        for source in (ROOT/directory).rglob('*.py'):
            relative=source.relative_to(ROOT);target=snapshot/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            copied[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    plan=json.loads((job/'plan.json').read_text());original_lists=Path(plan['list_folder'])
    list_copy=snapshot/'datasets/voc';list_copy.mkdir(parents=True,exist_ok=True)
    for source in list(original_lists.glob('*.txt'))+[original_lists/'cls_labels_onehot.npy']:
        shutil.copy2(source,list_copy/source.name)
        copied[str((list_copy/source.name).relative_to(snapshot))]=hashlib.sha256(source.read_bytes()).hexdigest()
    plan['original_list_folder']=str(original_lists);plan['list_folder']=str(list_copy)
    plan['source_snapshot']=str(snapshot)
    atomic_json(job/'plan.json',plan);atomic_json(job/'source_hashes.json',copied)
    rev=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    (job/'git_commit.txt').write_text(rev.stdout+rev.stderr)
    return snapshot


def finalize(job,plan):
    import csv
    import numpy as np
    baseline=json.loads(Path(plan['reference_baseline_metrics']).read_text())
    base=baseline['crf']['miou']*100
    results={}
    for mode in plan['modes']:
        for kind in ['student','ema']:
            tag=mode+'_'+kind
            value=json.loads((job/'evaluation'/tag/'metrics.json').read_text())
            assert value['metadata']['validation_images']==1449 and not value['metadata']['partial_validation']
            results[tag]=value
    checks={}
    for tag in ['raw_teacher_ema','crf_teacher_ema']:
        value=results[tag]
        diffs={name:(value['crf']['iou'][str(i)]-baseline['crf']['iou'][str(i)])*100
               for i,name in enumerate(baseline['class_names'])}
        checks[tag]={'raw':value['raw']['miou']*100,'crf':value['crf']['miou']*100,
                     'gain_pp':value['crf']['miou']*100-base,'worst_class_drop_pp':min(diffs.values()),
                     'gain_target_met':value['crf']['miou']*100-base>=plan['target_gain_pp'],
                     'class_guardrail_met':min(diffs.values())>=-plan['max_allowed_class_drop_pp']}
    eligible=[tag for tag,check in checks.items() if check['gain_target_met'] and check['class_guardrail_met']]
    chosen=max(eligible,key=lambda key:checks[key]['crf']) if eligible else 'baseline'
    crf_effect=checks['crf_teacher_ema']['crf']-checks['raw_teacher_ema']['crf']
    if eligible:
        explanation=(f'Nhánh {chosen} đạt {checks[chosen]["crf"]:.4f}% sau CRF, tăng {checks[chosen]["gain_pp"]:+.4f} điểm so với baseline và đạt tiêu chí lớp. '
                     'Đây là kết quả phát triển trên val với một seed, chưa phải bằng chứng tổng quát hóa.')
    else:
        explanation=('Chưa có checkpoint EMA nào đạt đồng thời mục tiêu tăng mIoU và giới hạn giảm theo lớp. Giữ baseline; các kết quả âm được giữ nguyên trong bảng.')
    decision={'chosen':chosen,'checks':checks,'crf_minus_raw_control_pp':crf_effect,
              'crf_hypothesis_supported_on_this_run':crf_effect>0,'explanation':explanation,
              'qualification':plan['qualification']}
    atomic_json(job/'decision.json',decision)
    with (job/'per_class_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);tags=list(results)
        writer.writerow(['class','baseline_crf_iou_percent']+[tag+'_crf_iou_percent' for tag in tags])
        for i,name in enumerate(baseline['class_names']):
            writer.writerow([name,baseline['crf']['iou'][str(i)]*100]+[results[t]['crf']['iou'][str(i)]*100 for t in tags])
    status(job,'completed','Hoàn tất cả hai nhánh và đánh giá đủ VOC val',decision=decision)
    render_report(job,refresh_plot=True)


def worker(job):
    plan=json.loads((job/'plan.json').read_text());(job/'logs').mkdir(exist_ok=True)
    python=sys.executable
    def interrupt(signum,frame):
        if ACTIVE is not None and ACTIVE.poll() is None:os.killpg(ACTIVE.pid,signal.SIGTERM)
        raise KeyboardInterrupt('Supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        from tools.infer_voc_checkpoint_pair import sha256
        train=(Path(plan['list_folder'])/(plan['train_split']+'.txt')).read_text().split()
        val=(Path(plan['list_folder'])/(plan['val_split']+'.txt')).read_text().split()
        assert len(train)==len(set(train))==10582 and len(val)==len(set(val))==1449
        assert not set(train)&set(val)
        checkpoint=Path(plan['teacher_run'])/'checkpoints'/plan['teacher_checkpoint']
        atomic_json(job/'input_provenance.json',{'teacher_sha256':sha256(checkpoint),
                    'train_images':len(train),'val_images':len(val),'split_overlap':0,
                    'teacher':str(checkpoint),'teacher_bytes':checkpoint.stat().st_size})
        run_stage(job,plan,'unit_tests','Chạy kiểm thử trên snapshot mã nguồn',
                  [python,'-m','unittest','discover','-s','tests','-p','test_*.py','-v'],gpu=False)
        run_stage(job,plan,'smoke_cache','Kiểm thử tạo pseudo-label từ teacher',
                  [python,'-u','scripts/cache_decoder_teacher.py','--plan',str(job/'plan.json'),
                   '--output',str(job/'checks/smoke_cache'),'--limit','4'],job/'checks/smoke_cache/progress.json')
        run_stage(job,plan,'smoke_train','Kiểm thử học decoder, neo KL, EMA và đóng băng trọng số',
                  [python,'-u','scripts/train_decoder_selftraining.py','--plan',str(job/'plan.json'),
                   '--cache',str(job/'checks/smoke_cache'),'--mode','crf_teacher',
                   '--output',str(job/'checks/smoke_train'),'--smoke'],job/'checks/smoke_train/progress.json')
        run_stage(job,plan,'smoke_evaluation','Kiểm thử strict loading và inference checkpoint mới',
                  [python,'-u','tools/evaluate_decoder_refinement.py','--plan',str(job/'plan.json'),
                   '--run-dir',str(job/'checks/smoke_train'),'--output',str(job/'checks/smoke_eval'),
                   '--tag','smoke_ema','--kind','ema','--limit','2'],job/'checks/smoke_eval/progress.json')
        import re
        tests=re.search(r'Ran (\d+) tests?',(job/'logs/unit_tests.log').read_text())
        smoke=json.loads((job/'checks/smoke_train/training_summary.json').read_text())
        atomic_json(job/'checks.json',{'unit_tests_passed':int(tests[1]) if tests else True,
                   'smoke_cache_passed':True,'smoke_training_passed':True,'smoke_inference_passed':True,
                   'frozen_weights_unchanged':smoke['frozen_weights_unchanged'],
                   'trainable_decoder_parameters':smoke['trainable_parameters'],
                   'segmentation_forward_parity':json.loads((job/'checks/smoke_cache/segmentation_path_parity.json').read_text())})
        run_stage(job,plan,'teacher_cache','Tạo pseudo-label cho đủ train_aug',
                  [python,'-u','scripts/cache_decoder_teacher.py','--plan',str(job/'plan.json'),
                   '--output',str(job/'teacher_cache')],job/'teacher_cache/progress.json')
        for mode in plan['modes']:
            run_stage(job,plan,'train_'+mode,'Tinh chỉnh decoder nhánh '+mode,
                      [python,'-u','scripts/train_decoder_selftraining.py','--plan',str(job/'plan.json'),
                       '--cache',str(job/'teacher_cache'),'--mode',mode,'--output',str(job/'train'/mode)],
                      job/'train'/mode/'progress.json')
            for kind in ['student','ema']:
                tag=mode+'_'+kind
                run_stage(job,plan,'eval_'+tag,'Đánh giá VOC val: '+tag,
                          [python,'-u','tools/evaluate_decoder_refinement.py','--plan',str(job/'plan.json'),
                           '--run-dir',str(job/'train'/mode),'--output',str(job/'evaluation'),
                           '--tag',tag,'--kind',kind],job/'evaluation/progress.json')
                render_report(job,refresh_plot=True)
        finalize(job,plan)
    except KeyboardInterrupt as error:
        status(job,'stopped','Đã dừng theo yêu cầu',reason=str(error));raise
    except Exception as error:
        status(job,'failed','Pipeline gặp lỗi; xem log, chưa có kết luận cải thiện',error=str(error));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job-dir',type=Path,required=True)
    ap.add_argument('--worker',action='store_true')
    opts=ap.parse_args();job=opts.job_dir.resolve()
    if opts.worker:worker(job);return
    assert (job/'plan.json').exists(),'Create a plan first'
    if (job/'supervisor.pid').exists():raise FileExistsError('Job already launched')
    snapshot=freeze_sources(job)
    render_report(job,refresh_plot=True)
    env=dict(os.environ,PYTHONPATH=str(snapshot)+os.pathsep+json.loads((job/'plan.json').read_text())['deps_folder'])
    with (job/'supervisor.log').open('x') as stream:
        proc=subprocess.Popen([sys.executable,'-u',str(snapshot/'scripts/run_decoder_experiment.py'),
                               '--worker','--job-dir',str(job)],cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (job/'supervisor.pid').write_text(str(proc.pid)+'\n')
    print(json.dumps({'job_dir':str(job),'pid':proc.pid,'report':str(job/'REPORT.md')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
