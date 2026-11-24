"""Source-frozen, detached matched representation/consistency/SAM experiments."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_decoder_experiment as runtime
from tools.infer_voc_checkpoint_pair import atomic_json,sha256
from utils.representation_experiment_report import render_report
runtime.render_report=render_report


def finalize(job,plan):
    baseline=json.loads(Path(plan['reference_baseline_metrics']).read_text())
    results={};checks={};base=baseline['crf']['miou']*100
    for arm in plan['arms']:
        name=arm['name'];v=json.loads((job/'evaluation'/name/'metrics.json').read_text())
        assert v['metadata']['validation_images']==1449 and not v['metadata']['partial_validation']
        results[name]=v
        diffs={c:(v['crf']['iou'][str(i)]-baseline['crf']['iou'][str(i)])*100
               for i,c in enumerate(baseline['class_names'])}
        gain=v['crf']['miou']*100-base
        checks[name]={'raw':v['raw']['miou']*100,'crf':v['crf']['miou']*100,'gain_pp':gain,
                      'class_delta_pp':diffs,'worst_class_drop_pp':min(diffs.values()),
                      'gain_target_met':gain>=plan['target_gain_pp'],
                      'class_guardrail_met':min(diffs.values())>=-plan['max_allowed_class_drop_pp'],
                      'external_segmentation_prior':arm['external_segmentation_prior']}
    eligible=[n for n,c in checks.items() if not c['external_segmentation_prior'] and c['gain_target_met'] and c['class_guardrail_met']]
    chosen=max(eligible,key=lambda n:checks[n]['crf']) if eligible else 'baseline'
    sam=checks['last2_sam'];sam_ok=sam['gain_target_met'] and sam['class_guardrail_met']
    explanation=(f'Tuyến không thêm mask pretraining: chọn {chosen}. '
                 f'last2_consistency so với decoder_control: {checks["last2_consistency"]["crf"]-checks["decoder_control"]["crf"]:+.4f} điểm. '
                 f'Tuyến có SAM: {sam["crf"]:.4f}% CRF ({sam["gain_pp"]:+.4f} điểm với baseline), '
                 f'{"đạt" if sam_ok else "chưa đạt"} tiêu chí định trước. '
                 'SAM dùng thêm giám sát mask từ pretraining; tất cả là kết quả một seed trên development val.')
    decision={'chosen_without_external_masks':chosen,'sam_track_qualifies':sam_ok,'checks':checks,
              'last2_minus_decoder_control_pp':checks['last2_consistency']['crf']-checks['decoder_control']['crf'],
              'sam_minus_last2_control_pp':sam['crf']-checks['last2_consistency']['crf'],
              'qualification':plan['qualification'],'explanation':explanation}
    atomic_json(job/'decision.json',decision)
    with (job/'per_class_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);names=list(results)
        writer.writerow(['class','baseline_crf_iou_percent']+[n+'_crf_iou_percent' for n in names])
        for i,c in enumerate(baseline['class_names']):
            writer.writerow([c,baseline['crf']['iou'][str(i)]*100]+[results[n]['crf']['iou'][str(i)]*100 for n in names])
    runtime.status(job,'completed','Hoàn tất ba nhánh và đánh giá đủ VOC val',decision=decision)
    render_report(job,refresh_plot=True)


def worker(job):
    plan=json.loads((job/'plan.json').read_text());(job/'logs').mkdir(exist_ok=True)
    python=sys.executable;planfile=str(job/'plan.json')
    def interrupt(signum,frame):
        if runtime.ACTIVE is not None and runtime.ACTIVE.poll() is None:
            os.killpg(runtime.ACTIVE.pid,signal.SIGTERM)
        raise KeyboardInterrupt('Supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def stage(name,description,command,progress=None,gpu=True):
        runtime.run_stage(job,plan,name,description,[python,'-u',*command],progress,gpu)
    def cache(kind,out,limit=0,reliable=None):
        args=['scripts/cache_representation_targets.py','--plan',planfile,'--output',str(out),'--kind',kind]
        if limit:args+=['--limit',str(limit)]
        if reliable:args+=['--reliable',str(reliable)]
        return args
    def train(mode,cachepath,out,smoke=False):
        args=['scripts/train_representation_consistency.py','--plan',planfile,'--cache',str(cachepath),
              '--mode',mode,'--output',str(out)]
        return args+(['--smoke'] if smoke else [])
    def evaluate(mode,out,tag,limit=0):
        args=['tools/evaluate_decoder_refinement.py','--plan',planfile,'--run-dir',str(out),
              '--output',str(job/('checks/evaluation' if limit else 'evaluation')),'--tag',tag,'--kind','ema']
        return args+(['--limit',str(limit)] if limit else [])
    try:
        train_ids=(Path(plan['list_folder'])/'train_aug.txt').read_text().split()
        val_ids=(Path(plan['list_folder'])/'val.txt').read_text().split()
        assert len(set(train_ids))==len(train_ids)==10582 and len(set(val_ids))==len(val_ids)==1449
        assert not set(train_ids)&set(val_ids)
        checkpoint=Path(plan['teacher_run'])/'checkpoints'/plan['teacher_checkpoint']
        atomic_json(job/'input_provenance.json',{'train_images':10582,'val_images':1449,'split_overlap':0,
            'teacher_sha256':sha256(checkpoint),'sam_sha256':sha256(plan['sam_checkpoint']),
            'training_pixel_gt_read':False,'sam_external_mask_pretraining':True})
        stage('unit_tests','Kiểm thử snapshot mã nguồn',['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],gpu=False)
        reliable=job/'checks/reliable';sam=job/'checks/sam'
        stage('smoke_reliable','Smoke test nhãn đa view',cache('reliable',reliable,4),reliable/'progress.json')
        stage('smoke_sam','Smoke test nhãn SAM',cache('sam',sam,4,reliable),sam/'progress.json')
        for mode,source in [('decoder_control',reliable),('last2_sam',sam)]:
            out=job/'checks'/mode
            stage('smoke_train_'+mode,'Smoke train '+mode,train(mode,source,out,True),out/'progress.json')
        stage('smoke_evaluation','Smoke strict load và inference checkpoint mới',
              evaluate('last2_sam',job/'checks/last2_sam','smoke_ema',2),job/'checks/evaluation/progress.json')
        tests=re.search(r'Ran (\d+) tests?',(job/'logs/unit_tests.log').read_text())
        partial=json.loads((job/'checks/last2_sam/training_summary.json').read_text())
        atomic_json(job/'checks.json',{'unit_tests_passed':int(tests[1]) if tests else True,
            'reliable_cache_passed':True,'sam_cache_passed':True,'decoder_training_passed':True,
            'partial_backbone_training_passed':True,'strict_checkpoint_inference_passed':True,
            'frozen_student_unchanged':partial['frozen_weights_unchanged'],'teacher_unchanged':partial['teacher_unchanged'],
            'last2_trainable_parameters':partial['trainable_parameters']})
        reliable=job/'cache/reliable';sam=job/'cache/sam'
        stage('reliable_cache','Tạo nhãn reliable cho toàn bộ train_aug',cache('reliable',reliable),reliable/'progress.json')
        for arm in plan['arms']:
            mode=arm['name'];out=job/'train'/mode
            if arm['cache']=='sam':stage('sam_cache','SAM sửa nhãn toàn bộ train_aug',cache('sam',sam,reliable=reliable),sam/'progress.json')
            source=sam if arm['cache']=='sam' else reliable
            stage('train_'+mode,'Fine-tuning '+mode,train(mode,source,out),out/'progress.json')
            stage('eval_'+mode,'Đánh giá đủ VOC val: '+mode,evaluate(mode,out,mode),job/'evaluation/progress.json')
            render_report(job,refresh_plot=True)
        finalize(job,plan)
    except KeyboardInterrupt as error:
        runtime.status(job,'stopped','Pipeline đã dừng',reason=str(error));raise
    except Exception as error:
        runtime.status(job,'failed','Pipeline gặp lỗi, chưa kết luận cải thiện',error=str(error));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--job-dir',type=Path,required=True)
    ap.add_argument('--worker',action='store_true');opts=ap.parse_args();job=opts.job_dir.resolve()
    if opts.worker:worker(job);return
    if (job/'supervisor.pid').exists():raise FileExistsError('Job already launched')
    plan=json.loads((job/'plan.json').read_text())
    snapshot=runtime.freeze_sources(job)
    sam_target=snapshot/'external/segment_anything'
    shutil.copytree(Path(plan['sam_module_dir'])/'segment_anything',sam_target,ignore=shutil.ignore_patterns('__pycache__'))
    hashes={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sam_target.rglob('*.py')}
    atomic_json(job/'sam_source_hashes.json',hashes)
    plan=json.loads((job/'plan.json').read_text());plan['original_sam_module_dir']=plan['sam_module_dir']
    plan['sam_module_dir']=str(snapshot/'external');atomic_json(job/'plan.json',plan)
    render_report(job,refresh_plot=True)
    env=dict(os.environ,PYTHONPATH=str(snapshot)+os.pathsep+plan['deps_folder'],MPLBACKEND='Agg')
    with (job/'supervisor.log').open('x') as stream:
        proc=subprocess.Popen([sys.executable,'-u',str(snapshot/'scripts/run_representation_experiment.py'),
            '--worker','--job-dir',str(job)],cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (job/'supervisor.pid').write_text(str(proc.pid)+'\n')
    print(json.dumps({'job_dir':str(job),'pid':proc.pid,'report':str(job/'REPORT.md')},ensure_ascii=False))


if __name__=='__main__':main()
