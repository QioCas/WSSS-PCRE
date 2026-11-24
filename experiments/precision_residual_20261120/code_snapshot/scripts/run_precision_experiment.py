"""Detached sequential ablations with source freeze, label probe and live report."""
import argparse,csv,hashlib,json,os,re,shutil,signal,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_decoder_experiment as runtime
from tools.infer_voc_checkpoint_pair import atomic_json,sha256
from utils.precision_experiment_report import render_report
runtime.render_report=render_report


def finalize(job,p):
    baseline=json.loads(Path(p['reference_baseline_metrics']).read_text());base=baseline['crf']['miou']*100
    results={};checks={}
    for arm in p['arms']:
        name=arm['name'];v=json.loads((job/'evaluation'/name/'metrics.json').read_text())
        assert v['metadata']['validation_images']==1449 and not v['metadata']['partial_validation']
        results[name]=v;raw=v['raw']['miou']*100;crf=v['crf']['miou']*100
        diffs={c:(v['crf']['iou'][str(i)]-baseline['crf']['iou'][str(i)])*100 for i,c in enumerate(baseline['class_names'])}
        parent=base if arm['parent']=='baseline' else checks[arm['parent']]['crf']
        checks[name]={'raw':raw,'crf':crf,'gain_pp':crf-base,'delta_vs_parent_pp':crf-parent,
                      'class_delta_pp':diffs,'worst_class_drop_pp':min(diffs.values()),
                      'gain_target_met':crf-base>=p['target_gain_pp'],
                      'class_guardrail_met':min(diffs.values())>=-p['max_allowed_class_drop_pp'],
                      'external_image_text_prior':arm['external_prior']}
    eligible=[n for n,v in checks.items() if not v['external_image_text_prior'] and v['gain_target_met'] and v['class_guardrail_met']]
    chosen=max(eligible,key=lambda n:checks[n]['crf']) if eligible else 'baseline'
    external=checks['D1_clip_region_filter'];external_ok=external['gain_target_met'] and external['class_guardrail_met']
    explanation=(f'Tuyến không thêm pretraining: {chosen}. '
                 f'CLIP region filter đạt {external["crf"]:.4f}% CRF ({external["gain_pp"]:+.4f} điểm), '
                 f'{"đạt" if external_ok else "chưa đạt"} tiêu chí quan sát. '
                 'Đây là một seed trên development val với ngân sách fine-tuning thêm; chưa kết luận cải thiện ổn định.')
    decision={'chosen_without_external_pretraining':chosen,'clip_track_qualifies':external_ok,'checks':checks,
              'explanation':explanation,'qualification':p['qualification']}
    atomic_json(job/'decision.json',decision)
    with (job/'per_class_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);modes=list(results)
        writer.writerow(['class','baseline_crf_iou_percent']+[n+'_crf_iou_percent' for n in modes])
        for i,c in enumerate(baseline['class_names']):
            writer.writerow([c,baseline['crf']['iou'][str(i)]*100]+[results[n]['crf']['iou'][str(i)]*100 for n in modes])
    runtime.status(job,'completed','Hoàn tất sáu nhánh, mỗi nhánh đủ VOC val',decision=decision)
    render_report(job,refresh_plot=True)


def worker(job):
    p=json.loads((job/'plan.json').read_text());planfile=str(job/'plan.json');(job/'logs').mkdir(exist_ok=True)
    def interrupt(signum,frame):
        if runtime.ACTIVE is not None and runtime.ACTIVE.poll() is None:os.killpg(runtime.ACTIVE.pid,signal.SIGTERM)
        raise KeyboardInterrupt('Supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def stage(name,description,args,progress=None,gpu=True):
        runtime.run_stage(job,p,name,description,[sys.executable,'-u',*args],progress,gpu)
    def bank(out,limit=0):
        args=['scripts/build_precision_bank.py','--plan',planfile,'--output',str(out)]
        return args+(['--limit',str(limit)] if limit else [])
    def cache(kind,out,bankpath,limit=0):
        args=['scripts/cache_precision_targets.py','--plan',planfile,'--output',str(out),'--bank',str(bankpath),'--kind',kind]
        return args+(['--limit',str(limit)] if limit else [])
    def train(mode,cachepath,out,smoke=False):
        args=['scripts/train_precision_residual.py','--plan',planfile,'--cache',str(cachepath),'--mode',mode,'--output',str(out)]
        return args+(['--smoke'] if smoke else [])
    def evaluate(mode,out,limit=0):
        args=['tools/evaluate_precision_residual.py','--plan',planfile,'--run-dir',str(out),
              '--output',str(job/('checks/evaluation' if limit else 'evaluation')),'--tag',mode]
        return args+(['--limit',str(limit)] if limit else [])
    try:
        names=(Path(p['list_folder'])/'train_aug.txt').read_text().split();val=(Path(p['list_folder'])/'val.txt').read_text().split()
        assert len(set(names))==len(names)==10582 and len(set(val))==len(val)==1449 and not set(names)&set(val)
        source=Path(p['reliable_cache']);manifest=json.loads((source/'manifest.json').read_text())
        assert not manifest['partial_cache'] and manifest['names']==names
        teacher_hash=sha256(Path(p['teacher_run'])/'checkpoints'/p['teacher_checkpoint'])
        assert teacher_hash==manifest['teacher_sha256']
        atomic_json(job/'input_provenance.json',{'train_images':10582,'val_images':1449,'split_overlap':0,
            'teacher_sha256':teacher_hash,'reliable_manifest_sha256':sha256(source/'manifest.json'),
            'clip_sha256':sha256(Path(p['clip_download_dir'])/'ViT-B-16.pt'),
            'training_pixel_gt_read':False,'probe_reads_val_gt_only':True})
        stage('unit_tests','Kiểm thử snapshot mã nguồn',['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],gpu=False)
        smallbank=job/'checks/bank'
        stage('smoke_bank','Smoke bank trên 32 ảnh train',bank(smallbank,32),smallbank/'progress.json')
        for kind in ['negative','density','clip']:
            out=job/'checks'/kind
            stage('smoke_targets_'+kind,'Smoke nhãn '+kind,cache(kind,out,smallbank,4),out/'progress.json')
        # Create an exact four-image subset of the existing immutable reliable cache.
        reliable=job/'checks/reliable';(reliable/'labels').mkdir(parents=True)
        partial=dict(manifest,names=names[:4],partial_cache=True)
        atomic_json(reliable/'manifest.json',partial)
        for name in names[:4]:os.symlink(source/'labels'/f'{name}.png',reliable/'labels'/f'{name}.png')
        for mode,cachepath in [('A1_residual_control',reliable),('A3_negative_learning',job/'checks/negative'),
                               ('B1_distribution_filter',job/'checks/density'),('C1_detail_residual',reliable),
                               ('D1_clip_region_filter',job/'checks/clip')]:
            out=job/'checks'/mode
            stage('smoke_train_'+mode,'Smoke train '+mode,train(mode,cachepath,out,True),out/'progress.json')
        for mode in ['A3_negative_learning','C1_detail_residual']:
            stage('smoke_eval_'+mode,'Smoke strict reload và multiscale '+mode,
                  evaluate(mode,job/'checks'/mode,2),job/'checks/evaluation/progress.json')
        stage('smoke_probe','Smoke chẩn đoán nhãn trên val',['tools/probe_precision_targets.py','--plan',planfile,
              '--bank',str(smallbank),'--output',str(job/'checks/probe'),'--limit','2'],job/'checks/probe/progress.json')
        tests=re.search(r'Ran (\d+) tests?',(job/'logs/unit_tests.log').read_text())
        trained=json.loads((job/'checks/A3_negative_learning/training_summary.json').read_text())
        atomic_json(job/'checks.json',{'unit_tests_passed':int(tests[1]) if tests else True,
            'bank_negative_density_clip_smokes':True,'all_five_training_variants_passed':True,
            'residual_and_detail_strict_reload_multiscale_passed':True,'label_quality_probe_passed':True,
            'zero_init_parity':trained['initial_parity'],'frozen_baseline_unchanged':trained['frozen_weights_unchanged']})
        fullbank=job/'bank'
        stage('bank','Tạo bank positive/negative và GMM từ toàn bộ train_aug',bank(fullbank),fullbank/'progress.json')
        stage('label_probe','Chẩn đoán precision/recall nhãn trên 128 ảnh development val',
              ['tools/probe_precision_targets.py','--plan',planfile,'--bank',str(fullbank),'--output',str(job/'probe')],job/'probe/progress.json')
        created={}
        for arm in p['arms']:
            mode=arm['name'];kind=arm['target'];out=job/'train'/mode
            if kind=='reliable':cachepath=source
            else:
                cachepath=job/'cache'/kind
                if kind not in created:
                    stage('targets_'+kind,'Tạo nhãn train '+kind,cache(kind,cachepath,fullbank),cachepath/'progress.json');created[kind]=True
            stage('train_'+mode,'Train '+mode,train(mode,cachepath,out),out/'progress.json')
            stage('eval_'+mode,'Đánh giá đủ 1.449 VOC val: '+mode,evaluate(mode,out),job/'evaluation/progress.json')
            render_report(job,refresh_plot=True)
        finalize(job,p)
    except KeyboardInterrupt as error:
        runtime.status(job,'stopped','Pipeline đã dừng',reason=str(error));raise
    except Exception as error:
        runtime.status(job,'failed','Pipeline gặp lỗi; chưa kết luận cải thiện',error=str(error));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--job-dir',type=Path,required=True)
    ap.add_argument('--worker',action='store_true');opts=ap.parse_args();job=opts.job_dir.resolve()
    if opts.worker:worker(job);return
    if (job/'supervisor.pid').exists():raise FileExistsError('Job already launched')
    original=json.loads((job/'plan.json').read_text());snapshot=runtime.freeze_sources(job)
    target=snapshot/'external/clip';shutil.copytree(Path(original['clip_module_dir'])/'clip',target,
        ignore=shutil.ignore_patterns('__pycache__'))
    hashes={str(f.relative_to(snapshot)):sha256(f) for f in target.rglob('*') if f.is_file()}
    atomic_json(job/'clip_source_hashes.json',hashes)
    p=json.loads((job/'plan.json').read_text());p['original_clip_module_dir']=p['clip_module_dir']
    p['clip_module_dir']=str(snapshot/'external');atomic_json(job/'plan.json',p)
    render_report(job,refresh_plot=True)
    env=dict(os.environ,PYTHONPATH=str(snapshot)+os.pathsep+p['deps_folder'],MPLBACKEND='Agg')
    with (job/'supervisor.log').open('x') as stream:
        proc=subprocess.Popen([sys.executable,'-u',str(snapshot/'scripts/run_precision_experiment.py'),
            '--worker','--job-dir',str(job)],cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (job/'supervisor.pid').write_text(str(proc.pid)+'\n')
    print(json.dumps({'job_dir':str(job),'pid':proc.pid,'report':str(job/'REPORT.md')},ensure_ascii=False))


if __name__=='__main__':main()
