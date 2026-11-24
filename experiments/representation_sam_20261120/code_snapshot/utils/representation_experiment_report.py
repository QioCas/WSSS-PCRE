"""Evidence-backed Vietnamese report for the matched representation/SAM trials."""
import json
import os
from pathlib import Path
from datetime import datetime,timezone
from utils.decoder_experiment_report import plot_comparison


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def observed(job,plan):
    rows=[]
    for arm in plan['arms']:
        name=arm['name'];value=read(job/'evaluation'/name/'metrics.json')
        if value:
            assert value['metadata']['validation_images']==1449 and not value['metadata']['partial_validation']
            rows.append((name,value['raw']['miou']*100,value['crf']['miou']*100))
    return rows


def render_report(job,refresh_plot=False):
    job=Path(job);plan=read(job/'plan.json');base=read(plan['reference_baseline_metrics'])
    baseline=base['crf']['miou']*100
    state=read(job/'status.json') or {'description':'Đang chuẩn bị kiểm thử'}
    failure=read(job/'failure_analysis.json');decision=read(job/'decision.json')
    historical=[('Baseline 20k',base['raw']['miou']*100,baseline)]
    paths={
      'Adaptive cũ':job.parent/'inference_voc_val_20261119/full/adaptive/metrics.json',
      'Consensus repair':job.parent/'pcre_repair_20261119/single_model_v2/inference/adaptive_consensus/metrics.json',
      'Ensemble hai model':job.parent/'pcre_repair_20261119/ensemble_full/metrics.json',
      'Decoder raw EMA':Path(plan['old_experiment'])/'evaluation/raw_teacher_ema/metrics.json',
      'Decoder CRF EMA':Path(plan['old_experiment'])/'evaluation/crf_teacher_ema/metrics.json'}
    for name,path in paths.items():
        v=read(path)
        if v:historical.append((name,v['raw']['miou']*100,v['crf']['miou']*100))
    new=observed(job,plan);measured={r[0]:r for r in new}
    finding=decision['explanation'] if decision else 'Chưa có kết luận rằng phương pháp mới vượt baseline.'
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines=['# PCRE: kiểm chứng biểu diễn, consistency và pseudo-label từ SAM','',
      f'Cập nhật {now}. VOC 2012; seed {plan["seed"]}.','',
      f'Baseline cố định: **{baseline:.4f}% mIoU sau CRF** trên đủ 1.449 ảnh val. **Trạng thái:** {state["description"]}. {finding}','',
      '## Vì sao các vòng trước chưa thắng?','',
      '| Phương án đã đo | mIoU raw (%) | mIoU CRF (%) | Δ CRF (điểm) |',
      '|---|---:|---:|---:|']
    for name,raw,crf in historical:lines.append(f'| {name} | {raw:.4f} | {crf:.4f} | {crf-baseline:+.4f} |')
    lines += ['',
      'Adaptive và consensus thay đổi pseudo-label/đường học nên vẫn có thể làm giảm một số lớp dù loss hội tụ. Ensemble từng tăng khoảng 0,097 điểm với hai model; mức tăng đó chưa phải bằng chứng một phương pháp đơn model tốt hơn. Các vòng fine-tuning decoder tiếp theo vẫn thấp hơn baseline khoảng 0,029 điểm. [Báo cáo trước](../decoder_selftrain_20261119/REPORT.md).','',
      f'Đo lại cache trên toàn bộ train_aug: chỉ **{failure["semantic_disagreement_fraction_given_both_labeled"]*100:.4f}%** pixel mà cả raw và CRF cùng gán nhãn có nhãn semantic khác nhau. Raw bỏ qua {failure["old_cache_raw_ignore_fraction"]*100:.3f}%, CRF bỏ qua {failure["old_cache_crf_ignore_fraction"]*100:.3f}% pixel. Hai nhánh vì thế nhận tín hiệu gần giống nhau, và ngưỡng cũ giữ gần như toàn bộ nhãn của teacher. [Số đo](failure_analysis.json).','',
      f'Decoder vẫn thay đổi trọng số (relative L2 {failure["decoder_relative_l2_change"]*100:.2f}%), nên thất bại không phải vì optimizer đứng yên. Tuy nhiên backbone đóng băng giới hạn biểu diễn; KL cũ dùng cùng view/features nên không buộc học tính bất biến giữa view. Đây là các giả thuyết hạn chế, chưa phải kết luận nhân quả. Không suy ra độ chính xác pseudo-label chỉ từ confidence hoặc độ phủ.','',
      '## Thiết kế thí nghiệm mới','',
      '| Nhánh | Trọng số được học | Pseudo-label | Mục đích |',
      '|---|---|---|---|',
      '| decoder_control | ASPP decoder | Reliable teacher | Đối chứng cho thay đổi augmentation/loss/nhãn |',
      '| last2_consistency | Decoder + 2 block ViT cuối + final norm | Cùng reliable teacher | Kiểm tra việc mở biểu diễn |',
      '| last2_sam | Giống last2_consistency | Reliable teacher + SAM | Kiểm tra tín hiệu hình dạng ngoài teacher |','',
      '**Hai tuyến kết quả:** hai nhánh đầu giữ nguồn giám sát của baseline, nhưng thêm ngân sách fine-tuning. Nhánh SAM sử dụng mask-supervised pretraining bên ngoài (SAM/SA-1B), phải báo cáo riêng; không được gọi là cải thiện WSSS cùng dữ liệu/giám sát. SAM không nhận VOC pixel ground truth từ pipeline này.','',
      '1. Khởi tạo tất cả student và teacher từ cùng baseline 20k. Teacher luôn đóng băng. Tạo cache cho 10.582 ảnh train_aug, kiểm tra không giao với 1.449 ảnh val.',
      '2. Teacher dự đoán 3 scales × 2 flip; softmax từng view trước khi lấy trung bình, tránh tăng độ sắc do cộng logits. Giới hạn lớp bằng nhãn cấp ảnh chỉ khi tạo nhãn train. Giữ pixel có mean confidence ≥ 0,8 và ít nhất 5/6 view đồng ý; còn lại ignore=255.',
      '3. Teacher online nhìn weak view; student nhìn strong view cùng resize/crop/flip. Strong dùng jitter 0,4, grayscale xác suất 0,1, blur 0,2, cutout 0,5 với diện tích 0,1. Loại cutout và padding khỏi cả CE/KL.',
      '4. Loss = CE cân bằng foreground/background + 0,25 × T² KL(teacher weak || student strong), T=2. KL chỉ dùng teacher confidence ≥ 0,7; bỏ pixel mà nhãn cache đã gán khác teacher để không neo ngược sửa đổi SAM.',
      '5. Mỗi nhánh 4.000 bước bổ sung, batch 4, crop 448, AdamW weight decay 0,01; LR decoder 2e-5, backbone 2e-6; warm-up 100, cosine; EMA 0,99. Chỉ checkpoint EMA cuối lịch được đánh giá chính.',
      '6. SAM ViT-B nhận positive point trong lõi teacher, box quanh connected component, có thể thêm negative point từ lớp khác. Chỉ giữ mask có predicted quality ≥ 0,85, teacher-mask IoU ≥ 0,5, reliable-core recall ≥ 0,9, tỷ lệ diện tích trong [0,5; 2]. Các số này là kiểm tra tương hợp teacher, không phải IoU với ground truth.',
      '7. Giới hạn sửa trong vùng 12 pixel quanh teacher component; lõi SAM erode 3 pixel làm nhãn, biên bất định thành ignore. Bảo vệ lõi reliable, cấm chuyển một foreground đáng tin cậy sang lớp foreground khác; xung đột proposal thành ignore.','',
      'Mở 2 block thay vì toàn backbone nhằm cho phép sửa đặc trưng với LR nhỏ. Đối chứng decoder và last2 dùng cùng ảnh, nhãn, augmentation, loss và số bước; so sánh last2 với SAM chỉ thay nguồn nhãn. ST++ và Noisy Student là cảm hứng về reliability/weak–strong self-training, không phải bản tái triển khai hay bằng chứng phương pháp này sẽ thắng.','',
      '## Đánh giá và tiêu chí định trước','',
      'Giữ nguyên baseline inference: float32, TF32 off, scales 1/1,25/1,5 + flip, fusion gốc, không dùng nhãn ảnh để lọc lớp val. DenseCRF: 10 vòng, pos_xy_std=2, pos_w=3, bi_xy_std=140, bi_rgb_std=1, bi_w=7. Đánh giá đủ 1.449 ảnh, giữ cả raw và CRF, IoU từng lớp và confusion matrix. Không chọn checkpoint giữa lịch theo val.','',
      f'Mục tiêu thực dụng: tăng ≥ {plan["target_gain_pp"]} điểm mIoU CRF và không lớp nào giảm quá {plan["max_allowed_class_drop_pp"]} điểm IoU. Chỉ so chọn giữa hai nhánh không SAM; nhánh SAM có quyết định riêng. Chưa đạt thì giữ baseline, giữ nguyên kết quả âm. Một seed, thêm 4k bước và val đã dùng phát triển phương pháp không đủ chứng minh tổng quát hóa hay công bằng tổng ngân sách.','',
      '| Nhánh EMA cuối lịch | Raw (%) | CRF (%) | Δ CRF (điểm) | Giám sát ngoài |',
      '|---|---:|---:|---:|---|']
    for arm in plan['arms']:
        name=arm['name'];external='SAM/SA-1B' if arm['external_segmentation_prior'] else 'không thêm'
        if name in measured:
            _,raw,crf=measured[name];lines.append(f'| {name} | {raw:.4f} | {crf:.4f} | {crf-baseline:+.4f} | {external} |')
        else:lines.append(f'| {name} | chưa có | chưa có | chưa có | {external} |')
    if decision:lines += ['','[Quyết định và đối chứng](decision.json) · [IoU từng lớp](per_class_results.csv)']
    lines += ['','## Kiểm tra triển khai và trạng thái chi tiết','',
      'Kiểm thử reliability, gradient qua block được mở, weak/strong geometry, padding/cutout, KL mask, lọc SAM, bảo vệ lớp và xung đột proposal. Smoke test GPU tạo nhãn → train → strict-load checkpoint → inference. So hash toàn bộ teacher và phần student đóng băng trước/sau train. Training dataset chỉ đọc JPEG, pseudo-label và nhãn cấp ảnh. Mã nguồn và danh sách ảnh được chụp vào code_snapshot; ghi SHA256 checkpoint và SAM.','']
    checks=read(job/'checks.json')
    if checks:lines += ['Kết quả kiểm tra: `'+json.dumps(checks,ensure_ascii=False)+'`','']
    for kind in ['reliable','sam']:
        cache=read(job/'cache'/kind/'summary.json')
        progress=read(job/'cache'/kind/'progress.json')
        if cache:
            lines += [f'- Cache {kind}: {cache["images"]:,} ảnh; foreground {cache["foreground"]*100:.2f}%, background {cache["background"]*100:.2f}%, ignore {cache["ignore"]*100:.2f}%.']
            if kind=='sam':lines += [f'  SAM chấp nhận {cache["accepted_proposals"]:,}/{cache["attempted_proposals"]:,} proposal, sửa {cache["changed_fraction"]*100:.3f}% pixel; đây là độ thay đổi, không phải độ chính xác.']
        elif progress:lines += [f'- Cache {kind}: {progress["completed"]:,}/{progress["total"]:,} ảnh.']
    for arm in plan['arms']:
        p=read(job/'train'/arm['name']/'progress.json')
        if p:lines += [f'- Train {arm["name"]}: {p.get("step",p.get("completed_steps"))}/{plan["refinement_steps"]} bước.']
    if refresh_plot:plot_comparison(job,historical+new)
    if (job/'figures/miou_delta.png').exists():lines += ['','![Kết quả đã đo, không điền dự đoán cho nhánh chưa chạy](figures/miou_delta.png)']
    lines += ['','## Tệp và tài liệu tham khảo','',
      f'- [Baseline metrics]({os.path.relpath(plan["reference_baseline_metrics"],job)}) · [Kế hoạch](plan.json) · [Phân tích thất bại](failure_analysis.json).',
      '- [Trạng thái](status.json) · [Sự kiện](events.jsonl) · logs/ · train/ · evaluation/.',
      '- [ST++, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_ST_Make_Self-Training_Work_Better_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.html).',
      '- [Noisy Student](https://arxiv.org/abs/1911.04252): self-training với nhiễu, nghiên cứu classification.',
      '- [SAM chính thức](https://github.com/facebookresearch/segment-anything) và [predictor API](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py). Checkpoint pretrained trên SA-1B, khoảng 11 triệu ảnh và 1,1 tỷ mask theo tác giả.',
      '- [Kiểm tra tài nguyên SAM](sam_asset_checks.json). Các giá trị SAM predicted quality không phải IoU đo trên VOC GT.','']
    temporary=job/'REPORT.md.tmp';temporary.write_text('\n'.join(lines));temporary.replace(job/'REPORT.md')
