"""Live evidence-backed report for sequential precision-preserving experiments."""
import json,os
from pathlib import Path
from datetime import datetime,timezone
from utils.decoder_experiment_report import plot_comparison


def read(path):
    path=Path(path);return json.loads(path.read_text()) if path.exists() else None


def render_report(job,refresh_plot=False):
    job=Path(job);p=read(job/'plan.json');baseline=read(p['reference_baseline_metrics'])
    state=read(job/'status.json') or {'description':'Đang chuẩn bị và kiểm thử'}
    base=baseline['crf']['miou']*100;decision=read(job/'decision.json')
    finding=decision['explanation'] if decision else 'Chưa có kết luận vượt baseline.'
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines=['# PCRE: residual, negative prototypes và kiểm chứng precision','',
      f'Cập nhật {now}. **Trạng thái:** {state["description"]}. {finding}','',
      f'Baseline bất biến: **{base:.4f}% mIoU CRF**, 1.449 ảnh VOC val. Mọi nhánh bắt đầu từ cùng checkpoint baseline 20k; thêm 4.000 bước residual, batch 4, seed 0.','',
      '## Câu hỏi và thứ tự thí nghiệm','',
      'Vòng trước tăng recall nhưng giảm precision ở một số lớp; self-training làm sai thêm các vùng baseline đã đúng. Vòng này kiểm tra bằng chứng âm và giới hạn phạm vi sửa. Không tuyên bố đây là bản tái triển khai đầy đủ FPR/U2PL/BRNF/FFR/WeCLIP.','',
      '| Nhánh | Thay đổi | Đối chứng trực tiếp | Giám sát bổ sung |','|---|---|---|---|',
      '| A1_residual_control | Head nhỏ, zero-init, baseline đóng băng | Baseline | Không |',
      '| A2_negative_filter | Bỏ foreground mâu thuẫn với negative bank khỏi CE/KL | A1 | Không |',
      '| A3_negative_learning | Thêm loss phủ định cho lớp bị veto | A2 | Không |',
      '| B1_distribution_filter | GMM kiểm tra phân bố feature và loại outlier | A1 | Không |',
      '| C1_detail_residual | Thêm feature block 4 và thành phần high-pass vào residual | A1 | Không |',
      '| D1_clip_region_filter | CLIP kiểm tra nhãn connected component | A1 | Image–text pretraining |','',
      'Các nhánh chạy tuần tự và độc lập, không lấy checkpoint nhánh trước làm khởi tạo nhánh sau. “Thêm” trong A2/A3 là thêm thành phần phương pháp với cùng khởi tạo/ngân sách, không cộng dồn số bước train.','',
      '## Phương pháp và tham số chốt trước khi đo','',
      '`logits_new = logits_baseline + gate * limit * tanh(residual / limit)`','',
      f'Toàn bộ baseline đóng băng. Head 1×1→GELU→3×3→GELU→1×1, width 64; lớp cuối khởi tạo 0, limit={p["residual_limit"]} logits. Gate giống nhau ở mọi nhánh, chỉ dùng prediction không lọc tags: cho phép sửa foreground dự đoán và background có confidence <0,95. Background tự tin được giữ logits trên lưới feature; interpolation/TTA/CRF vẫn có thể ảnh hưởng mask cuối. Zero-init chỉ bảo đảm trùng đầu ra trước khi học, không bảo đảm không hồi quy sau train.','',
      'Loss gồm CE cân bằng foreground/background + KL giữ posterior baseline, weight 1; chỉ áp KL khi cache không ignore, có support thật và argmax baseline đồng ý nhãn cache. A3 thêm `-log(1-p(rejected_class))`, weight 0,1; không suy ra nhãn dương từ một veto. Không đưa pixel ignore trở lại qua argmax/KL.','',
      'AdamW LR 1e-4, weight decay 0,01; warm-up 100 bước, cosine; EMA 0,99. Resize short side [336,672], crop 448, flip, jitter 0,1. Không strong/cutout trong vòng này. Chỉ EMA cuối lịch dùng cho kết quả chính.','',
      '### Nguồn nhãn và bank','',
      'Tái dùng reliable cache của baseline trên đủ 10.582 ảnh train_aug: mean probability của 6 views, confidence ≥0,8 và 5/6 votes. Bank chỉ đọc ảnh train, cache và image tags. Positive features lấy từ lõi erode 1 patch, tối đa 16 patch/lớp/ảnh, reservoir 2.048/lớp; KMeans 4 positive prototypes.','',
      'Negative features lấy ở ảnh không có lớp c, từ raw CAM c trước khi lọc tags, đồng thời yêu cầu decoder baseline tin background ≥0,8. Điều kiện background nhằm giảm negatives từ bộ phận foreground dùng chung. Giữ 512 ứng viên/lớp, tối đa 2/ảnh; chọn 8 hướng hard negative. Veto khi cosine với negative vượt positive hơn 0,08. Chỉ chuyển foreground tương ứng sang ignore; không đổi lớp hoặc tạo background mới. Đây là FPR-inspired, không phải nguyên bản FPR.','',
      'B1 dùng random orthogonal projection 32 chiều, L2-normalized, diagonal GMM 2 thành phần/lớp, variance regularizer 0,005; cutoff theo quantile 2% thấp nhất của positive train samples. GMM có thể kế thừa bias seed. Đây là thử nghiệm density gate nhỏ, không phải normalizing-flow BRNF.','',
      'C1 lấy high-pass của feature block 4 bằng `early - avg_pool_3x3(early)`, nối với final features vào head. Không có boundary GT, không gọi đây là full FFR hay tái tạo được chi tiết dưới kích thước patch.','',
      'D1 dùng frozen OpenAI CLIP ViT-B/16, prompt tên 20 lớp VOC và background, so cosine trên crop của tối đa 2 connected component/lớp/ảnh; bỏ component nếu background/lớp khác có mặt thắng hơn 0,05. Đây là semantic region verifier lấy cảm hứng từ WeCLIP, không phải dense WeCLIP. CLIP chỉ dùng khi tạo nhãn train/diagnostic, không thêm model CLIP lúc inference student.','',
      '## Kiểm tra và chẩn đoán nhãn trước full training','',
      'Kiểm tra logits zero-init, đường forward so với segmentation baseline, phần đóng băng và gradient, gate, KL loại ignore/padding/conflict, loss phủ định, negative/density support, hình học RGB/nhãn đồng bộ và strict checkpoint reload. Mã thực thi và danh sách ảnh được freeze vào code_snapshot.','']
    checks=read(job/'checks.json')
    if checks:lines += ['Kết quả smoke/unit tests: `'+json.dumps(checks,ensure_ascii=False)+'`','']
    bank=read(job/'bank/manifest.json')
    if bank:lines += [f'Bank hoàn tất: {bank["images"]:,} ảnh; training pixel GT read = {bank["training_pixel_gt_read"]}.','']
    probe=read(job/'probe/summary.json')
    if probe:
        lines += [f'Chẩn đoán trên {probe["metadata"]["images"]} ảnh val chọn cố định bằng seed 0; dùng image tags để mô phỏng pseudo-label train. GT chỉ được đọc sau khi tạo nhãn để chấm diagnostic. Không dùng masks/GT diagnostic để train, không fit bank trên val và không đổi ngưỡng theo diagnostic. Đây không phải mIoU model inference trên toàn val.','',
                  '| Pseudo-label | Foreground micro precision (%) | Foreground micro recall (%) | Pixel đúng bị bỏ | Pixel sai bị bỏ |',
                  '|---|---:|---:|---:|---:|']
        for kind,v in probe['results'].items():
            lines.append(f'| {kind} | {v["foreground_micro_precision"]*100:.3f} | {v["foreground_micro_recall"]*100:.3f} | {v.get("removed_correct","—")} | {v.get("removed_wrong","—")} |')
        lines += ['','Recall tính TP/tất cả pixel GT của lớp, vì thế pixel ignore vẫn là bỏ sót; không bỏ ignore khỏi mẫu số để làm đẹp recall. Micro precision không thay thế mean IoU/từng lớp. [Thống kê từng lớp](probe/summary.json).','']
    else:lines += ['Probe dự kiến 128 ảnh development val cố định, chưa có kết quả.','']
    for kind in ['negative','density','clip']:
        v=read(job/'cache'/kind/'summary.json')
        if v:lines += [f'- Cache {kind}: {v["images"]:,} ảnh; bỏ {v["removed_fraction_foreground"]*100:.3f}% foreground gốc. Đây là độ thay đổi, không phải độ chính xác.']
    lines += ['','## Kết quả inference đủ VOC val','',
      '| Nhánh | Raw mIoU (%) | CRF mIoU (%) | Δ CRF với baseline (điểm) | Δ với đối chứng (điểm) |',
      '|---|---:|---:|---:|---:|',
      f'| Baseline 20k | {baseline["raw"]["miou"]*100:.4f} | {base:.4f} | +0,0000 | — |']
    observed={};plot=[('Baseline 20k',baseline['raw']['miou']*100,base)]
    for arm in p['arms']:
        name=arm['name'];v=read(job/'evaluation'/name/'metrics.json')
        if v:
            assert v['metadata']['validation_images']==1449 and not v['metadata']['partial_validation']
            raw=v['raw']['miou']*100;crf=v['crf']['miou']*100;observed[name]=crf
            parent=base if arm['parent']=='baseline' else observed.get(arm['parent'])
            effect=f'{crf-parent:+.4f}' if parent is not None else 'chưa có'
            lines.append(f'| {name} | {raw:.4f} | {crf:.4f} | {crf-base:+.4f} | {effect} |');plot.append((name,raw,crf))
        else:lines.append(f'| {name} | chưa có | chưa có | chưa có | chưa có |')
    lines += ['',
      'Giữ nguyên protocol cũ cho cả baseline và mọi nhánh: float32/TF32 off, 448×scales [1;1,25;1,5], flip, fusion gốc, không dùng image tags khi model inference val; CRF 10 vòng, pos_xy=2,pos_w=3,bi_xy=140,bi_rgb=1,bi_w=7. Không đổi sang cấu hình CRF khác để chọn kết quả.','',
      'Tiêu chí quan sát định trước: EMA cuối lịch tăng ≥0,2 điểm CRF và không lớp nào giảm >2 điểm IoU. CLIP báo cáo riêng vì thêm image-text pretraining. Một seed, thêm 4k bước và development val đã dùng nhiều vòng chưa chứng minh cải thiện tổng quát hóa; nếu có ứng viên cần kiểm chứng thêm seed/đánh giá độc lập.','']
    if refresh_plot:plot_comparison(job,plot)
    if (job/'figures/miou_delta.png').exists():lines += ['![Chỉ các kết quả đã đo](figures/miou_delta.png)','']
    lines += ['## Tài liệu và tệp kiểm chứng','',
      '- [Kế hoạch](plan.json) · [Trạng thái](status.json) · [Sự kiện](events.jsonl) · [Kiểm tra CLIP](clip_asset_checks.json).',
      f'- [Baseline metrics]({os.path.relpath(p["reference_baseline_metrics"],job)}) · [Phân tích vòng trước](../representation_sam_20261120/REPORT.md).',
      '- [FPR, ICCV 2023](https://www4.comp.polyu.edu.hk/~cslzhang/paper/ICCV23-FPR.pdf) · [U2PL, CVPR 2022](https://arxiv.org/abs/2203.03884).',
      '- [BRNF, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Bias-Resilient_Weakly_Supervised_Semantic_Segmentation_Using_Normalizing_Flows_ICCV_2025_paper.html) · [FFR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_FFR_Frequency_Feature_Rectification_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf).',
      '- [WeCLIP, CVPR 2024](https://arxiv.org/abs/2406.11189) · [CLIP chính thức](https://github.com/openai/CLIP).',
      '- bank/ lưu tensors và manifest; cache/ lưu nhãn và veto theo lớp; train/ lưu loss, parity, kiểm tra trọng số và adapter; evaluation/ lưu masks/raw/CRF/confusion/metrics. Adapter nhỏ tham chiếu baseline bằng SHA256, cần giữ cả baseline và snapshot để tái lập.','']
    temp=job/'REPORT.md.tmp';temp.write_text('\n'.join(lines));temp.replace(job/'REPORT.md')
