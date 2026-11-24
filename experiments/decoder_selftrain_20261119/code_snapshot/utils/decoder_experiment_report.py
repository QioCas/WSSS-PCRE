"""Render the experiment's persistent Markdown research report from observed files."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def render_report(job,refresh_plot=False):
    job=Path(job);plan=read(job/'plan.json');status=read(job/'status.json') or {'stage':'preparing','description':'Đang chuẩn bị và kiểm thử'}
    baseline=read(plan['reference_baseline_metrics'])
    rows=[('Baseline 20k',baseline['raw']['miou']*100,baseline['crf']['miou']*100)]
    for name,path in plan['historical_metrics'].items():
        value=read(path)
        if value:rows.append((name,value['raw']['miou']*100,value['crf']['miou']*100))
    new=[]
    for mode in plan['modes']:
        for kind in ['student','ema']:
            tag=mode+'_'+kind;value=read(job/'evaluation'/tag/'metrics.json')
            if value:
                assert value['metadata']['validation_images']==1449 and not value['metadata']['partial_validation']
                new.append((tag,value['raw']['miou']*100,value['crf']['miou']*100))
    decision=read(job/'decision.json')
    if decision:
        finding=decision['explanation']
    else:
        finding='Chưa có kết luận rằng ý tưởng mới vượt baseline. Các ô chưa đo được giữ là “chưa có”, không thay bằng 0.'
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines=['# Tinh chỉnh decoder PCRE bằng teacher baseline cố định','',f'Cập nhật: {now} · VOC 2012 · seed {plan["seed"]}','',
      '## Tóm tắt kết quả và trạng thái','',
      f'Baseline hiện tại đạt **{baseline["crf"]["miou"]*100:.4f}% mIoU sau CRF** trên toàn bộ 1.449 ảnh VOC val. Đây là mốc cần vượt. [1]','',
      f'**Trạng thái:** {status.get("description",status["stage"])}. {finding}','',
      'Ý tưởng mới giữ nguyên trọng số backbone, classifier và prototype của baseline; chỉ học lại decoder từ pseudo-label teacher. Hai nhánh được chạy cùng ngân sách để phân biệt ảnh hưởng của CRF với ảnh hưởng của việc huấn luyện thêm. [3]','',
      '## Kết quả đã có trước khi chạy ý tưởng mới','',
      '| Phương án | mIoU trước CRF (%) | mIoU sau CRF (%) | Δ sau CRF so với baseline (điểm) |',
      '|---|---:|---:|---:|']
    for name,raw,crf in rows:lines.append(f'| {name} | {raw:.4f} | {crf:.4f} | {crf-rows[0][2]:+.4f} |')
    lines += ['',
      'Adaptive cũ và adaptive consensus đều chưa vượt baseline. Consensus phục hồi được một số lớp như cat/dog nhưng vẫn giảm ở tvmonitor/boat/bicycle. Ensemble tăng rất nhỏ với chi phí hai model. Vì vậy thí nghiệm này chuyển sang giữ nguyên biểu diễn đã học thay vì tiếp tục sửa ngưỡng CAM và cập nhật toàn bộ backbone. [2]','',
      '## Giả thuyết cần kiểm chứng','',
      'Pseudo-label từ segmentation teacher đã hội tụ, kết hợp multi-scale/flip và nhãn lớp cấp ảnh, có thể hữu ích hơn nhãn CAM đang thay đổi theo quá trình học. CRF có thể bổ sung thông tin biên RGB. Đây là giả thuyết, chưa phải kết quả chất lượng nhãn được xác nhận.','',
      'Đóng băng backbone và dùng loss KL neo decoder về teacher có thể giảm trôi kiến thức. Điều này không bảo đảm mọi lớp giữ nguyên IoU; thay đổi decoder vẫn có thể gây hồi quy.','',
      'Self-training với pseudo-label và augmentation có tiền lệ trong ST++, nhưng báo cáo này không tái triển khai ST++ và không dùng thêm pixel ground truth để train. EMA chỉ trung bình trọng số decoder để giữ một model khi inference. [4,5]','',
      '## Phương pháp đã chốt trước khi đo kết quả','',
      '1. Teacher là checkpoint baseline 20k, tải strict và giữ cố định. Teacher dự đoán trên 10.582 ảnh train_aug bằng scales 1.0/1.25/1.5 và lật ngang.','2. Chỉ ở train, giới hạn lớp theo nhãn cấp ảnh. Lưu hai bộ pseudo-label: xác suất teacher gốc và xác suất sau DenseCRF.','3. Pixel không đạt cả ngưỡng confidence và khoảng cách giữa hai lớp đứng đầu được gán ignore=255. Cùng quy tắc được áp dụng cho hai nhánh.','4. Student khởi tạo từ chính baseline. Đóng băng toàn bộ phần ngoài ASPP decoder. Dùng resize/crop/flip đồng bộ giữa RGB và pseudo-label, cùng color jitter chỉ trên RGB.','5. Loss gồm CE cân bằng foreground/background trên pseudo-label và KL về decoder teacher cố định trên cùng feature đã đóng băng. Không có gradient qua teacher/backbone.','6. Cập nhật EMA của decoder. Checkpoint EMA cuối lịch là đầu ra chính đã định trước; checkpoint student cuối lịch là kết quả phụ. Không chọn checkpoint giữa lịch theo val.','',
      '`L = CE_balanced(student, pseudo_label) + lambda * T^2 * KL(p_teacher^T || p_student^T)`','',
      'Nếu batch thiếu foreground hoặc background, CE chỉ lấy trung bình các nhóm thực sự có nhãn. Khi toàn bộ pixel là ignore, phần CE bằng 0 và vẫn kiểm tra loss hữu hạn.','',
      '| Thành phần | Giá trị |','|---|---|',
      f'| Nguồn khởi tạo | Baseline đã train 20.000 bước |',
      f'| Hai nhánh | raw_teacher (đối chứng) / crf_teacher (đề xuất) |',
      f'| Bước bổ sung mỗi nhánh | {plan["refinement_steps"]:,} bước decoder |',
      f'| Batch / crop / seed | {plan["batch_size"]} / {plan["crop_size"]} / {plan["seed"]} |',
      f'| LR / optimizer / weight decay | {plan["lr"]} / AdamW / {plan["weight_decay"]} |',
      f'| Warm-up / lịch LR | {plan["warmup_steps"]} bước / cosine |',
      f'| Confidence / margin | {plan["confidence"]} / {plan["margin"]} |',
      f'| KL weight / temperature | {plan["anchor_weight"]} / {plan["anchor_temperature"]} |',
      f'| EMA decay | {plan["ema_decay"]} |',
      '| CRF | 10 vòng; pos_xy_std=2, pos_w=3, bi_xy_std=140, bi_rgb_std=1, bi_w=7 |','',
      '**Khác biệt ngân sách:** đây là fine-tuning bổ sung trên baseline 20k. Không được diễn giải thành phương pháp mới tốt hơn với tổng ngân sách chỉ 20k. Đối chứng raw_teacher dùng cùng dữ liệu, augmentation, LR, số bước và loss neo để kiểm tra phần đóng góp của nhãn CRF.','',
      '## Chống rò rỉ dữ liệu và kiểm tra triển khai','',
      '- Bộ tạo nhãn chỉ mở JPEGImages và nhãn lớp cấp ảnh của train_aug; dataset fine-tuning chỉ mở RGB và cache pseudo-label. Không mở mask ground truth để train.',
      '- Kiểm tra train/val không trùng ID. VOC val chỉ phục vụ đánh giá và đã được dùng trong phát triển phương pháp, nên không phải tập kiểm tra độc lập.',
      '- Kiểm tra đường forward encoder→decoder rút gọn cho cùng logits với forward đầy đủ ở chế độ eval trước khi tạo cache.',
      '- Hash trọng số ngoài decoder trước/sau train phải giống hệt. Loss phải hữu hạn; checkpoint dùng cùng kiến trúc và strict loading.',
      '- Không thay tham số inference giữa các nhánh; đánh giá đủ 1.449 ảnh, không lọc lớp bằng nhãn ảnh ở val.',
      '- Mã nguồn thực thi được chụp riêng trong code_snapshot để thay đổi repo về sau không âm thầm đổi thí nghiệm đang treo.','']
    tests=read(job/'checks.json')
    if tests:lines += ['**Kiểm tra đã ghi nhận:** '+json.dumps(tests,ensure_ascii=False), '']
    cache=read(job/'teacher_cache/summary.json')
    if cache:
        lines += ['### Độ phủ pseudo-label quan sát được','',
          'Độ phủ không phải độ chính xác; không dùng pixel GT để đo chất lượng cache.','',
          '| Bộ nhãn | Foreground (%) | Background (%) | Ignore (%) |','|---|---:|---:|---:|']
        for mode in plan['modes']:
            lines.append(f'| {mode} | {cache[mode]["foreground"]*100:.2f} | {cache[mode]["background"]*100:.2f} | {cache[mode]["ignore"]*100:.2f} |')
        lines.append('')
    lines += ['## Kết quả của thí nghiệm mới','',
      '| Nhánh / checkpoint | mIoU trước CRF (%) | mIoU sau CRF (%) | Δ sau CRF so với baseline (điểm) |','|---|---:|---:|---:|']
    measured={row[0]:row for row in new}
    for mode in plan['modes']:
        for kind in ['student','ema']:
            tag=mode+'_'+kind
            if tag in measured:
                _,raw,crf=measured[tag];lines.append(f'| {tag} | {raw:.4f} | {crf:.4f} | {crf-rows[0][2]:+.4f} |')
            else:lines.append(f'| {tag} | chưa có | chưa có | chưa có |')
    lines += ['',
      f'**Tiêu chí định trước:** checkpoint EMA của mỗi nhánh cần hơn baseline ít nhất {plan["target_gain_pp"]} điểm mIoU sau CRF và không có lớp nào giảm quá {plan["max_allowed_class_drop_pp"]} điểm IoU. Trong hai EMA đạt điều kiện, chọn mIoU cao nhất; checkpoint student chỉ là kết quả phụ, không dùng để chọn. Nếu không nhánh nào đạt, giữ baseline và ghi nhận kết quả âm. So sánh riêng crf_teacher_ema với raw_teacher_ema để kiểm chứng phần đóng góp của CRF; nếu nhánh raw tốt hơn thì không quy mức tăng cho CRF.','',
      'Một seed và một tập val không đủ chứng minh tổng quát hóa. Nếu có tăng điểm, vẫn cần lặp seed và đánh giá độc lập trước khi kết luận ổn định. Không thay ngưỡng hoặc chọn checkpoint khác sau khi thấy val để làm đẹp kết quả.','']
    if (job/'figures/miou_delta.png').exists() or refresh_plot:
        if refresh_plot:plot_comparison(job,rows+new)
        lines += ['![Chênh lệch mIoU sau CRF so với baseline](figures/miou_delta.png)','',
                  'Hình biểu diễn chênh lệch theo điểm phần trăm, không phải phần trăm tương đối. Các phương án fine-tuning có ngân sách huấn luyện bổ sung.','']
    lines += ['## Nguồn và tệp kiểm chứng','',
       f'1. [Metrics baseline]({os.path.relpath(plan["reference_baseline_metrics"],job)}): mIoU và confusion matrix trên VOC val.',
       f'2. [Báo cáo vòng thử trước]({os.path.relpath(Path(plan["teacher_run"]).parents[1]/"pcre_repair_20261119/RUN_STATUS.md",job)}): kết quả adaptive/consensus/ensemble.',
       '3. [Kế hoạch và tham số cố định](plan.json); [trạng thái thực thi](status.json); [nhật ký pipeline](events.jsonl).',
       '4. [ST++, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_ST_Make_Self-Training_Work_Better_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.html): cảm hứng self-training với pseudo-label và augmentation.',
       '5. [PyTorch: EMA/AveragedModel](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html): phép trung bình trọng số EMA.',
       '6. Thư mục teacher_cache lưu manifest, SHA256 teacher/tags, thống kê và mask; train/<mode> lưu loss, kiểm tra đóng băng và checkpoint; evaluation/<mode>_<kind> lưu metrics, confusion matrix và mask.','']
    tmp=job/'REPORT.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(job/'REPORT.md')


def plot_comparison(job,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline=rows[0][2];names=[r[0] for r in rows];delta=[r[2]-baseline for r in rows]
    fig,ax=plt.subplots(figsize=(10,max(3.2,.45*len(rows)+1)),layout='constrained')
    ax.axvline(0,color='#777777',linewidth=1)
    ax.scatter(delta,range(len(rows)),color=['#555555' if i<4 else '#2265a0' for i in range(len(rows))],s=45)
    for i,v in enumerate(delta):ax.annotate(f'{v:+.3f}',(v,i),xytext=(7,0),textcoords='offset points',va='center')
    ax.set_yticks(range(len(rows)),names);ax.invert_yaxis();ax.set_xlabel('CRF mIoU difference vs baseline (percentage points)')
    ax.set_title('VOC val: observed results only');ax.spines[['top','right']].set_visible(False)
    low=min(delta+[0])-.15;high=max(delta+[0])+.4;ax.set_xlim(low,high)
    (job/'figures').mkdir(exist_ok=True);fig.savefig(job/'figures/miou_delta.png',dpi=160);plt.close(fig)
