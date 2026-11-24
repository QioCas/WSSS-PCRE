# Tinh chỉnh decoder PCRE bằng teacher baseline cố định

Cập nhật: 2026-11-19 19:57 UTC · VOC 2012 · seed 0

## Tóm tắt kết quả và trạng thái

Baseline hiện tại đạt **74.2243% mIoU sau CRF** trên toàn bộ 1.449 ảnh VOC val. Đây là mốc cần vượt. [1]

**Trạng thái:** Hoàn tất cả hai nhánh và đánh giá đủ VOC val. Chưa có checkpoint EMA nào đạt đồng thời mục tiêu tăng mIoU và giới hạn giảm theo lớp. Giữ baseline; các kết quả âm được giữ nguyên trong bảng.

Ý tưởng mới giữ nguyên trọng số backbone, classifier và prototype của baseline; chỉ học lại decoder từ pseudo-label teacher. Hai nhánh được chạy cùng ngân sách để phân biệt ảnh hưởng của CRF với ảnh hưởng của việc huấn luyện thêm. [3]

## Kết quả đã có trước khi chạy ý tưởng mới

| Phương án | mIoU trước CRF (%) | mIoU sau CRF (%) | Δ sau CRF so với baseline (điểm) |
|---|---:|---:|---:|
| Baseline 20k | 73.7895 | 74.2243 | +0.0000 |
| Adaptive cũ | 72.9588 | 73.4448 | -0.7795 |
| Adaptive consensus | 73.1463 | 73.7189 | -0.5054 |
| Ensemble hai model | 73.8795 | 74.3210 | +0.0968 |

Adaptive cũ và adaptive consensus đều chưa vượt baseline. Consensus phục hồi được một số lớp như cat/dog nhưng vẫn giảm ở tvmonitor/boat/bicycle. Ensemble tăng rất nhỏ với chi phí hai model. Vì vậy thí nghiệm này chuyển sang giữ nguyên biểu diễn đã học thay vì tiếp tục sửa ngưỡng CAM và cập nhật toàn bộ backbone. [2]

## Giả thuyết cần kiểm chứng

Pseudo-label từ segmentation teacher đã hội tụ, kết hợp multi-scale/flip và nhãn lớp cấp ảnh, có thể hữu ích hơn nhãn CAM đang thay đổi theo quá trình học. CRF có thể bổ sung thông tin biên RGB. Đây là giả thuyết, chưa phải kết quả chất lượng nhãn được xác nhận.

Đóng băng backbone và dùng loss KL neo decoder về teacher có thể giảm trôi kiến thức. Điều này không bảo đảm mọi lớp giữ nguyên IoU; thay đổi decoder vẫn có thể gây hồi quy.

Self-training với pseudo-label và augmentation có tiền lệ trong ST++, nhưng báo cáo này không tái triển khai ST++ và không dùng thêm pixel ground truth để train. EMA chỉ trung bình trọng số decoder để giữ một model khi inference. [4,5]

## Phương pháp đã chốt trước khi đo kết quả

1. Teacher là checkpoint baseline 20k, tải strict và giữ cố định. Teacher dự đoán trên 10.582 ảnh train_aug bằng scales 1.0/1.25/1.5 và lật ngang.
2. Chỉ ở train, giới hạn lớp theo nhãn cấp ảnh. Lưu hai bộ pseudo-label: xác suất teacher gốc và xác suất sau DenseCRF.
3. Pixel không đạt cả ngưỡng confidence và khoảng cách giữa hai lớp đứng đầu được gán ignore=255. Cùng quy tắc được áp dụng cho hai nhánh.
4. Student khởi tạo từ chính baseline. Đóng băng toàn bộ phần ngoài ASPP decoder. Dùng resize/crop/flip đồng bộ giữa RGB và pseudo-label, cùng color jitter chỉ trên RGB.
5. Loss gồm CE cân bằng foreground/background trên pseudo-label và KL về decoder teacher cố định trên cùng feature đã đóng băng. Không có gradient qua teacher/backbone.
6. Cập nhật EMA của decoder. Checkpoint EMA cuối lịch là đầu ra chính đã định trước; checkpoint student cuối lịch là kết quả phụ. Không chọn checkpoint giữa lịch theo val.

`L = CE_balanced(student, pseudo_label) + lambda * T^2 * KL(p_teacher^T || p_student^T)`

Nếu batch thiếu foreground hoặc background, CE chỉ lấy trung bình các nhóm thực sự có nhãn. Khi toàn bộ pixel là ignore, phần CE bằng 0 và vẫn kiểm tra loss hữu hạn.

| Thành phần | Giá trị |
|---|---|
| Nguồn khởi tạo | Baseline đã train 20.000 bước |
| Hai nhánh | raw_teacher (đối chứng) / crf_teacher (đề xuất) |
| Bước bổ sung mỗi nhánh | 3,000 bước decoder |
| Batch / crop / seed | 4 / 448 / 0 |
| LR / optimizer / weight decay | 2e-05 / AdamW / 0.01 |
| Warm-up / lịch LR | 100 bước / cosine |
| Confidence / margin | 0.7 / 0.5 |
| KL weight / temperature | 1.0 / 2.0 |
| EMA decay | 0.99 |
| CRF | 10 vòng; pos_xy_std=2, pos_w=3, bi_xy_std=140, bi_rgb_std=1, bi_w=7 |

**Khác biệt ngân sách:** đây là fine-tuning bổ sung trên baseline 20k. Không được diễn giải thành phương pháp mới tốt hơn với tổng ngân sách chỉ 20k. Đối chứng raw_teacher dùng cùng dữ liệu, augmentation, LR, số bước và loss neo để kiểm tra phần đóng góp của nhãn CRF.

## Chống rò rỉ dữ liệu và kiểm tra triển khai

- Bộ tạo nhãn chỉ mở JPEGImages và nhãn lớp cấp ảnh của train_aug; dataset fine-tuning chỉ mở RGB và cache pseudo-label. Không mở mask ground truth để train.
- Kiểm tra train/val không trùng ID. VOC val chỉ phục vụ đánh giá và đã được dùng trong phát triển phương pháp, nên không phải tập kiểm tra độc lập.
- Kiểm tra đường forward encoder→decoder rút gọn cho cùng logits với forward đầy đủ ở chế độ eval trước khi tạo cache.
- Hash trọng số ngoài decoder trước/sau train phải giống hệt. Loss phải hữu hạn; checkpoint dùng cùng kiến trúc và strict loading.
- Không thay tham số inference giữa các nhánh; đánh giá đủ 1.449 ảnh, không lọc lớp bằng nhãn ảnh ở val.
- Mã nguồn thực thi được chụp riêng trong code_snapshot để thay đổi repo về sau không âm thầm đổi thí nghiệm đang treo.

**Kiểm tra đã ghi nhận:** {"unit_tests_passed": 21, "smoke_cache_passed": true, "smoke_training_passed": true, "smoke_inference_passed": true, "frozen_weights_unchanged": true, "trainable_decoder_parameters": 580692, "segmentation_forward_parity": {"passed": true, "max_abs_error": 0.0, "rtol": 1e-05, "atol": 1e-05}}

### Độ phủ pseudo-label quan sát được

Độ phủ không phải độ chính xác; không dùng pixel GT để đo chất lượng cache.

| Bộ nhãn | Foreground (%) | Background (%) | Ignore (%) |
|---|---:|---:|---:|
| raw_teacher | 29.94 | 68.55 | 1.51 |
| crf_teacher | 30.39 | 69.43 | 0.18 |

## Kết quả của thí nghiệm mới

| Nhánh / checkpoint | mIoU trước CRF (%) | mIoU sau CRF (%) | Δ sau CRF so với baseline (điểm) |
|---|---:|---:|---:|
| raw_teacher_student | 73.7032 | 74.1950 | -0.0293 |
| raw_teacher_ema | 73.7029 | 74.1950 | -0.0292 |
| crf_teacher_student | 73.7300 | 74.1954 | -0.0289 |
| crf_teacher_ema | 73.7299 | 74.1954 | -0.0289 |

**Tiêu chí định trước:** checkpoint EMA của mỗi nhánh cần hơn baseline ít nhất 0.2 điểm mIoU sau CRF và không có lớp nào giảm quá 2.0 điểm IoU. Trong hai EMA đạt điều kiện, chọn mIoU cao nhất; checkpoint student chỉ là kết quả phụ, không dùng để chọn. Nếu không nhánh nào đạt, giữ baseline và ghi nhận kết quả âm. So sánh riêng crf_teacher_ema với raw_teacher_ema để kiểm chứng phần đóng góp của CRF; nếu nhánh raw tốt hơn thì không quy mức tăng cho CRF.

Một seed và một tập val không đủ chứng minh tổng quát hóa. Nếu có tăng điểm, vẫn cần lặp seed và đánh giá độc lập trước khi kết luận ổn định. Không thay ngưỡng hoặc chọn checkpoint khác sau khi thấy val để làm đẹp kết quả.

![Chênh lệch mIoU sau CRF so với baseline](figures/miou_delta.png)

Hình biểu diễn chênh lệch theo điểm phần trăm, không phải phần trăm tương đối. Các phương án fine-tuning có ngân sách huấn luyện bổ sung.

## Nguồn và tệp kiểm chứng

1. [Metrics baseline](../inference_voc_val_20261119/full/baseline/metrics.json): mIoU và confusion matrix trên VOC val.
2. [Báo cáo vòng thử trước](../pcre_repair_20261119/RUN_STATUS.md): kết quả adaptive/consensus/ensemble.
3. [Kế hoạch và tham số cố định](plan.json); [trạng thái thực thi](status.json); [nhật ký pipeline](events.jsonl).
4. [ST++, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_ST_Make_Self-Training_Work_Better_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.html): cảm hứng self-training với pseudo-label và augmentation.
5. [PyTorch: EMA/AveragedModel](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html): phép trung bình trọng số EMA.
6. Thư mục teacher_cache lưu manifest, SHA256 teacher/tags, thống kê và mask; train/<mode> lưu loss, kiểm tra đóng băng và checkpoint; evaluation/<mode>_<kind> lưu metrics, confusion matrix và mask.
