# PCRE: residual, negative prototypes và kiểm chứng precision

Cập nhật 2026-11-20 09:34 UTC. **Trạng thái:** Hoàn tất sáu nhánh, mỗi nhánh đủ VOC val. Tuyến không thêm pretraining: baseline. CLIP region filter đạt 74.2666% CRF (+0.0423 điểm), chưa đạt tiêu chí quan sát. Đây là một seed trên development val với ngân sách fine-tuning thêm; chưa kết luận cải thiện ổn định.

Baseline bất biến: **74.2243% mIoU CRF**, 1.449 ảnh VOC val. Mọi nhánh bắt đầu từ cùng checkpoint baseline 20k; thêm 4.000 bước residual, batch 4, seed 0.

## Câu hỏi và thứ tự thí nghiệm

Vòng trước tăng recall nhưng giảm precision ở một số lớp; self-training làm sai thêm các vùng baseline đã đúng. Vòng này kiểm tra bằng chứng âm và giới hạn phạm vi sửa. Không tuyên bố đây là bản tái triển khai đầy đủ FPR/U2PL/BRNF/FFR/WeCLIP.

| Nhánh | Thay đổi | Đối chứng trực tiếp | Giám sát bổ sung |
|---|---|---|---|
| A1_residual_control | Head nhỏ, zero-init, baseline đóng băng | Baseline | Không |
| A2_negative_filter | Bỏ foreground mâu thuẫn với negative bank khỏi CE/KL | A1 | Không |
| A3_negative_learning | Thêm loss phủ định cho lớp bị veto | A2 | Không |
| B1_distribution_filter | GMM kiểm tra phân bố feature và loại outlier | A1 | Không |
| C1_detail_residual | Thêm feature block 4 và thành phần high-pass vào residual | A1 | Không |
| D1_clip_region_filter | CLIP kiểm tra nhãn connected component | A1 | Image–text pretraining |

Các nhánh chạy tuần tự và độc lập, không lấy checkpoint nhánh trước làm khởi tạo nhánh sau. “Thêm” trong A2/A3 là thêm thành phần phương pháp với cùng khởi tạo/ngân sách, không cộng dồn số bước train.

## Phương pháp và tham số chốt trước khi đo

`logits_new = logits_baseline + gate * limit * tanh(residual / limit)`

Toàn bộ baseline đóng băng. Head 1×1→GELU→3×3→GELU→1×1, width 64; lớp cuối khởi tạo 0, limit=4.0 logits. Gate giống nhau ở mọi nhánh, chỉ dùng prediction không lọc tags: cho phép sửa foreground dự đoán và background có confidence <0,95. Background tự tin được giữ logits trên lưới feature; interpolation/TTA/CRF vẫn có thể ảnh hưởng mask cuối. Zero-init chỉ bảo đảm trùng đầu ra trước khi học, không bảo đảm không hồi quy sau train.

Loss gồm CE cân bằng foreground/background + KL giữ posterior baseline, weight 1; chỉ áp KL khi cache không ignore, có support thật và argmax baseline đồng ý nhãn cache. A3 thêm `-log(1-p(rejected_class))`, weight 0,1; không suy ra nhãn dương từ một veto. Không đưa pixel ignore trở lại qua argmax/KL.

AdamW LR 1e-4, weight decay 0,01; warm-up 100 bước, cosine; EMA 0,99. Resize short side [336,672], crop 448, flip, jitter 0,1. Không strong/cutout trong vòng này. Chỉ EMA cuối lịch dùng cho kết quả chính.

### Nguồn nhãn và bank

Tái dùng reliable cache của baseline trên đủ 10.582 ảnh train_aug: mean probability của 6 views, confidence ≥0,8 và 5/6 votes. Bank chỉ đọc ảnh train, cache và image tags. Positive features lấy từ lõi erode 1 patch, tối đa 16 patch/lớp/ảnh, reservoir 2.048/lớp; KMeans 4 positive prototypes.

Negative features lấy ở ảnh không có lớp c, từ raw CAM c trước khi lọc tags, đồng thời yêu cầu decoder baseline tin background ≥0,8. Điều kiện background nhằm giảm negatives từ bộ phận foreground dùng chung. Giữ 512 ứng viên/lớp, tối đa 2/ảnh; chọn 8 hướng hard negative. Veto khi cosine với negative vượt positive hơn 0,08. Chỉ chuyển foreground tương ứng sang ignore; không đổi lớp hoặc tạo background mới. Đây là FPR-inspired, không phải nguyên bản FPR.

B1 dùng random orthogonal projection 32 chiều, L2-normalized, diagonal GMM 2 thành phần/lớp, variance regularizer 0,005; cutoff theo quantile 2% thấp nhất của positive train samples. GMM có thể kế thừa bias seed. Đây là thử nghiệm density gate nhỏ, không phải normalizing-flow BRNF.

C1 lấy high-pass của feature block 4 bằng `early - avg_pool_3x3(early)`, nối với final features vào head. Không có boundary GT, không gọi đây là full FFR hay tái tạo được chi tiết dưới kích thước patch.

D1 dùng frozen OpenAI CLIP ViT-B/16, prompt tên 20 lớp VOC và background, so cosine trên crop của tối đa 2 connected component/lớp/ảnh; bỏ component nếu background/lớp khác có mặt thắng hơn 0,05. Đây là semantic region verifier lấy cảm hứng từ WeCLIP, không phải dense WeCLIP. CLIP chỉ dùng khi tạo nhãn train/diagnostic, không thêm model CLIP lúc inference student.

## Kiểm tra và chẩn đoán nhãn trước full training

Kiểm tra logits zero-init, đường forward so với segmentation baseline, phần đóng băng và gradient, gate, KL loại ignore/padding/conflict, loss phủ định, negative/density support, hình học RGB/nhãn đồng bộ và strict checkpoint reload. Mã thực thi và danh sách ảnh được freeze vào code_snapshot.

Kết quả smoke/unit tests: `{"unit_tests_passed": 37, "bank_negative_density_clip_smokes": true, "all_five_training_variants_passed": true, "residual_and_detail_strict_reload_multiscale_passed": true, "label_quality_probe_passed": true, "zero_init_parity": {"zero_residual_logits_bitwise_equal": true, "base_segmentation_max_abs_error": 0.0, "initial_residual_parameters": 87509}, "frozen_baseline_unchanged": true}`

Bank hoàn tất: 10,582 ảnh; training pixel GT read = False.

Chẩn đoán trên 128 ảnh val chọn cố định bằng seed 0; dùng image tags để mô phỏng pseudo-label train. GT chỉ được đọc sau khi tạo nhãn để chấm diagnostic. Không dùng masks/GT diagnostic để train, không fit bank trên val và không đổi ngưỡng theo diagnostic. Đây không phải mIoU model inference trên toàn val.

| Pseudo-label | Foreground micro precision (%) | Foreground micro recall (%) | Pixel đúng bị bỏ | Pixel sai bị bỏ |
|---|---:|---:|---:|---:|
| reliable | 86.410 | 81.354 | — | — |
| negative | 86.522 | 80.950 | 22630 | 10382 |
| density | 86.599 | 78.767 | 144946 | 33946 |
| clip | 86.410 | 81.354 | 0 | 0 |

Recall tính TP/tất cả pixel GT của lớp, vì thế pixel ignore vẫn là bỏ sót; không bỏ ignore khỏi mẫu số để làm đẹp recall. Micro precision không thay thế mean IoU/từng lớp. [Thống kê từng lớp](probe/summary.json).

- Cache negative: 10,582 ảnh; bỏ 1.057% foreground gốc. Đây là độ thay đổi, không phải độ chính xác.
- Cache density: 10,582 ảnh; bỏ 4.285% foreground gốc. Đây là độ thay đổi, không phải độ chính xác.
- Cache clip: 10,582 ảnh; bỏ 0.299% foreground gốc. Đây là độ thay đổi, không phải độ chính xác.

## Kết quả inference đủ VOC val

| Nhánh | Raw mIoU (%) | CRF mIoU (%) | Δ CRF với baseline (điểm) | Δ với đối chứng (điểm) |
|---|---:|---:|---:|---:|
| Baseline 20k | 73.7895 | 74.2243 | +0,0000 | — |
| A1_residual_control | 73.7510 | 74.2682 | +0.0439 | +0.0439 |
| A2_negative_filter | 73.7624 | 74.2774 | +0.0531 | +0.0092 |
| A3_negative_learning | 72.5888 | 72.8327 | -1.3916 | -1.4447 |
| B1_distribution_filter | 73.7733 | 74.2845 | +0.0602 | +0.0163 |
| C1_detail_residual | 73.7717 | 74.2860 | +0.0617 | +0.0177 |
| D1_clip_region_filter | 73.7501 | 74.2666 | +0.0423 | -0.0016 |

Giữ nguyên protocol cũ cho cả baseline và mọi nhánh: float32/TF32 off, 448×scales [1;1,25;1,5], flip, fusion gốc, không dùng image tags khi model inference val; CRF 10 vòng, pos_xy=2,pos_w=3,bi_xy=140,bi_rgb=1,bi_w=7. Không đổi sang cấu hình CRF khác để chọn kết quả.

Tiêu chí quan sát định trước: EMA cuối lịch tăng ≥0,2 điểm CRF và không lớp nào giảm >2 điểm IoU. CLIP báo cáo riêng vì thêm image-text pretraining. Một seed, thêm 4k bước và development val đã dùng nhiều vòng chưa chứng minh cải thiện tổng quát hóa; nếu có ứng viên cần kiểm chứng thêm seed/đánh giá độc lập.

![Chỉ các kết quả đã đo](figures/miou_delta.png)

## Tài liệu và tệp kiểm chứng

- [Kế hoạch](plan.json) · [Trạng thái](status.json) · [Sự kiện](events.jsonl) · [Kiểm tra CLIP](clip_asset_checks.json).
- [Baseline metrics](../inference_voc_val_20261119/full/baseline/metrics.json) · [Phân tích vòng trước](../representation_sam_20261120/REPORT.md).
- [FPR, ICCV 2023](https://www4.comp.polyu.edu.hk/~cslzhang/paper/ICCV23-FPR.pdf) · [U2PL, CVPR 2022](https://arxiv.org/abs/2203.03884).
- [BRNF, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Bias-Resilient_Weakly_Supervised_Semantic_Segmentation_Using_Normalizing_Flows_ICCV_2025_paper.html) · [FFR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_FFR_Frequency_Feature_Rectification_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf).
- [WeCLIP, CVPR 2024](https://arxiv.org/abs/2406.11189) · [CLIP chính thức](https://github.com/openai/CLIP).
- bank/ lưu tensors và manifest; cache/ lưu nhãn và veto theo lớp; train/ lưu loss, parity, kiểm tra trọng số và adapter; evaluation/ lưu masks/raw/CRF/confusion/metrics. Adapter nhỏ tham chiếu baseline bằng SHA256, cần giữ cả baseline và snapshot để tái lập.
