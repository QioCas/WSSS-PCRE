# PCRE: kiểm chứng biểu diễn, consistency và pseudo-label từ SAM

Cập nhật 2026-11-20 05:44 UTC. VOC 2012; seed 0.

Baseline cố định: **74.2243% mIoU sau CRF** trên đủ 1.449 ảnh val. **Trạng thái:** Hoàn tất ba nhánh và đánh giá đủ VOC val. Tuyến không thêm mask pretraining: chọn baseline. last2_consistency so với decoder_control: +0.0317 điểm. Tuyến có SAM: 74.1076% CRF (-0.1167 điểm với baseline), chưa đạt tiêu chí định trước. SAM dùng thêm giám sát mask từ pretraining; tất cả là kết quả một seed trên development val.

## Vì sao các vòng trước chưa thắng?

| Phương án đã đo | mIoU raw (%) | mIoU CRF (%) | Δ CRF (điểm) |
|---|---:|---:|---:|
| Baseline 20k | 73.7895 | 74.2243 | +0.0000 |
| Adaptive cũ | 72.9588 | 73.4448 | -0.7795 |
| Consensus repair | 73.1463 | 73.7189 | -0.5054 |
| Ensemble hai model | 73.8795 | 74.3210 | +0.0968 |
| Decoder raw EMA | 73.7029 | 74.1950 | -0.0292 |
| Decoder CRF EMA | 73.7299 | 74.1954 | -0.0289 |

Adaptive và consensus thay đổi pseudo-label/đường học nên vẫn có thể làm giảm một số lớp dù loss hội tụ. Ensemble từng tăng khoảng 0,097 điểm với hai model; mức tăng đó chưa phải bằng chứng một phương pháp đơn model tốt hơn. Các vòng fine-tuning decoder tiếp theo vẫn thấp hơn baseline khoảng 0,029 điểm. [Báo cáo trước](../decoder_selftrain_20261119/REPORT.md).

Đo lại cache trên toàn bộ train_aug: chỉ **0.5080%** pixel mà cả raw và CRF cùng gán nhãn có nhãn semantic khác nhau. Raw bỏ qua 1.508%, CRF bỏ qua 0.185% pixel. Hai nhánh vì thế nhận tín hiệu gần giống nhau, và ngưỡng cũ giữ gần như toàn bộ nhãn của teacher. [Số đo](failure_analysis.json).

Decoder vẫn thay đổi trọng số (relative L2 3.69%), nên thất bại không phải vì optimizer đứng yên. Tuy nhiên backbone đóng băng giới hạn biểu diễn; KL cũ dùng cùng view/features nên không buộc học tính bất biến giữa view. Đây là các giả thuyết hạn chế, chưa phải kết luận nhân quả. Không suy ra độ chính xác pseudo-label chỉ từ confidence hoặc độ phủ.

## Thiết kế thí nghiệm mới

| Nhánh | Trọng số được học | Pseudo-label | Mục đích |
|---|---|---|---|
| decoder_control | ASPP decoder | Reliable teacher | Đối chứng cho thay đổi augmentation/loss/nhãn |
| last2_consistency | Decoder + 2 block ViT cuối + final norm | Cùng reliable teacher | Kiểm tra việc mở biểu diễn |
| last2_sam | Giống last2_consistency | Reliable teacher + SAM | Kiểm tra tín hiệu hình dạng ngoài teacher |

**Hai tuyến kết quả:** hai nhánh đầu giữ nguồn giám sát của baseline, nhưng thêm ngân sách fine-tuning. Nhánh SAM sử dụng mask-supervised pretraining bên ngoài (SAM/SA-1B), phải báo cáo riêng; không được gọi là cải thiện WSSS cùng dữ liệu/giám sát. SAM không nhận VOC pixel ground truth từ pipeline này.

1. Khởi tạo tất cả student và teacher từ cùng baseline 20k. Teacher luôn đóng băng. Tạo cache cho 10.582 ảnh train_aug, kiểm tra không giao với 1.449 ảnh val.
2. Teacher dự đoán 3 scales × 2 flip; softmax từng view trước khi lấy trung bình, tránh tăng độ sắc do cộng logits. Giới hạn lớp bằng nhãn cấp ảnh chỉ khi tạo nhãn train. Giữ pixel có mean confidence ≥ 0,8 và ít nhất 5/6 view đồng ý; còn lại ignore=255.
3. Teacher online nhìn weak view; student nhìn strong view cùng resize/crop/flip. Strong dùng jitter 0,4, grayscale xác suất 0,1, blur 0,2, cutout 0,5 với diện tích 0,1. Loại cutout và padding khỏi cả CE/KL.
4. Loss = CE cân bằng foreground/background + 0,25 × T² KL(teacher weak || student strong), T=2. KL chỉ dùng teacher confidence ≥ 0,7; bỏ pixel mà nhãn cache đã gán khác teacher để không neo ngược sửa đổi SAM.
5. Mỗi nhánh 4.000 bước bổ sung, batch 4, crop 448, AdamW weight decay 0,01; LR decoder 2e-5, backbone 2e-6; warm-up 100, cosine; EMA 0,99. Chỉ checkpoint EMA cuối lịch được đánh giá chính.
6. SAM ViT-B nhận positive point trong lõi teacher, box quanh connected component, có thể thêm negative point từ lớp khác. Chỉ giữ mask có predicted quality ≥ 0,85, teacher-mask IoU ≥ 0,5, reliable-core recall ≥ 0,9, tỷ lệ diện tích trong [0,5; 2]. Các số này là kiểm tra tương hợp teacher, không phải IoU với ground truth.
7. Giới hạn sửa trong vùng 12 pixel quanh teacher component; lõi SAM erode 3 pixel làm nhãn, biên bất định thành ignore. Bảo vệ lõi reliable, cấm chuyển một foreground đáng tin cậy sang lớp foreground khác; xung đột proposal thành ignore.

Mở 2 block thay vì toàn backbone nhằm cho phép sửa đặc trưng với LR nhỏ. Đối chứng decoder và last2 dùng cùng ảnh, nhãn, augmentation, loss và số bước; so sánh last2 với SAM chỉ thay nguồn nhãn. ST++ và Noisy Student là cảm hứng về reliability/weak–strong self-training, không phải bản tái triển khai hay bằng chứng phương pháp này sẽ thắng.

## Đánh giá và tiêu chí định trước

Giữ nguyên baseline inference: float32, TF32 off, scales 1/1,25/1,5 + flip, fusion gốc, không dùng nhãn ảnh để lọc lớp val. DenseCRF: 10 vòng, pos_xy_std=2, pos_w=3, bi_xy_std=140, bi_rgb_std=1, bi_w=7. Đánh giá đủ 1.449 ảnh, giữ cả raw và CRF, IoU từng lớp và confusion matrix. Không chọn checkpoint giữa lịch theo val.

Mục tiêu thực dụng: tăng ≥ 0.2 điểm mIoU CRF và không lớp nào giảm quá 2.0 điểm IoU. Chỉ so chọn giữa hai nhánh không SAM; nhánh SAM có quyết định riêng. Chưa đạt thì giữ baseline, giữ nguyên kết quả âm. Một seed, thêm 4k bước và val đã dùng phát triển phương pháp không đủ chứng minh tổng quát hóa hay công bằng tổng ngân sách.

| Nhánh EMA cuối lịch | Raw (%) | CRF (%) | Δ CRF (điểm) | Giám sát ngoài |
|---|---:|---:|---:|---|
| decoder_control | 73.3887 | 74.0427 | -0.1816 | không thêm |
| last2_consistency | 73.4515 | 74.0744 | -0.1499 | không thêm |
| last2_sam | 73.4787 | 74.1076 | -0.1167 | SAM/SA-1B |

[Quyết định và đối chứng](decision.json) · [IoU từng lớp](per_class_results.csv)

## Kiểm tra triển khai và trạng thái chi tiết

Kiểm thử reliability, gradient qua block được mở, weak/strong geometry, padding/cutout, KL mask, lọc SAM, bảo vệ lớp và xung đột proposal. Smoke test GPU tạo nhãn → train → strict-load checkpoint → inference. So hash toàn bộ teacher và phần student đóng băng trước/sau train. Training dataset chỉ đọc JPEG, pseudo-label và nhãn cấp ảnh. Mã nguồn và danh sách ảnh được chụp vào code_snapshot; ghi SHA256 checkpoint và SAM.

Kết quả kiểm tra: `{"unit_tests_passed": 28, "reliable_cache_passed": true, "sam_cache_passed": true, "decoder_training_passed": true, "partial_backbone_training_passed": true, "strict_checkpoint_inference_passed": true, "frozen_student_unchanged": true, "teacher_unchanged": true, "last2_trainable_parameters": 14757972}`

- Cache reliable: 10,582 ảnh; foreground 27.00%, background 66.52%, ignore 6.48%.
- Cache sam: 10,582 ảnh; foreground 27.95%, background 66.12%, ignore 5.94%.
  SAM chấp nhận 11,602/17,342 proposal, sửa 3.151% pixel; đây là độ thay đổi, không phải độ chính xác.
- Train decoder_control: 4000/4000 bước.
- Train last2_consistency: 4000/4000 bước.
- Train last2_sam: 4000/4000 bước.

![Kết quả đã đo, không điền dự đoán cho nhánh chưa chạy](figures/miou_delta.png)

## Tệp và tài liệu tham khảo

- [Baseline metrics](../inference_voc_val_20261119/full/baseline/metrics.json) · [Kế hoạch](plan.json) · [Phân tích thất bại](failure_analysis.json).
- [Trạng thái](status.json) · [Sự kiện](events.jsonl) · logs/ · train/ · evaluation/.
- [ST++, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_ST_Make_Self-Training_Work_Better_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.html).
- [Noisy Student](https://arxiv.org/abs/1911.04252): self-training với nhiễu, nghiên cứu classification.
- [SAM chính thức](https://github.com/facebookresearch/segment-anything) và [predictor API](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py). Checkpoint pretrained trên SA-1B, khoảng 11 triệu ảnh và 1,1 tỷ mask theo tác giả.
- [Kiểm tra tài nguyên SAM](sam_asset_checks.json). Các giá trị SAM predicted quality không phải IoU đo trên VOC GT.

## Phân tích sau khi hoàn tất: vì sao chưa vượt baseline?

Đã tái dựng confusion matrix từ toàn bộ mask CRF của baseline và ba nhánh trên cùng 1.449 ảnh val; kết quả khớp metric đã lưu. Các số dưới đây không đến từ loss train hoặc ảnh smoke test.

### Mất precision khi tăng recall

Nhánh SAM dự đoán tổng diện tích foreground lớn hơn baseline 4,34% tương đối. Nó sửa được 1.991.047 pixel baseline dự đoán sai nhưng làm sai 2.022.924 pixel baseline đã dự đoán đúng. Đây là đếm pixel, không phải công thức mIoU. Với mIoU trung bình 21 lớp, phần tăng đóng góp +0,4776 điểm và phần giảm đóng góp −0,5943 điểm, tổng −0,1167 điểm.

| Lớp, nhánh SAM so với baseline | Δ precision (điểm) | Δ recall (điểm) | Δ IoU (điểm) |
|---|---:|---:|---:|
| bicycle | -4.659 | +4.401 | -2.306 |
| bird | -3.867 | +1.471 | -2.249 |
| boat | -2.345 | +0.614 | -1.706 |
| tvmonitor | -2.197 | +0.973 | -1.208 |
| person | -1.861 | +4.527 | +2.873 |
| sofa | +3.269 | -4.571 | -1.441 |

Person/cow/dog được lợi từ tăng recall; bicycle/bird/boat bị mất precision, còn sofa giảm recall. Vì vậy không thể mô tả mọi lỗi là chỉ nở foreground.

### Cơ chế phù hợp với bằng chứng, chưa phải khẳng định nhân quả

- Teacher cố định tạo cả pseudo-label và mục tiêu KL. Đồng thuận giữa sáu view vẫn là đồng thuận của cùng một model; không xác nhận nhãn đúng. Raw và CRF cache vòng trước chỉ khác semantic ở 0,508% pixel cùng được gán nhãn.
- Cache reliable yêu cầu confidence 0,8 và 5/6 votes, nhưng KL online dùng confidence 0,7 và cho phép labels==255. Trên các batch có log của last2_consistency, CE dùng trung bình 84,75% pixel, KL dùng 87,36%; trung bình ít nhất 2,84% tổng pixel đi qua KL dù nhãn cache là ignore. Đây là cận dưới từ độ phủ, không phải phép đo trực tiếp giao hai mask. Padding/cutout vẫn bị loại. Điều này làm yếu ý nghĩa của việc bỏ nhãn không chắc chắn khỏi CE, nhưng chưa có ablation để kết luận mức ảnh hưởng.
- SAM nhận lớp, prompt và vùng sửa từ teacher; cần giữ core recall ≥0,9 và không được đổi reliable foreground sang lớp foreground khác. Nó giúp hình dạng nhưng khó sửa lỗi semantic đã được teacher tin chắc. Erode/ignore 3 pixel và lọc thành phần nhỏ có nguy cơ mất cấu trúc mảnh; chưa đo trực tiếp tác động đó trên từng cấu trúc.
- So với reliable cache, SAM thêm 31.396.881 pixel foreground và trim 1.950.371 pixel foreground về background; thay đổi 3,15% tổng pixel. Độ phủ hoặc predicted quality không chứng minh độ chính xác.
- Mở hai block chỉ tăng 0,0317 điểm so với decoder_control; SAM thêm 0,0332 điểm so với last2_consistency. Các thay đổi chung của decoder_control đã gây giảm 0,1816 điểm so với baseline; nhánh này cùng lúc đổi nhãn, augmentation và loss so với baseline, nên chưa tách riêng được thành phần gây giảm.
- Vòng consensus cũ đã lệch baseline tại 5k (50,9592 so với 51,9176 mIoU val trong lúc train), trước khi gate kích hoạt tại 8k. Không thể quy toàn bộ mức giảm cuối lịch cho consensus. Cần đối chứng cùng runtime trước khi so gate.

### Độ bất định của chênh lệch nhỏ

Paired whole-image bootstrap, 1.000 lần, seed 0; mỗi lần tính lại global dataset mIoU từ confusion matrix. Khoảng percentile 95% chỉ phản ánh lấy mẫu lại ảnh val của model cố định, không gồm biến động seed train hay kiểm tra độc lập.

| So sánh | Khoảng 95% cho Δ mIoU (điểm) |
|---|---|
| decoder_control so với baseline | [-0.6633; +0.2993] |
| last2_consistency so với baseline | [-0.6495; +0.3268] |
| last2_sam so với baseline | [-0.6436; +0.4096] |
| last2_minus_decoder | [-0.0812; +0.1499] |
| sam_minus_last2 | [-0.0715; +0.1417] |

Tất cả khoảng đều chứa 0. Điểm quan sát chưa thắng baseline; dữ liệu một seed cũng chưa đủ khẳng định chênh lệch rất nhỏ là hiệu ứng ổn định.

### Hạn chế trong thiết kế và việc cần kiểm chứng trước một vòng train mới

Các kiểm thử xác nhận forward/gradient, checkpoint và phần đóng băng; chúng không xác nhận pseudo-label chính xác hơn. Những vòng trước thiếu phép đo precision/recall của nhãn mới và thiếu ablation riêng cho KL-mask, strong augmentation, nguồn nhãn.

Ưu tiên kế tiếp là đo chất lượng và độ phủ pseudo-label theo lớp trên tập chẩn đoán val (không dùng pixel GT để train), rồi thử từng biến trên cùng runtime: giữ/loại KL ở pixel bị cache từ chối; weak so với strong augmentation; nhãn teacher so với SAM với kiểm soát precision. Chỉ mở rộng lịch train sau khi xác nhận tín hiệu nhãn hoặc đối chứng có lợi. Chưa có bằng chứng chỉ cần train lâu hơn, tăng LR hoặc dùng model lớn hơn sẽ giải quyết vấn đề.

![IoU từng lớp so với baseline](diagnostics/class_iou_deltas.png)

[Số đo và bootstrap](diagnostics/diagnosis.json) · [Precision/recall từng lớp](diagnostics/class_precision_recall.csv) · [Mã tái tính](../../tools/diagnose_representation_results.py) · [Chẩn đoán vòng consensus](../pcre_repair_20261119/REPAIR_NOTES.md).

## Hướng nghiên cứu tiếp theo từ paper

Đây là ý tưởng đề xuất sau khi tra cứu, chưa được cài đặt/đo mIoU. Không lấy kết quả trong paper làm dự báo mức tăng cho PCRE. Ưu tiên giảm false positive có bằng chứng âm và bảo vệ logits baseline trước khi mở rộng nhãn.

| Ưu tiên | Hướng | Cơ sở nghiên cứu | Thích nghi cần thử trên PCRE |
|---|---|---|---|
| 1 | Positive/negative prototype bank | [FPR, ICCV 2023](https://www4.comp.polyu.edu.hk/~cslzhang/paper/ICCV23-FPR.pdf) | Lấy false activation của lớp vắng mặt trước khi lọc image tags; dùng feature âm để kiểm tra vùng foreground đáng ngờ. Tránh negatives từ bộ phận dùng chung giữa các lớp. |
| 2 | Nhánh residual giữ baseline | Đề xuất thiết kế riêng cho checkpoint hiện có | Đóng băng baseline; thêm residual head zero-init và gate chỉ sửa vùng bất định/mâu thuẫn với negative bank. Gate inference không dùng image tags hoặc GT. Trùng logits lúc khởi tạo không bảo đảm không hồi quy sau train hay sau CRF. |
| 3 | Học phủ định trên vùng bất định | [U2PL, CVPR 2022](https://arxiv.org/abs/2203.03884) | Không ép pixel ignore theo argmax teacher qua KL. Chỉ dùng tín hiệu loại lớp khi có căn cứ; chưa đủ căn cứ thì ignore. U2PL gốc là semi-supervised, không sao chép nhánh pixel-GT vào WSSS. |
| 4 | Feature-distribution reliability | [BRNF, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Qiu_Bias-Resilient_Weakly_Supervised_Semantic_Segmentation_Using_Normalizing_Flows_ICCV_2025_paper.pdf) | Trước tiên thử GMM nhỏ trên frozen features từ seed train, làm ablation kiểm tra reliability; đây không phải tái triển khai full normalizing-flow BRNF và vẫn có nguy cơ kế thừa bias teacher. |
| 5 | Chi tiết không gian/tần số | [FFR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_FFR_Frequency_Feature_Rectification_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf) | Một nhánh feature tầng sớm/detail refinement; không tăng mọi cạnh RGB thành ranh giới lớp. Cần chẩn đoán boundary trước vì precision giảm chưa chứng minh lỗi do tần số cao. |
| Tuyến riêng | Semantic teacher ngoài baseline | [WeCLIP, CVPR 2024](https://arxiv.org/abs/2406.11189) | CLIP dense semantic cues hỗ trợ kiểm tra vùng đáng ngờ trong train; inference chỉ một student. Khai báo thêm image-text pretraining. |

[DuPL, CVPR 2024](https://arxiv.org/abs/2403.11184) là đối chiếu cho confirmation bias: diversity và cross-supervision khác với lấy hai bản copy cùng teacher. Chưa ưu tiên full hai student do chi phí và kết quả ensemble hiện tại tăng rất nhỏ.

Trình tự kiểm chứng: (1) residual control không sửa nhãn; (2) cùng control + negative bank; (3) thêm negative learning ở vùng bất định; (4) thử distribution gate riêng, tránh gộp mọi module. Cùng init, runtime, số bước và protocol val; đo precision/recall, retained/removed labels và IoU từng lớp, không chỉ độ phủ. Sàng lọc phát triển không thay thế đánh giá full val; kết quả tốt cần nhiều seed và đánh giá độc lập vì val đã được dùng phát triển nhiều vòng.

Mã công khai kiểm tra trong lần tra cứu: [FPR](https://github.com/mt-cly/FPR), [U2PL](https://github.com/Haochen-Wang409/U2PL), [DuPL](https://github.com/Wu0409/DuPL) có nguồn triển khai; [FFR](https://github.com/yay97/FFR) và [BRNF](https://github.com/DpDark/BRNF) chưa hiển thị đầy đủ mã train trong cây nguồn đã kiểm tra. Không mặc định các module này cắm vào PCRE được ngay.

Một kiểm tra tái hiện riêng: [PCRE chính thức](https://github.com/xxf011/WSSS-PCRE) báo VOC val 75,5%, còn baseline hiện tại 74,2243%. Script infer_seg_voc.py trong repo dùng CRF pos_w=1, trong khi protocol đối chứng hiện tại cố định pos_w=3. Cần đối chiếu checkpoint/cấu hình/ngân sách và preprocessing; khác biệt điểm chưa chứng minh lỗi. Nếu điều chỉnh protocol thì phải đánh giá lại baseline và mọi phương pháp, không tính riêng việc đổi inference là đóng góp học mới.
