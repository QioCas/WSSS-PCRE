# Xem kết quả inference VOC val

Hai checkpoint 20k được đánh giá cùng scales 1.0, 1.25, 1.5, horizontal flip và công thức tổng hợp của tools/infer_seg_voc.py.
DenseCRF: 10 vòng; pos_xy_std=2, pos_w=3, bi_xy_std=140, bi_rgb_std=1, bi_w=7.

| Kết quả mIoU (%) | Baseline | Adaptive | Δ adaptive − baseline (điểm) |
|---|---:|---:|---:|
| Trước CRF | 79.4113 | 79.0305 | -0.3807 |
| Sau CRF | 78.9703 | 78.6062 | -0.3641 |

## Cách xem ảnh

1. Trong VS Code/JupyterLab, mở thư mục kết quả này.
2. Mở comparisons/crf/<image_id>.png để xem 4 cột: ảnh gốc → ground truth → baseline → adaptive.
3. Mở comparisons/raw/<image_id>.png để xem cùng ảnh trước CRF.
4. Mở legend.png để xem tên lớp tương ứng từng màu. Vùng xám trong GT là ignore (255), không tính điểm.
5. overview_crf.png / overview_raw.png: 3 ảnh adaptive giảm mạnh nhất, rồi 3 ảnh tăng mạnh nhất theo chênh lệch image mIoU. Đây là các trường hợp cực trị, không đại diện toàn bộ val.
6. Mở per_image_comparison.csv và sắp xếp crf_delta_pp để tìm các ảnh cần kiểm tra; cột classes liệt kê lớp foreground trong GT.

Điểm trên từng ảnh chỉ trung bình trên các lớp có trong GT ảnh đó. mIoU toàn bộ 1.449 ảnh tính từ confusion matrix cộng dồn, không lấy trung bình điểm từng ảnh.

## Các file

- baseline|adaptive/raw|crf/color/*.png: mask màu ở độ phân giải ảnh gốc.
- baseline|adaptive/raw|crf/masks/*.png: mask ID uint8, giá trị 0..20. Mask ID có thể trông đen khi mở bằng trình xem ảnh thông thường; dùng thư mục color hoặc comparisons.
- baseline|adaptive/metrics.json: đầy đủ IoU, precision, recall theo lớp và thông tin checkpoint.
- per_class_comparison.csv: so sánh IoU 21 lớp.
- summary.json: bảng mIoU tổng.
- inference.log, progress.json: nhật ký và trạng thái chạy.

Đã kiểm tra lại toàn bộ mask: đủ ảnh, đúng kích thước, nhãn 0..20 và confusion matrix khớp các metrics đã ghi.
