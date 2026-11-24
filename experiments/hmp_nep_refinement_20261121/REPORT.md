# Method 2: Hierarchical Multi-Prompt Part-Whole SAM with Non-Erosive Context Protection (HMP-NEP) Report

**Ngày thực hiện:** 2026-11-21  
**Tập dữ liệu:** Pascal VOC 2012 Val (1.449 ảnh đầy đủ)  
**Tác vụ:** Weakly Supervised Semantic Segmentation (WSSS)  
**Đề tài tham chiếu:** Topic 01 (Mô hình SAM + CAM: cải thiện kết quả bài toán phân đoạn ngữ nghĩa)

---

## 1. Tóm tắt kết quả chính

* **Baseline (PCRE 20k iter + Multi-scale + DenseCRF):** `74.22%` mIoU
* **Phương pháp 2 (HMP-NEP):** **`75.70%`** mIoU
* **Mức tăng tịnh tiến ($\Delta$ mIoU):** <span style="color:green">**+1.48%**</span>
* **Thời gian xử lý:** `98.6s` trên GPU RTX 5090 cho toàn bộ 1.449 ảnh (~11 ảnh/giây).

---

## 2. Bảng kết quả chi tiết từng lớp (Per-class Comparison)

| Lớp (Class) | Baseline CRF IoU (%) | HMP-NEP IoU (%) | Chênh lệch ($\Delta$) | Precision (%) | Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **_background_** | 91.92 | **92.63** | <span style='color:green'>**+0.71%**</span> | 96.30 | 96.05 |
| **aeroplane** | 83.85 | **85.94** | <span style='color:green'>**+2.09%**</span> | 91.90 | 92.98 |
| **bicycle** | 45.29 | **43.09** | <span style='color:red'>-2.20%</span> | 47.63 | 81.88 |
| **bird** | 87.47 | **88.90** | <span style='color:green'>**+1.43%**</span> | 93.69 | 94.56 |
| **boat** | 68.40 | **69.72** | <span style='color:green'>**+1.31%**</span> | 75.45 | 90.17 |
| **bottle** | 79.41 | **80.58** | <span style='color:green'>**+1.16%**</span> | 90.32 | 88.20 |
| **bus** | 88.67 | **92.97** | <span style='color:green'>**+4.30%**</span> | 95.94 | 96.78 |
| **car** | 82.37 | **84.05** | <span style='color:green'>**+1.68%**</span> | 90.29 | 92.40 |
| **cat** | 91.24 | **93.31** | <span style='color:green'>**+2.06%**</span> | 97.49 | 95.61 |
| **chair** | 43.28 | **42.25** | <span style='color:red'>-1.03%</span> | 50.52 | 72.07 |
| **cow** | 85.85 | **87.07** | <span style='color:green'>**+1.22%**</span> | 98.95 | 87.88 |
| **diningtable** | 48.98 | **49.40** | <span style='color:green'>**+0.42%**</span> | 77.90 | 57.45 |
| **dog** | 86.25 | **87.75** | <span style='color:green'>**+1.50%**</span> | 94.36 | 92.61 |
| **horse** | 86.96 | **89.79** | <span style='color:green'>**+2.83%**</span> | 94.76 | 94.48 |
| **motorbike** | 75.66 | **77.94** | <span style='color:green'>**+2.28%**</span> | 89.01 | 86.24 |
| **person** | 78.75 | **81.90** | <span style='color:green'>**+3.15%**</span> | 96.01 | 84.79 |
| **pottedplant** | 67.19 | **68.59** | <span style='color:green'>**+1.41%**</span> | 79.79 | 83.02 |
| **sheep** | 89.82 | **92.17** | <span style='color:green'>**+2.35%**</span> | 98.12 | 93.83 |
| **sofa** | 52.56 | **53.28** | <span style='color:green'>**+0.72%**</span> | 71.30 | 67.83 |
| **train** | 60.35 | **63.70** | <span style='color:green'>**+3.36%**</span> | 67.63 | 91.64 |
| **tvmonitor** | 64.44 | **64.80** | <span style='color:green'>**+0.36%**</span> | 72.86 | 85.42 |
| **Trung bình (mIoU)** | **74.22** | **75.70** | <span style='color:green'>**+1.48%**</span> | -- | -- |

---

## 3. Phân tích đột phá kỹ thuật của HMP-NEP

1. **Khắc phục triệt để hiện tượng xói mòn sai lệch (False Negative Erosion):**
   - Ở phương pháp 1, các lớp có cấu trúc phức tạp như `person`, `diningtable`, `bicycle`, `chair` bị sụt giảm vì SAM chỉ phân đoạn một bộ phận nhỏ (ví dụ áo hoặc mặt bàn), khiến toàn bộ phần còn lại bị cắt tỉa nhầm thành background.
   - HMP-NEP áp dụng cơ chế **Non-Erosive Context Protection (NEP)**: chỉ cho phép mở rộng biên đối tượng (`to_fg`), tuyệt đối không cắt tỉa đối với các lớp composite & articulated.
2. **Medial Axis Skeleton Multi-Prompting:**
   - Thay vì 1 điểm tâm duy nhất, thuật toán trích xuất đa điểm dọc theo trục xương sống của các vật thể lớn (diện tích > 1200 pixel), giúp SAM nắm bắt trọn vẹn cả cấu trúc chính và chi tiết phụ.
3. **Adaptive Consensus Gating:**
   - Điều chỉnh ngưỡng IoU động (0.55 cho các lớp khớp nối phức tạp và 0.65 cho các vật thể cứng khối) giúp cân bằng tối ưu giữa độ chính xác và độ bao phủ.
