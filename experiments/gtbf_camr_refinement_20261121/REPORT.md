# Method 3: Geometric-Topological Boundary Fusion with Class-Adaptive Morphological Rectification (GTBF-CAMR) Report

**Ngày thực hiện:** 2026-11-21  
**Tập dữ liệu:** Pascal VOC 2012 Val (1.449 ảnh đầy đủ)  
**Tác vụ:** Weakly Supervised Semantic Segmentation (WSSS)  
**Đề tài tham chiếu:** Topic 01 (Mô hình SAM + CAM: cải thiện kết quả bài toán phân đoạn ngữ nghĩa)

---

## 1. Tóm tắt kết quả chính

* **Baseline (PCRE 20k iter + Multi-scale + DenseCRF):** `74.22%` mIoU
* **Phương pháp 3 (GTBF-CAMR):** **`75.85%`** mIoU
* **Mức tăng tịnh tiến ($\Delta$ mIoU):** <span style="color:green">**+1.63%**</span>
* **Thời gian xử lý:** `339.1s` trên GPU RTX 5090 cho toàn bộ 1.449 ảnh (~11 ảnh/giây).

---

## 2. Bảng kết quả chi tiết từng lớp (Per-class Comparison)

| Lớp (Class) | Baseline CRF IoU (%) | GTBF-CAMR IoU (%) | Chênh lệch ($\Delta$) | Precision (%) | Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **_background_** | 91.92 | **92.71** | <span style='color:green'>**+0.79%**</span> | 96.33 | 96.11 |
| **aeroplane** | 83.85 | **85.87** | <span style='color:green'>**+2.02%**</span> | 91.57 | 93.24 |
| **bicycle** | 45.29 | **45.09** | <span style='color:red'>-0.21%</span> | 51.76 | 77.76 |
| **bird** | 87.47 | **88.92** | <span style='color:green'>**+1.45%**</span> | 93.62 | 94.66 |
| **boat** | 68.40 | **69.64** | <span style='color:green'>**+1.24%**</span> | 75.00 | 90.70 |
| **bottle** | 79.41 | **80.74** | <span style='color:green'>**+1.32%**</span> | 90.29 | 88.41 |
| **bus** | 88.67 | **93.22** | <span style='color:green'>**+4.55%**</span> | 95.88 | 97.11 |
| **car** | 82.37 | **83.86** | <span style='color:green'>**+1.50%**</span> | 89.85 | 92.64 |
| **cat** | 91.24 | **93.33** | <span style='color:green'>**+2.09%**</span> | 97.33 | 95.79 |
| **chair** | 43.28 | **43.33** | <span style='color:green'>**+0.04%**</span> | 52.51 | 71.23 |
| **cow** | 85.85 | **87.19** | <span style='color:green'>**+1.34%**</span> | 98.87 | 88.07 |
| **diningtable** | 48.98 | **49.40** | <span style='color:green'>**+0.42%**</span> | 77.90 | 57.45 |
| **dog** | 86.25 | **87.84** | <span style='color:green'>**+1.59%**</span> | 94.28 | 92.78 |
| **horse** | 86.96 | **89.80** | <span style='color:green'>**+2.84%**</span> | 94.57 | 94.67 |
| **motorbike** | 75.66 | **77.92** | <span style='color:green'>**+2.27%**</span> | 88.62 | 86.58 |
| **person** | 78.75 | **81.90** | <span style='color:green'>**+3.15%**</span> | 96.01 | 84.79 |
| **pottedplant** | 67.19 | **68.30** | <span style='color:green'>**+1.12%**</span> | 78.88 | 83.59 |
| **sheep** | 89.82 | **92.28** | <span style='color:green'>**+2.46%**</span> | 98.03 | 94.03 |
| **sofa** | 52.56 | **53.28** | <span style='color:green'>**+0.72%**</span> | 71.30 | 67.83 |
| **train** | 60.35 | **63.32** | <span style='color:green'>**+2.97%**</span> | 67.01 | 92.00 |
| **tvmonitor** | 64.44 | **64.92** | <span style='color:green'>**+0.48%**</span> | 72.70 | 85.85 |
| **Trung bình (mIoU)** | **74.22** | **75.85** | <span style='color:green'>**+1.63%**</span> | -- | -- |

---

## 3. Phân tích đột phá kỹ thuật của GTBF-CAMR

1. **Phân tách 3 nhánh hình học thích ứng theo nhóm lớp (Tripartite Geometric Branching):**
   - **Slender & Hollow (`bicycle`, `chair`):** Tuyệt đối không sử dụng Box Prompt nhằm tránh hiện tượng SAM nhận nhầm khoảng trống giữa nan hoa và chân ghế là đối tượng. Chỉ sử dụng prompt điểm trên trục xương sống (Medial-axis points).
   - **Composite & Articulated (`person`, `diningtable`, `sofa`):** Sử dụng đa điểm trục xương sống kết hợp cơ chế bảo toàn ngữ cảnh phi xói mòn (Non-Erosive Context Protection).
   - **Solid & Rigid (`bus`, `train`, `car`, v.v.):** Áp dụng hộp bao có đệm biên an toàn kết hợp cắt tỉa ranh giới nền thừa.
2. **Cổng kiểm soát gradient màu (Color-Gradient Edge Gating):**
   - Khi SAM đề xuất mở rộng sang vùng nền cho các đối tượng thanh mảnh, thuật toán kiểm tra độ sai biệt màu Euclidean ($\Delta RGB < 50$) so với màu trung tâm của vật thể. Nếu màu sắc sai lệch quá lớn (thuộc về nền phía sau khe nan hoa), vùng đó sẽ bị triệt tiêu ngay lập tức.
3. **Vá lỗ rỗng tô pô (Topological Hole Healing):**
   - Tự động đóng kín các lỗ rỗng nội bộ (cửa sổ xe bus/tàu hỏa, decal, bóng đổ) trên toàn bộ các lớp thể rắn, hoàn thiện độ đồng nhất khối của vật thể.
