# SAM-CAM Consensus Semantic Refinement (SCSR) Report

**Ngày thực hiện:** 2026-11-20  
**Tập dữ liệu:** Pascal VOC 2012 Val (1.449 ảnh đầy đủ)  
**Tác vụ:** Weakly Supervised Semantic Segmentation (WSSS)  
**Đề tài tham chiếu:** Topic 01 (Mô hình SAM + CAM: cải thiện kết quả bài toán phân đoạn ngữ nghĩa)

---

## 1. Tóm tắt kết quả chính

* **Baseline (PCRE 20k iter + Multi-scale + DenseCRF):** `74.22%` mIoU
* **Phương pháp mới (SAM-CAM Consensus Semantic Refinement):** **`75.66%`** mIoU
* **Mức tăng tịnh tiến ($\Delta$ mIoU):** <span style="color:green">**+1.43%**</span> (Vượt ngưỡng 75.5% công bố trong bài báo gốc PCRE CVPR 2025).
* **Số lớp cải thiện:** 18 / 21 lớp.
* **Thời gian xử lý:** 132.9 giây trên GPU RTX 5090 cho toàn bộ 1.449 ảnh (tương đương ~11 ảnh/giây).

---

## 2. Bảng kết quả chi tiết từng lớp (Per-class Comparison)

| Lớp (Class) | Baseline CRF IoU (%) | SAM-CAM Refined IoU (%) | Chênh lệch ($\Delta$) | Precision (%) | Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **_background_** | 91.92 | **92.59** | <span style="color:green">+0.67%</span> | 95.81 | 96.49 |
| **aeroplane** | 83.85 | **85.65** | <span style="color:green">+1.80%</span> | 91.18 | 93.38 |
| **bicycle** | 45.29 | 44.62 | -0.67% | 50.63 | 78.99 |
| **bird** | 87.47 | **88.76** | <span style="color:green">+1.29%</span> | 93.55 | 94.54 |
| **boat** | 68.40 | **69.52** | <span style="color:green">+1.12%</span> | 75.30 | 90.05 |
| **bottle** | 79.41 | **80.95** | <span style="color:green">+1.53%</span> | 90.60 | 88.37 |
| **bus** | 88.67 | **93.32** | <span style="color:green">**+4.65%**</span> | 96.05 | 97.05 |
| **car** | 82.37 | **85.09** | <span style="color:green">**+2.73%**</span> | 91.60 | 92.29 |
| **cat** | 91.24 | **93.24** | <span style="color:green">**+2.00%**</span> | 97.53 | 95.49 |
| **chair** | 43.28 | **43.72** | <span style="color:green">+0.44%</span> | 53.94 | 69.75 |
| **cow** | 85.85 | **87.01** | <span style="color:green">+1.15%</span> | 99.01 | 87.77 |
| **diningtable** | 48.98 | 45.57 | -3.41% | 78.38 | 52.13 |
| **dog** | 86.25 | **87.49** | <span style="color:green">+1.24%</span> | 94.15 | 92.52 |
| **horse** | 86.96 | **90.06** | <span style="color:green">**+3.10%**</span> | 95.07 | 94.47 |
| **motorbike** | 75.66 | **77.66** | <span style="color:green">**+2.00%**</span> | 88.54 | 86.34 |
| **person** | 78.75 | **79.54** | <span style="color:green">+0.79%</span> | 97.77 | 81.01 |
| **pottedplant** | 67.19 | **68.84** | <span style="color:green">+1.66%</span> | 80.32 | 82.81 |
| **sheep** | 89.82 | **92.03** | <span style="color:green">**+2.21%**</span> | 98.19 | 93.62 |
| **sofa** | 52.56 | **53.48** | <span style="color:green">+0.92%</span> | 74.35 | 65.58 |
| **train** | 60.35 | **64.74** | <span style="color:green">**+4.39%**</span> | 69.09 | 91.15 |
| **tvmonitor** | 64.44 | **64.88** | <span style="color:green">+0.44%</span> | 72.60 | 85.92 |
| **Trung bình (mIoU)** | **74.22** | **75.66** | <span style="color:green">**+1.43%**</span> | -- | -- |

---

## 3. Cơ chế hoạt động của thuật toán SCSR

1. **Semantic Anchoring từ PCRE Baseline:**
   - Sử dụng dự đoán phân đoạn của mô hình PCRE tốt nhất làm neo ngữ nghĩa (class labels).
2. **Multi-Prompting cho SAM:**
   - Trích xuất từng connected component của vật thể với diện tích hợp lệ.
   - Định vị **Peak Point Prompt** bằng phép biến đổi khoảng cách Euclidean Distance Transform (`argmax(EDT)`), đưa điểm prompt vào sâu trong lõi vật thể.
   - Định vị **Box Prompt** bao quanh vùng kích hoạt với biên an toàn padding 8 pixel.
3. **Consensus Agreement Filtering:**
   - SAM sinh ra các candidate masks. Chỉ chấp nhận mask khi:
     * Chỉ số chất lượng dự đoán của SAM $Score > 0.85$.
     * Độ tương đồng giao diện với dự đoán gốc $IoU(M_{SAM}, M_{Baseline}) > 0.65$.
4. **Boundary Optimization & Trimming:**
   - Khôi phục phần biên bị thiếu (expand sang background).
   - Cắt gọt phần rò rỉ nền ngoài ranh giới hình học của SAM.
   - Không can thiệp sang các vùng của lớp đối tượng khác nhằm triệt tiêu xung đột đa lớp.
