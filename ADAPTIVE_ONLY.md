# Chỉ adaptive, chạy lại từ đầu

```bash
python scripts/run_adaptive_only.py --detach
```

Khởi tạo từ pretrained ImageNet như baseline, không nạp weights của run đã dừng.
Chỉ adaptive 20.000 bước; không có baseline trong hàng đợi. Tạo thư mục kết quả
mới, không xóa hoặc ghi đè kết quả cũ. Dataset dùng symlink `dataset/VOC2012`.

Launcher đợi ổ chứa kết quả có tối thiểu 10 GiB trống, sau đó preflight kiểm tra
lại ngân sách checkpoint. Không train nếu chưa đủ chỗ lưu. Chờ đĩa tối đa 24 giờ.
Khi qua preflight, chờ GPU 0 có >=16.000 MiB VRAM trống qua 3 lần kiểm tra cách
nhau 20 giây; utilization không chặn. Chạy smoke trước rồi adaptive.

Seed 0, batch 4, crop 448, lr 3e-5, 20.000 bước. Validation/weights mỗi 5.000 bước,
checkpoint đầy đủ dạng cặp mỗi 10.000 bước và khi kết thúc.

Theo dõi:

```bash
python scripts/pcre_experiment_status.py
```

`waiting_for_disk`: chưa train vì thiếu dung lượng lưu checkpoint.
`waiting_for_gpu`: đang chờ VRAM. `training_running`: trainer đã khởi động.
Tạo file `STOP` trong thư mục queue để hủy khi đang chờ; cách này không dừng run
train đã khởi động. Xem `EXPERIMENTS_ADAPTIVE.md` để biết cách lưu và resume.


Người dùng yêu cầu ngày 17/11 chạy ngay dù thiếu đĩa: cấu hình allow_low_disk=true,
skip_smoke=true. Bỏ chờ và ngân sách đĩa ban đầu, bỏ dự phòng 2 GiB khi lưu.
Vẫn cần đủ dung lượng thực tế cho file checkpoint; lỗi lưu có thể dừng run.
Chỉ adaptive từ đầu, không nạp weights run cũ, không chạy baseline.
