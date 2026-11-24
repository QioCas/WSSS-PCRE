# Chạy lại PCRE baseline với checkpoint đầy đủ

Baseline VOC chạy mới từ pretrained ImageNet, không nạp weights của adaptive
hoặc run cũ. Chế độ `baseline` trong `scripts/train_voc_adaptive.py` giữ CAM
gốc, không áp dụng ngưỡng thích nghi. Các kết quả cũ được giữ nguyên.

```bash
python scripts/run_pcre_experiments.py --config configs/pcre_baseline_only.json --check
python scripts/run_pcre_experiments.py --config configs/pcre_baseline_only.json --detach
```

Chỉ launch một lần; lệnh detach tạo queue mới mỗi lần chạy. Supervisor chạy nền
và chờ GPU 0 có ít nhất 20.000 MiB VRAM trống qua 3 lần kiểm tra cách nhau
20 giây, tối đa 24 giờ. Sau đó chạy smoke baseline 4 bước, validation 2 ảnh,
lưu thử checkpoint; nếu thành công mới chạy baseline đầy đủ. Không dùng smoke
metrics làm kết quả nghiên cứu. Không dừng các tiến trình đang dùng GPU.

Seed 0, batch 4, crop 448, lr 3e-5, CAM scales 1.0/0.5/0.75/1.5, 20.000 bước.
Validation trên toàn bộ VOC val mỗi 5.000 bước. Cấu hình train khớp adaptive;
baseline này lưu checkpoint đầy đủ thường xuyên hơn (mỗi 5.000 bước).
Preflight kiểm tra dataset, dependencies, pretrained và ngân sách checkpoint;
mỗi lần lưu checkpoint giữ dự phòng 2 GiB đĩa trống.

Kết quả nằm trong `experiments/pcre_sat_<timestamp>_<id>/baseline/`.
Mỗi mốc 5.000, 10.000, 15.000, 20.000 gồm một cặp checkpoint:

- `checkpoints/model_iter_XXXXXXXX.pth`: weights dùng cho inference.
- `checkpoints/training_iter_XXXXXXXX.pth`: optimizer, lịch học, iteration,
  cls-token loss, trạng thái ngưỡng, RNG, sampler và cấu hình; tham chiếu file
  model cùng thư mục. Phải giữ cả hai file để resume.

Ví dụ resume trong thư mục kết quả mới:

```bash
WANDB_MODE=offline python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/train_voc_adaptive.py --experiment_mode baseline --spg 4 \
  --eval_iters 5000 --checkpoint_iters 5000 --log_iters 100 \
  --resume experiments/pcre_sat_JOB/baseline/checkpoints/training_iter_00010000.pth \
  --run_dir experiments/pcre_sat_JOB/baseline_resumed
```

RNG augmentation của DataLoader worker khởi động lại khi resume; không đảm bảo
tiếp tục khớp từng bit. Xem `EXPERIMENTS_ADAPTIVE.md` để biết cấu trúc log và
chi tiết checkpoint. W&B chạy offline.

Theo dõi queue mới nhất hoặc truyền đường dẫn queue cụ thể:

```bash
python scripts/pcre_experiment_status.py
python scripts/pcre_experiment_status.py experiments/pcre_sat_JOB
```

`waiting_for_gpu` nghĩa là chưa bắt đầu train; `smoke_running` là đang kiểm tra
nhanh; `training_running` là baseline đã khởi động. Tạo file `STOP` trong thư
mục queue để hủy khi đang chờ; file này không dừng trainer đã khởi động.
