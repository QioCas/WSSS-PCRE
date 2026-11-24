# Thực nghiệm PCRE với ngưỡng CAM thích nghi

Các file gốc và kết quả cũ được giữ nguyên. Trainer mới được sao chép từ
`scripts/train_voc.py` tại thời điểm thiết lập; validation mới sao chép từ
`engine/validatation_engine.py` và chỉ bổ sung trả về metrics.

## Phương pháp

Nguồn ý tưởng: FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning,
ICLR 2023, https://arxiv.org/abs/2205.07246.
Đây là adaptation của ý tưởng SAT cho CAM, **không phải triển khai đầy đủ
FreeMatch**. CAM là điểm activation chuẩn hóa, không phải xác suất được hiệu chuẩn.

Với mỗi pixel nằm trong crop hợp lệ, chỉ xét các lớp có nhãn cấp ảnh. Pixel có
điểm CAM lớn nhất <= 0.25 không đóng góp thống kê. Không dùng mask ground truth
để tính ngưỡng. Dataset gốc vẫn đọc mask như trước; phương pháp mới chỉ nhận CAM,
image-level tags và crop boxes.

- `g`: EMA của điểm CAM thắng trung bình trên các pixel được chọn.
- `q[c]`: EMA của điểm thắng trung bình cho từng lớp c. Lớp không xuất hiện trong
  batch không cập nhật; lớp chưa từng có pixel thắng giữ ngưỡng 0.7.
- `tau[c] = clamp(g * q[c] / max(q[seen]), 0.55, 0.85)`.
- EMA momentum 0.99. Khởi tạo g và q bằng 0.7.
- Warm-up 2.000 bước; nội suy từ ngưỡng 0.7 sang tau trong 1.000 bước kế tiếp.

Áp dụng `scaled_cam[c] = valid_cam[c] * 0.7 / tau[c]` **trước PAR** và dùng lại
hàm refine gốc. Trước refinement, đây tương ứng với ngưỡng cao tau[c] và ngưỡng
thấp `0.25 * tau[c] / 0.7`. Sau softmax và PAR, không thể diễn giải như chỉ thay
một phép threshold độc lập: phân phối cạnh tranh giữa các lớp cũng thay đổi.
Không clamp scaled CAM về [0, 1] vì sẽ làm mất ý nghĩa của phép rescale.

Chỉ nhãn giả của nhánh segmentation chính thay đổi. Các nhãn này cấp cho
segmentation loss và energy regularization như code gốc. Nhánh auxiliary/PTC,
prototype và CRME giữ cách tạo nhãn cũ trong chế độ mặc định của bộ thực nghiệm.
Không thay backbone, không thêm tham số mô hình học được, không thêm forward ViT.

## Các cấu hình

`configs/pcre_adaptive_experiments.json` xếp hai run tuần tự:

1. `adaptive`: PCRE + ngưỡng thích nghi có warm-up.
2. `baseline`: PCRE với CAM không rescale, bật lưu checkpoint và log mới.

Cùng seed 0, batch/GPU 4, crop 448, 20.000 iterations, lr 3e-5,
cam_scales 1.0/0.5/0.75/1.5, validation mỗi 5.000 bước trên toàn bộ VOC val.
Validation thưa hơn run cũ để giảm thời gian và dung lượng checkpoint; hai run
mới có cùng lịch validation. Log cũ là thông tin tham khảo, không thay thế đối
chứng mới khi kết luận tác dụng của phương pháp.

`configs/pcre_adaptive_ablation.json` chạy `adaptive_no_warmup` nếu còn thời gian.
Ablation này không tự động chạy trong queue hai run chính.

## Kiểm tra và chạy

Môi trường Python hiện tại đã có đủ dependency, gồm torch 2.10.0+cu128,
torchvision 0.25.0 và extension bilateralfilter. Không cài lại hoặc đổi package
đang dùng. Preflight ghi phiên bản chính xác của môi trường vào kết quả.

```bash
python -m unittest discover -s tests -p test_adaptive_experiment.py -v
python scripts/run_pcre_experiments.py --check
python scripts/run_pcre_experiments.py --detach
```

Mỗi lần launch tạo một thư mục `experiments/pcre_sat_<timestamp>_<id>/` mới.
Không launch thêm queue nếu queue trước vẫn đang chạy. Xem trạng thái:

```bash
python scripts/pcre_experiment_status.py
```

Supervisor chờ GPU 0 có ít nhất 16.000 MiB VRAM trống; không chặn theo utilization
trong 3 lần kiểm tra liên tiếp cách nhau 20 giây. Tối đa chờ 24 giờ cho mỗi lượt.
Supervisor không dừng tiến trình khác. Sau khi GPU trống, chạy smoke test 4 bước
với batch/crop/scales như run thật, bật ngay segmentation và adaptive, đi qua
cả hai nhánh trước/sau seg_iter, validation trên 2 ảnh và lưu checkpoint.
Smoke test thất bại thì queue dừng; không tự chạy full hoặc retry lỗi.
Smoke metrics không được đưa vào kết quả nghiên cứu.

Để hủy queue khi nó **đang chờ GPU**, tạo file `STOP` trong thư mục queue.
File STOP không được xử lý trong lúc một run train đang hoạt động.

## Checkpoint và theo dõi

Mỗi run có:

- `config.json`, `source_hashes.json`, `source_snapshot/`: cấu hình và mã nguồn.
- `train.log`, `training_metrics.jsonl`: loss, LR, ngưỡng từng lớp, tỷ lệ nhãn.
- `validation_metrics.jsonl`: mIoU segmentation, CAM và IoU/precision/recall từng lớp.
- `checkpoints.jsonl`: đường dẫn checkpoint, loại và số iteration.
- `checkpoints/model_iter_00005000.pth`, ...: weights lưu mỗi 5.000 bước.
- `checkpoints/training_iter_00010000.pth` và `training_iter_00020000.pth`:
  tham chiếu tới weights cùng thư mục, optimizer, global_step, adaptive EMA/observations, cls-token
  loss, RNG, vị trí sampler, cấu hình và best mIoU để tiếp tục train.

Tên checkpoint có iteration; file đã tồn tại sẽ gây lỗi, không bị ghi đè.
Checkpoint được ghi vào `.partial`, fsync rồi rename sau khi hoàn tất.
Không dọn checkpoint cũ. Bản `.partial` còn lại sau sự cố không được xem là
checkpoint hoàn chỉnh.
Preflight ước lượng dung lượng cho cả queue và smoke; mỗi lần lưu còn kiểm tra
đủ chỗ cho checkpoint và chừa lại 2 GiB. Dung lượng vẫn có thể thay đổi nếu
tiến trình khác ghi thêm dữ liệu trong lúc train.

`model_iter_*.pth` là state_dict với tiền tố `module.` như trainer gốc, dùng
cho các tool inference của repo. Checkpoint đầy đủ dùng `--resume`, ví dụ:

```bash
WANDB_MODE=offline python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/train_voc_adaptive.py --experiment_mode adaptive --spg 4 \
  --eval_iters 5000 --checkpoint_iters 10000 --log_iters 100 \
  --resume experiments/pcre_sat_JOB/adaptive/checkpoints/training_iter_00010000.pth \
  --run_dir experiments/pcre_sat_JOB/adaptive_resumed
```

Chỉ load checkpoint tin cậy do mình tạo. Resume khôi phục optimizer/LR, model,
EMA, RNG chính và vị trí trong sampler. RNG augmentation của DataLoader worker
khởi động lại, nên đây không phải continuation khớp từng bit. Run resumed phải
được ghi rõ trong báo cáo. Hiện chỉ hỗ trợ một GPU để tránh lệch thống kê/lưu.

W&B chạy offline; dữ liệu không tự upload. Queue có `environment_freeze.txt`,
`preflight.json`, `queue_config.json`, `queue_events.jsonl`, `supervisor.log`
và console log riêng cho từng run. Công cụ status chỉ đọc file, không thay đổi
kết quả. So sánh segmentation mIoU với segmentation mIoU; CAM là cột riêng.

Chưa có bằng chứng phương pháp tăng mIoU. Kết quả cần báo cáo cả trường hợp
không cải thiện, IoU từng lớp và đánh đổi độ phủ/nhiễu của nhãn giả.


### Dung lượng và checkpoint ghép cặp (v2)

Để không lưu trùng trọng số, training_iter_*.pth chứa optimizer, lịch học,
ngưỡng, RNG và tên model_iter_*.pth nằm cùng thư mục. Hai file tạo thành một
checkpoint đầy đủ. Khi di chuyển checkpoint để resume, phải giữ cả hai file
cùng thư mục. Weights model_iter_*.pth vẫn dùng trực tiếp cho inference.
