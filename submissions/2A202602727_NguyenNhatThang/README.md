# Lab Day 2 — DeepWeeds · Nguyễn Nhật Thắng · 2A202602727

Bài làm cá nhân Lab Day 2 (Track 4): so sánh backbone, công thức huấn luyện và phương pháp suy luận trên DeepWeeds, fold 0.
Đề bài, quy tắc và thang điểm: [`README.md`](../../README.md), [`GUIDE.md`](../../GUIDE.md), [`RUBRIC.md`](../../RUBRIC.md) ở gốc repo.

## Trạng thái hiện tại (cập nhật 2026-10-05)

| Bước | Trạng thái |
|---|---|
| Chuẩn bị (fork, thư mục bài làm, GPU, dữ liệu + MD5, nơi lưu bền) | ✅ Xong. Đã chạy trên Colab T4 (2026-10-03): 17.509 ảnh, MD5 OK |
| Code `code/` (hoàn thiện mọi `TODO` của `starter/`) | ✅ Đã viết. Trên Colab: `test_code` 17/17 và `tests/` của repo 38/38 đều OK |
| Bước 0: EDA, kiểm tra pipeline | ⏳ Ô đã có trong notebook (thông báo/assert đã viết lại rõ ràng 2026-10-05), cần chạy và lưu output |
| Bước 0: ngân sách GPU (đo 1 epoch mỗi backbone, lập kế hoạch) | ⏳ Cần chạy ô đo, sau đó **chốt số epoch (12 hay 10) và số backbone làm ablation (1 hay 2)** |
| Bước 1 → 5 | ⬜ Chưa chạy. Ô đã viết sẵn, nằm sau dòng "⛔ Hết Bước 0" trong notebook |

**Việc tiếp theo:** chạy notebook từ đầu tới "⛔ Hết Bước 0", xem bảng `budget_epoch_time.csv` / `budget_plan.csv`, chốt kế hoạch,
ghi quyết định vào mục *Nhật ký quyết định* bên dưới, rồi mới sang Bước 1.

## Môi trường

- **Google Colab miễn phí, GPU T4.** Colab miễn phí không có hạn mức GPU công bố, nên chạy nhiều phiên. `GPU_BUDGET_H` trong notebook là ngân sách tự đặt.
- Notebook: [`code/lab_day2.ipynb`](code/lab_day2.ipynb). Mở trên Colab: *File → Open notebook → GitHub* →
  `nhat-thang/K4-DAY02-NguyenNhatThang-2A202602727` → `submissions/2A202602727_NguyenNhatThang/code/lab_day2.ipynb`.
- Phiên bản thư viện (Colab, 2026-10-03): python 3.13.15 · torch 2.11.0+cu130 · torchvision 0.26.0+cu130 · timm 1.0.29 ·
  numpy 2.1.3 · pandas 2.2.3 · sklearn 1.6.1 · GPU Tesla T4 · 2 nhân CPU.
- Dữ liệu: `images.zip` (MD5 `b7b30f96d466fba86016aa5a26606e0f`) đặt ở Google Drive `MyDrive/K4-day02/images.zip`.
  Notebook chép zip về `/content/data`, kiểm tra MD5, giải nén, rồi tải `labels.csv`, `train/val/test_subset0.csv` từ GitHub của tác giả.
  Không có zip trên Drive thì notebook tải từ Zenodo.

## Cách chạy

1. *Runtime → Change runtime type → T4 GPU*.
2. Chạy **lần lượt** từng ô, đến hết bước đang làm. **Không bấm Run all** khi chưa muốn huấn luyện các bước sau.
3. Phiên bị ngắt: mở lại, chạy lại từ ô đầu. Mọi lần chạy ghi vào Drive:
   - lần chạy đã có `summary.json` thì được bỏ qua;
   - lần chạy đang dở thì tiếp tục từ `last.pt`;
   - bảng đo ngân sách cũng được cache.

Thứ tự ô trong notebook: cài đặt → tải dữ liệu → test code → Bước 0 (EDA, kiểm tra pipeline, ngân sách GPU) → ⛔ →
Bước 1 (backbone `B01–B06`) → Bước 2 (`T00–T11`) → Bước 3 (`I00–I08`) → Bước 4 (`F01` + mốc `T00`, 3 seed, test) →
Bước 5 (`results.xlsx`, chép sản phẩm vào thư mục này).

## Nơi lưu kết quả

| Ở đâu | Nội dung |
|---|---|
| Drive `MyDrive/K4-day02/runs/<exp_id>/seed<k>/` | `config.json`, `history.csv`, `summary.json`, `best.pt`, `last.pt`, `val_logits.npy` (+ `test_logits.npy` ở chung kết) |
| Drive `MyDrive/K4-day02/submission/` | `curves/`, `predictions/`, `figures/`, `results.xlsx`, `report_tables.md`, `eval_out/` |
| Drive `MyDrive/K4-day02/tables/` | CSV trung gian: `eda_split`, `budget_*`, `backbones`, `training`, `inference`, `latency`, `final`, `perclass` |
| Thư mục này (git) | Ô cuối notebook chép `curves/`, `predictions/`, `figures/`, `tables/`, `results.xlsx` vào đây. **Không commit** ảnh, zip, checkpoint (`.gitignore` đã chặn `*.pt`, `*.zip`, `data/`) |

## Cấu trúc code (`code/`)

| File | Nội dung |
|---|---|
| `dataset.py` | Đọc split fold 0, `check_split` (giao rỗng, hợp = 17.509, file tồn tại), transform (`basic`, `flipv`, `color`, `trivial`, `randaug`), `DeepWeedsDataset` (nạp trước bytes JPEG vào RAM), `make_loader` (sampler cân bằng) |
| `model.py` | `timm` backbone, đóng băng + giữ BN ở eval (`set_train_mode`), 4 nhóm tham số (không weight decay cho norm/bias), đếm params và GMAC (`FlopCounterMode` ÷ 2) |
| `losses.py` | CE, label smoothing (tự cài), focal, CE có trọng số / class-balanced, Mixup, CutMix (λ theo diện tích thực) |
| `train.py` | `Config` + **một** hàm `run(cfg)` cho mọi thí nghiệm: AMP, warmup + cosine theo bước, EMA, chọn checkpoint theo macro-F1 val (hoà lấy epoch sớm hơn), resume, vẽ đường cong. Thêm `alias_run` (dùng lại lần chạy trùng cấu hình), `load_checkpoint`, `initial_loss`, `overfit_one_batch`, `time_one_epoch` |
| `inference.py` | TTA lật / 5-crop / đa tỉ lệ, gộp xác suất vs logit, ensemble, temperature scaling (LBFGS trên log T), gộp BN vào conv |
| `benchmark.py` | Đo độ trễ: warmup 10, `cuda.synchronize`, p50/p95/p99, FP32 / AMP / FP16. **Không tính tiền xử lý** |
| `experiments.py` | Danh sách phương pháp suy luận `I00–I04`, `final_predict` (Bước 4: test đúng một lần mỗi seed, T khớp trên val), ghi `results.xlsx` |
| `test_code.py` | Kiểm tra tự viết: focal γ=0 ≡ CE, label smoothing, CutMix λ, gộp BN, temperature, scheduler, EMA, parse CLI. Chạy: `cd code && python -m unittest test_code -v` |

Chạy một thí nghiệm từ dòng lệnh: `python code/train.py --set exp_id=B01 backbone=resnet50 seed=0 images_dir=... labels_dir=...`

## Quy ước và quy tắc phải giữ

- **Chỉ val để chọn** backbone, siêu tham số, phương pháp suy luận, checkpoint, nhiệt độ T. **Test chạy đúng một lần mỗi seed** ở Bước 4 (`experiments.final_predict`).
  `final_predict` không chạy lại nếu file `predictions/*_test.csv` đã có. **Đừng xoá các file đó để chạy lại.**
- Không sửa `eval.py` và các CSV chia dữ liệu. Không gộp val vào train.
- `exp_id`: `B0x` backbone · `T0x` công thức huấn luyện · `I0x` suy luận · `F01` chung kết. Ảnh `curves/<exp_id>_<desc>.png`.
- Seed: ablation chạy seed 0; `T00` chạy seed 0, 1, 2 (để đo nhiễu); chung kết `F01` và mốc `T00` chạy seed 0, 1, 2.
- Lần chạy dùng lại bằng `alias_run` được ghi `alias_of` trong `summary.json`: `T00` seed 0 = backbone đã chọn ở Bước 1; `F01` seed 0 = cấu hình đã chốt ở Bước 2.
- Mọi số trong `results.xlsx` / báo cáo phải lấy từ lần chạy thật (log trên Drive), không chép số từ bài báo.

## Nhật ký quyết định

Ghi mỗi quyết định kèm số liệu val làm căn cứ.

| Ngày | Quyết định | Căn cứ |
|---|---|---|
| 2026-10-03 | Dùng Colab miễn phí, lưu bền trên Drive `MyDrive/K4-day02` | Hướng dẫn lab cho phép Colab/Kaggle |
| 2026-10-03 | Tiền xử lý val/test: Resize 256 → CenterCrop 224, chuẩn hoá ImageNet | GUIDE 1.4 |
| | **TODO:** số epoch (12 / 10) và số backbone làm ablation | `tables/budget_plan.csv` |

## Seed đã dùng

**TODO:** điền sau khi chạy (dự kiến: 0 cho sàng lọc; 0, 1, 2 cho `T00` và `F01`).
