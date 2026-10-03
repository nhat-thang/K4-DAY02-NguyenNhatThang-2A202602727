"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.

Mỗi lần chạy ghi vào <out_dir>/<exp_id>/seed<k>/:
    config.json, history.csv, summary.json, best.pt (checkpoint tốt nhất), last.pt (để resume),
    val_logits.npy (+ test_logits.npy nếu là chung kết), val_pred.csv
và ảnh <curves_dir>/<exp_id>_<desc>.png. Phiên Colab bị ngắt: gọi lại run(cfg) sẽ tiếp tục từ last.pt;
nếu summary.json đã có thì trả về luôn (skip_if_done=True).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# eval.py nằm ở gốc repo: <repo>/submissions/<mssv>_<ten>/code/train.py -> parents[2]
for _p in (_HERE.parents[2], _HERE.parent):
    if (_p / "eval.py").exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dataset as D  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
from eval import compute_metrics, save_predictions  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float | None = None
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | flipv | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    preload: bool = True              # nạp bytes JPEG vào RAM
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer: str = "adamw"          # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- hành vi ---
    skip_if_done: bool = True
    measure_latency: bool = True      # độ trễ sơ bộ batch 1 (Bước 1)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    desc = cfg.desc or cfg.backbone
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}.png"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Cố định random, numpy, torch (CPU + CUDA). Seed cho worker ở dataset.seed_worker.

    Mặc định cudnn.benchmark=True (nhanh hơn, không tất định từng bit): hai lần chạy cùng seed
    có thể lệch nhỏ. Đặt deterministic=True nếu cần tái lập tuyệt đối (chậm hơn).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD+momentum) với các nhóm tham số của model.param_groups."""
    groups = M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer không hợp lệ: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật THEO BƯỚC (mỗi iteration)."""
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """W_ema <- d * W_ema + (1 - d) * W cho MỌI phần tử float trong state_dict (gồm cả running
    mean/var của BN, giống timm ModelEmaV2); buffer nguyên (num_batches_tracked) chép thẳng.
    Đánh giá bằng `ema.module` (bản sao riêng)."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1 - self.decay)
            else:
                v.copy_(src)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch. Trả về train_loss, train_acc (vô nghĩa khi có Mixup/CutMix), lr cuối, lrs theo bước."""
    M.set_train_mode(model)
    tot_loss, tot_correct, n, lrs = 0.0, 0, 0, []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            if cfg.mix:
                x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
                logits = model(x)
                loss = L.mixed_loss(criterion, logits, targets)
            else:
                logits = model(x)
                loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} (NaN/Inf)")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_correct += (logits.argmax(1) == y).sum().item()
        n += bs
        lrs.append(optimizer.param_groups[0]["lr"])
    return {"train_loss": tot_loss / n, "train_acc": tot_correct / n, "lr": lrs[-1], "lrs": lrs}


@torch.no_grad()
def evaluate(model, loader, criterion=None, device=None, amp: bool = True):
    """model.eval() + inference_mode. Trả về (filenames, y_true[N], logits[N, 9], loss CE).

    Loss val luôn là CE thường (không trọng số/làm mịn) để so sánh được giữa các cấu hình.
    """
    device = device or next(model.parameters()).device
    model.eval()
    names, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(x)
            outs.append(logits.float().cpu())
            ys.append(y)
            names.extend(f)
    logits = torch.cat(outs)
    y = torch.cat(ys)
    loss = F.cross_entropy(logits, y).item()
    return names, y.numpy(), logits.numpy(), loss


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics_from_logits(y, logits) -> dict:
    probs = softmax_np(logits.astype(np.float64))
    return compute_metrics(y, probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lrs: list[float] | None = None) -> None:
    """3 panel: loss train/val · macro-F1 và top-1 val (+ acc train) · LR theo bước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax[1].plot(ep, [h["val_top1"] for h in history], "s-", label="val top-1")
    ax[1].plot(ep, [h["train_acc"] for h in history], "^--", alpha=0.6, label="train acc (batch đã aug)")
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))
    ax[1].axvline(best["epoch"], color="gray", ls=":", label=f"best ep {best['epoch']} F1={best['val_macro_f1']:.4f}")
    ax[1].set(xlabel="epoch", ylabel="metric", title="Val metrics")
    if lrs:
        ax[2].plot(np.arange(1, len(lrs) + 1), lrs, label="LR (group 0: backbone)")
        ax[2].set(xlabel="bước", ylabel="LR", title="Lịch LR (warmup + cosine)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def env_info() -> dict:
    import timm
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình, chọn checkpoint theo macro-F1 VAL (hoà lấy epoch sớm hơn), lưu mọi thứ.

    Test chỉ được đánh giá khi cfg.save_test_predictions=True (Bước 4), đúng MỘT lần, sau khi
    đã chốt checkpoint bằng val.
    """
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    if cfg.skip_if_done and (rd / "summary.json").exists():
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã xong, đọc summary.json")
        return json.loads((rd / "summary.json").read_text(encoding="utf-8"))

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (rd / "config.json").write_text(json.dumps({**dataclasses.asdict(cfg), "env": env_info()}, indent=2),
                                    encoding="utf-8")

    # 2. dữ liệu
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)
    tf_train = D.build_transforms(True, cfg.img_size, cfg.aug)
    tf_eval = D.build_transforms(False, cfg.img_size)
    train_loader = D.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size, True, cfg.sampler,
                                 cfg.num_workers, cfg.seed, cfg.preload)
    val_loader = D.make_loader(val_df, cfg.images_dir, tf_eval, cfg.batch_size * 2, False, None,
                               cfg.num_workers, cfg.seed, cfg.preload)

    # 4. mô hình, loss, tối ưu
    model = M.build_model(cfg.backbone, True, D.NUM_CLASSES, cfg.drop_rate, cfg.init, cfg.drop_path_rate).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    params_m = M.count_params(model)
    try:
        gmacs = M.count_gmacs(model, cfg.img_size)
    except Exception as e:  # không để việc đếm FLOPs làm hỏng lần chạy
        print("Không đếm được GMAC:", e)
        gmacs = float("nan")
    counts = np.bincount(train_df["Label"].to_numpy(), minlength=D.NUM_CLASSES)
    criterion = L.build_criterion(cfg.loss, smoothing=cfg.label_smoothing or 0.1, gamma=cfg.focal_gamma,
                                  weight=L.class_weights(counts, cfg.class_weight_beta or 0.0)).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    history, all_lrs, start_epoch = [], [], 1
    best = {"f1": -1.0, "epoch": 0}
    if (rd / "last.pt").exists():  # resume sau khi phiên bị ngắt
        ck = torch.load(rd / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema") is not None:
            ema.module.load_state_dict(ck["ema"])
        history, all_lrs, best = ck["history"], ck["lrs"], ck["best"]
        start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["rng_cpu"].cpu())
        np.random.set_state(ck["rng_np"])
        random.setstate(ck["rng_py"])
        print(f"[{cfg.exp_id} seed{cfg.seed}] resume từ epoch {start_epoch}")

    # 5. vòng epoch
    for epoch in range(start_epoch, cfg.epochs + 1):
        _sync(device)
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        _sync(device)
        t_train = time.perf_counter() - t0
        eval_model = ema.module if ema is not None else model
        _, yv, lv, val_loss = evaluate(eval_model, val_loader, device=device, amp=cfg.amp)
        mv = metrics_from_logits(yv, lv)
        row = {"epoch": epoch, "train_loss": tr["train_loss"], "train_acc": tr["train_acc"], "lr": tr["lr"],
               "val_loss": val_loss, "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"],
               "val_bal_acc": mv["balanced_acc"], "val_ece": mv["ece"], "epoch_time_s": t_train}
        if ema is not None:  # để so sánh: cùng epoch nhưng trọng số thường
            _, yr, lr_, _ = evaluate(model, val_loader, device=device, amp=cfg.amp)
            row["val_macro_f1_raw"] = metrics_from_logits(yr, lr_)["macro_f1"]
        history.append(row)
        all_lrs.extend(tr["lrs"])
        print(f"[{cfg.exp_id} s{cfg.seed}] ep {epoch:2d}/{cfg.epochs} loss {tr['train_loss']:.4f} "
              f"val_loss {val_loss:.4f} val_F1 {mv['macro_f1']:.4f} val_top1 {mv['top1']:.4f} ({t_train:.0f}s)")
        if mv["macro_f1"] > best["f1"]:  # strict > : hoà giữ epoch sớm hơn
            best = {"f1": mv["macro_f1"], "epoch": epoch}
            torch.save(eval_model.state_dict(), rd / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.module.state_dict() if ema is not None else None,
                    "history": history, "lrs": all_lrs, "best": best, "epoch": epoch,
                    "rng_cpu": torch.get_rng_state(), "rng_np": np.random.get_state(),
                    "rng_py": random.getstate()}, rd / "last.pt")

    # 6. nạp checkpoint tốt nhất, lưu dự đoán val
    final_model = ema.module if ema is not None else model
    final_model.load_state_dict(torch.load(rd / "best.pt", map_location=device))
    names_v, yv, lv, _ = evaluate(final_model, val_loader, device=device, amp=cfg.amp)
    np.save(rd / "val_logits.npy", lv)
    np.save(rd / "val_labels.npy", yv)
    pv = softmax_np(lv.astype(np.float64))
    save_predictions(rd / "val_pred.csv", names_v, yv, pv)
    mv = compute_metrics(yv, pv.argmax(1), pv)

    # 7. test: CHỈ ở chung kết, đúng một lần
    if cfg.save_test_predictions:
        test_loader = D.make_loader(test_df, cfg.images_dir, tf_eval, cfg.batch_size * 2, False, None,
                                    cfg.num_workers, cfg.seed, cfg.preload)
        names_t, yt, lt, _ = evaluate(final_model, test_loader, device=device, amp=cfg.amp)
        np.save(rd / "test_logits.npy", lt)
        np.save(rd / "test_labels.npy", yt)
        (rd / "test_filenames.json").write_text(json.dumps(names_t), encoding="utf-8")
        save_predictions(pred_path(cfg, "test"), names_t, yt, softmax_np(lt.astype(np.float64)))
        save_predictions(pred_path(cfg, "val"), names_v, yv, pv)
    (rd / "val_filenames.json").write_text(json.dumps(names_v), encoding="utf-8")

    # 8. log, biểu đồ, tóm tắt
    import pandas as pd
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lrs.npy", np.asarray(all_lrs))
    plot_curves(history, curve_path(cfg),
                f"{cfg.exp_id} · {cfg.backbone} · {cfg.desc or cfg.init} · seed {cfg.seed}", all_lrs)

    lat = {}
    if cfg.measure_latency and device.type == "cuda":
        from benchmark import latency_report
        lat = latency_report(final_model, 1, cfg.img_size, "fp32", "cuda", warmup=10, iters=50)

    epoch_times = [h["epoch_time_s"] for h in history]
    f1 = mv["f1"]
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": model.weight_tag,
        "init": cfg.init, "params_M": params_m, "gmacs": gmacs, "img_size": cfg.img_size, "epochs": cfg.epochs,
        "best_epoch": best["epoch"], "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"],
        "val_bal_acc": mv["balanced_acc"], "val_ece": mv["ece"],
        "val_f1_chinee": float(f1[0]), "val_f1_snake": float(f1[7]),
        "val_f1_per_class": [float(v) for v in f1],
        "train_time_per_epoch_s": float(np.mean(epoch_times)),
        "latency_b1_p50_ms": lat.get("p50"), "latency_b1_p95_ms": lat.get("p95"),
        "curve": str(curve_path(cfg)), "env": env_info(),
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (rd / "last.pt").unlink(missing_ok=True)  # đã xong, bỏ checkpoint resume cho đỡ chỗ
    return summary


def load_config(rd: str | Path) -> Config:
    raw = json.loads((Path(rd) / "config.json").read_text(encoding="utf-8"))
    names = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in names})


def load_checkpoint(rd: str | Path, device=None):
    """Dựng lại model từ <run_dir>/config.json + best.pt (không tải trọng số ImageNet). Trả về (model, cfg)."""
    cfg = load_config(rd)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.build_model(cfg.backbone, False, D.NUM_CLASSES, cfg.drop_rate, "finetune", cfg.drop_path_rate)
    model.load_state_dict(torch.load(Path(rd) / "best.pt", map_location="cpu"))
    return model.to(device).eval(), cfg


def alias_run(src: Config, dst: Config) -> dict:
    """Dùng lại một lần chạy đã có cho exp_id khác khi HAI CẤU HÌNH TRÙNG NHAU (chỉ khác exp_id/desc),
    ví dụ T00 seed0 = B01 seed0. Chép thư mục chạy, vẽ lại đường cong với tên mới, ghi chú `alias_of`."""
    import shutil
    a, b = dataclasses.asdict(src), dataclasses.asdict(dst)
    ignore = {"exp_id", "desc", "save_test_predictions", "measure_latency", "skip_if_done"}
    diff = {k for k in a if k not in ignore and a[k] != b[k]}
    if diff:
        raise ValueError(f"Không alias được: cấu hình khác nhau ở {sorted(diff)}")
    s, d = run_dir(src), run_dir(dst)
    if not (s / "summary.json").exists():
        raise FileNotFoundError(f"{s} chưa chạy xong")
    if not (d / "summary.json").exists():
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)
        summary = json.loads((s / "summary.json").read_text(encoding="utf-8"))
        summary.update(exp_id=dst.exp_id, alias_of=f"{src.exp_id}/seed{src.seed}", curve=str(curve_path(dst)))
        (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        cfg_json = json.loads((d / "config.json").read_text(encoding="utf-8"))
        cfg_json.update(exp_id=dst.exp_id, desc=dst.desc, alias_of=summary["alias_of"])
        (d / "config.json").write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")
        import pandas as pd
        hist = pd.read_csv(d / "history.csv").to_dict("records")
        lrs = np.load(d / "lrs.npy").tolist() if (d / "lrs.npy").exists() else None
        plot_curves(hist, curve_path(dst), f"{dst.exp_id} · {dst.backbone} · {dst.desc or dst.init} · seed {dst.seed}"
                    f" (cùng lần chạy {summary['alias_of']})", lrs)
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- kiểm tra pipeline (GUIDE 1.3)
def initial_loss(cfg: Config, n_batches: int = 3) -> float:
    """Loss CE ban đầu của model với head mới, trên vài batch val; kỳ vọng ≈ ln 9 = 2.197."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_df, _ = D.load_split(cfg.labels_dir, cfg.fold)
    loader = D.make_loader(val_df.sample(frac=1, random_state=0), cfg.images_dir,
                           D.build_transforms(False, cfg.img_size), cfg.batch_size, False, num_workers=0)
    model = M.build_model(cfg.backbone, True, D.NUM_CLASSES, init=cfg.init).to(device).eval()
    losses = []
    with torch.inference_mode():
        for i, (x, y, _) in enumerate(loader):
            if i >= n_batches:
                break
            losses.append(F.cross_entropy(model(x.to(device)).float(), y.to(device)).item())
    return float(np.mean(losses))


def overfit_one_batch(cfg: Config, n: int = 16, steps: int = 60) -> list[float]:
    """Overfit n ảnh train (không augmentation) — loss phải về gần 0."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, _, _ = D.load_split(cfg.labels_dir, cfg.fold)
    sub = train_df.groupby("Label", group_keys=False).head(2).head(n)
    ds = D.DeepWeedsDataset(sub, cfg.images_dir, D.build_transforms(False, cfg.img_size))
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    y = torch.tensor(ds.labels, device=device)
    model = M.build_model(cfg.backbone, True, D.NUM_CLASSES, init=cfg.init).to(device)
    opt = torch.optim.AdamW(M.param_groups(model, cfg.lr_backbone * 10, cfg.lr_head, 0.0))
    hist = []
    model.train()
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return hist


# ----------------------------------------------------------------------------- CLI
def _cast(value: str, type_str: str):
    if value.lower() in ("none", "null") and "None" in type_str:
        return None
    base = type_str.replace("| None", "").replace("None |", "").strip()
    if base == "bool":
        if value.lower() in ("1", "true", "yes", "y"):
            return True
        if value.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"giá trị bool không hợp lệ: {value}")
    if base == "int":
        return int(value)
    if base == "float":
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict đã ép kiểu theo field của Config."""
    fields = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' phải có dạng KEY=VALUE")
        k, v = pair.split("=", 1)
        if k not in fields:
            raise KeyError(f"'{k}' không có trong Config. Các key hợp lệ: {sorted(fields)}")
        out[k] = _cast(v, fields[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
