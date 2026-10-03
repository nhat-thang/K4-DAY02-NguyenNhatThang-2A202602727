"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype, batch, độ phân giải, có/không gộp BN, phiên bản torch
  - Lựa chọn: KHÔNG tính tiền xử lý (decode JPEG, resize); chỉ đo forward của model trên tensor
    đã nằm sẵn trên GPU. Với TTA, đo cả phần tạo view (lật/crop) vì nó chạy trên GPU.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo `fn()` (mili-giây): warmup rồi bỏ; mỗi lần đo sync -> perf_counter -> fn -> sync."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "std": float(t.std(ddof=1)), "n": iters}


def _prepare(model, dtype: str, device: str):
    model = model.eval().to(device)
    if dtype == "fp16":
        model = model.half()
    return model


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False, views_fn=None,
                   name: str = "") -> dict:
    """Độ trễ forward với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half(), CHỈ dùng trên bản sao model).
    views_fn: nếu có (TTA), mỗi lần đo chạy model trên mọi view trả về bởi views_fn(x).
    """
    dev = torch.device(device)
    if dtype == "fp16":
        import copy
        model = copy.deepcopy(model)
    model = _prepare(model, dtype, device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        x = x.half()
    views_fn = views_fn or (lambda t: [t])

    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.float16,
                                                    enabled=(dtype == "amp" and dev.type == "cuda")):
            for v in views_fn(x):
                model(v)

    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    r = bench(fn, warmup, iters, sync)
    return {"config": name, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "dtype": dtype, "batch": batch_size, "img_size": img_size, "fused_bn": fused_bn,
            "k_views": len(views_fn(x)), "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "mean": r["mean"], "n_iters": iters, "warmup": warmup,
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__,
            "preprocessing_included": False}


def tta_latency(model, k_views: int, views_fn=None, **kw) -> dict:
    """Độ trễ TTA K view, đo thật, kèm tỉ lệ so với K * p50 của 1 view."""
    one = latency_report(model, views_fn=None, **kw)
    if views_fn is None:
        views_fn = lambda t: [t] * k_views  # noqa: E731
    tta = latency_report(model, views_fn=views_fn, **kw)
    tta["p50_1view"] = one["p50"]
    tta["ratio_vs_K_times_1view"] = tta["p50"] / (k_views * one["p50"])
    return tta
