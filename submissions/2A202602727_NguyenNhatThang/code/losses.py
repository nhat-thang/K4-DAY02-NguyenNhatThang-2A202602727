"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_criterion(kind: str = "ce", **kw):
    """kind: "ce" | "ls" (label smoothing) | "focal" | "ce_weighted".

    kw: smoothing (ls, mặc định 0.1), gamma (focal, 2.0), alpha (focal, tensor hoặc None),
        weight (ce_weighted, tensor trọng số lớp tính từ TRAIN).
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        w = kw.get("weight")
        if w is None:
            raise ValueError("ce_weighted cần weight=class_weights(counts_train)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    raise ValueError(f"loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """CE với label smoothing, tự cài đặt: q'(k) = (1 - eps) * 1[k == y] + eps / K.

    loss = -(sum_k q'(k) * log p_k). eps = 0 cho đúng CE (kiểm tra trong tests_code.py).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        uniform = -logp.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), trung bình batch. gamma = 0, alpha = None -> CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp của TRAIN.

    beta = 0: 1/n_c, chuẩn hoá trung bình = 1.
    beta > 0: class-balanced (Cui et al.): (1 - beta) / (1 - beta^n_c), chuẩn hoá tổng = K.
    """
    n = np.asarray(counts, dtype=np.float64)
    if beta and beta > 0:
        w = (1 - beta) / (1 - np.power(beta, n))
    else:
        w = 1.0 / n
    w = w * len(n) / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = rng.integers(h), rng.integers(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", rng: np.random.Generator | None = None):
    """Trộn một batch: trả về (x_mix, (y_a, y_b, lam)).

    CutMix: lam được tính lại theo diện tích thực của hộp sau khi cắt bởi biên ảnh.
    rng mặc định dùng np.random (đã được seed trong train.set_seed).
    """
    rng = rng or np.random.default_rng(np.random.randint(2 ** 31))
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = rand_bbox(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    else:
        raise ValueError(f"mix không hợp lệ: {mode}")
    return x_mix, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
