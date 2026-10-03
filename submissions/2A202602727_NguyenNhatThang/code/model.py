"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (train(), giữ phần đóng băng ở eval)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float | None = None):
    """Tạo model 9 lớp. init: "scratch" | "frozen" | "finetune".

    Tag trọng số thực sự được tải nằm ở `model.weight_tag` (lấy từ model.pretrained_cfg).
    """
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    kw = {"pretrained": pretrained and init != "scratch", "num_classes": num_classes, "drop_rate": drop_rate}
    if drop_path_rate is not None:
        kw["drop_path_rate"] = drop_path_rate
    model = timm.create_model(name, **kw)
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    tag = cfg.get("tag") or ""
    arch = cfg.get("architecture") or name
    model.weight_tag = f"{arch}.{tag}" if (kw["pretrained"] and tag) else (arch if kw["pretrained"] else "none (scratch)")
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số trừ head. BN của backbone được giữ ở eval qua set_train_mode."""
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(); nếu backbone đóng băng thì đưa mọi module trừ head về eval
    (BN không cập nhật running stats, dropout/drop-path trong backbone tắt)."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head_modules = set(model.get_classifier().modules())
        for m in model.modules():
            if m is not model and m not in head_modules:
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm (slide trang 52): backbone ndim>1 (có wd) · norm/bias backbone (wd=0) · head (lr_head).

    Trong head cũng tách bias ra không áp dụng weight decay.
    """
    head = _head_param_ids(model)
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name \
            or "cls_token" in name or "relative_position_bias_table" in name
        part = "head" if id(p) in head else "backbone"
        groups[f"{part}_{'no_decay' if no_decay else 'decay'}"].append(p)
    out = []
    for key, params in groups.items():
        if not params:
            continue
        out.append({"params": params, "name": key,
                    "lr": lr_head if key.startswith("head") else lr_backbone,
                    "weight_decay": 0.0 if key.endswith("no_decay") else weight_decay})
    return out


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm FLOPs của conv/matmul/attention,
    1 MAC = 2 FLOPs) rồi chia 2. Không đếm phép element-wise (BN, activation).
    """
    from torch.utils.flop_counter import FlopCounterMode
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 2 / 1e9
