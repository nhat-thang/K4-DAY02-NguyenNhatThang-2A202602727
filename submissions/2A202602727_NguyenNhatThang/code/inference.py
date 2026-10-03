"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views_fn)   -> (filenames, y_true, [logits_view_k])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def predict_views(model, loader, device, views_fn, amp: bool = True, dtype: torch.dtype | None = None):
    """Chạy model trên từng view do `views_fn(x) -> list[batch]` sinh ra. Trả về logit cho từng view."""
    model.eval()
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if dtype is not None:
                x = x.to(dtype)
            views = views_fn(x)
            if outs is None:
                outs = [[] for _ in views]
            for k, v in enumerate(views):
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp and dtype is None and device.type == "cuda"):
                    outs[k].append(model(v).float().cpu())
            ys.append(y)
            names.extend(f)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Logit theo đúng thứ tự file. `view`: hàm biến đổi batch (ví dụ view_hflip) hoặc None."""
    view = view or view_identity
    names, y, outs = predict_views(model, loader, device, lambda x: [view(x)], amp)
    return names, y, outs[0]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[3])


def views_flip2(x):
    """TTA K=2: gốc + lật ngang."""
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int, flip: bool = False, out_size: int | None = None):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch x (ví dụ crop 224 từ ảnh gốc 256).
    out_size: nếu có, resize mỗi crop về out_size. flip=True thêm bản lật (K = 10)."""
    h, w = x.shape[-2:]
    tops = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), ((h - crop) // 2, (w - crop) // 2)]
    out = []
    for t, l in tops:
        c = x[..., t:t + crop, l:l + crop]
        if out_size is not None and out_size != crop:
            c = F.interpolate(c, size=(out_size, out_size), mode="bilinear", align_corners=False, antialias=True)
        out.append(c)
        if flip:
            out.append(view_hflip(c))
    return out


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`.

    Chỉ dùng cho CNN có global pooling. ViT/DeiT (pos_embed cố định) và Swin (cửa sổ cố định)
    không nhận kích thước khác 224 nếu không nội suy lại pos_embed -> không áp dụng.
    """
    return [x if x.shape[-1] == s else
            F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view. space="prob": trung bình softmax; "logit": trung bình logit rồi softmax."""
    arr = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        p = _softmax(arr).mean(0)
    elif space == "logit":
        p = _softmax(arr.mean(0))
    else:
        raise ValueError(space)
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    p = np.mean(np.stack([np.asarray(q, dtype=np.float64) for q in list_of_probs]), axis=0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL. Tối ưu log T bằng LBFGS, khởi đầu từ lưới thô."""
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = np.exp(np.linspace(np.log(0.05), np.log(10), 200))
    nll = [F.cross_entropy(z / t, y).item() for t in grid]
    log_t = torch.tensor([np.log(grid[int(np.argmin(nll))])], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode).to(conv.weight.device)
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight if bn.affine else torch.ones_like(std)
    beta = bn.bias if bn.affine else torch.zeros_like(std)
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * (gamma / std).reshape(-1, 1, 1, 1))
        fused.bias.copy_(beta + gamma * (b - bn.running_mean) / std)
    return fused


def fuse_conv_bn(model):
    """Gộp BN vào conv liền trước:  w' = gamma*w/sqrt(var+eps),  b' = beta + gamma*(b-mean)/sqrt(var+eps).

    Trả về BẢN SAO đã gộp (model gốc giữ nguyên) và gán `fused.n_fused` = số cặp đã gộp.
    Cách dò cặp: duyệt các module con theo thứ tự khai báo; một BatchNorm2d đứng ngay sau Conv2d
    trong cùng module cha được coi là liền kề. Đúng cho ResNet/ResNeXt torchvision/timm
    (conv1->bn1, conv2->bn2, downsample.0->downsample.1). Với khối timm BatchNormAct2d (EfficientNet,
    MobileNetV3) BN là lớp con có activation đi kèm: gộp phần BN, giữ activation (xem _fuse_bnact).
    ConvNeXt, ViT, Swin dùng LayerNorm -> không áp dụng. Luôn kiểm tra sai số đầu ra sau khi gộp.
    """
    fused = copy.deepcopy(model).eval()
    n = 0

    def visit(parent: nn.Module):
        nonlocal n
        children = list(parent.named_children())
        for (name_a, a), (name_b, b) in zip(children, children[1:]):
            if isinstance(a, nn.Conv2d) and type(b) is nn.BatchNorm2d and a.out_channels == b.num_features:
                setattr(parent, name_a, _fuse(a, b))
                setattr(parent, name_b, nn.Identity())
                n += 1
            elif isinstance(a, nn.Conv2d) and _is_bnact(b) and a.out_channels == b.num_features:
                setattr(parent, name_a, _fuse(a, b))
                setattr(parent, name_b, _fuse_bnact(b))
                n += 1
        for _, c in parent.named_children():
            visit(c)

    visit(fused)
    fused.n_fused = n
    return fused


def _is_bnact(m) -> bool:
    return type(m).__name__ == "BatchNormAct2d"


def _fuse_bnact(m) -> nn.Module:
    """BatchNormAct2d của timm = BN + drop + act: sau khi gộp BN vào conv chỉ còn drop + act."""
    return nn.Sequential(m.drop, m.act)


@torch.no_grad()
def max_abs_diff(model_a, model_b, x) -> float:
    model_a.eval()
    model_b.eval()
    return float((model_a(x) - model_b(x)).abs().max().item())
