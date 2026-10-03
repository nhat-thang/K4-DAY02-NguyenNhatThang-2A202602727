"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Lựa chọn tiền xử lý lúc đánh giá: ảnh gốc 256x256 -> Resize(round(img_size / 0.875)) -> CenterCrop(img_size).
Với img_size = 224 thì Resize(256) là giữ nguyên ảnh, rồi cắt giữa 224.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1). Không sửa/lọc/chia lại."""
    labels_dir = Path(labels_dir)
    dfs = [pd.read_csv(labels_dir / f"{s}_subset{fold}.csv") for s in ("train", "val", "test")]
    for df in dfs:
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV thiếu cột {missing}")
    return tuple(dfs)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Lỗi thì raise AssertionError."""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in splits.items()}
    total = sum(n.values())
    frac = {k: v / total for k, v in n.items()}

    per_class = pd.DataFrame({k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for k, v in splits.items()})
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(1)

    names = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, v in splits.items():
        assert v["Filename"].is_unique, f"{k}: có Filename trùng trong cùng một tập"
    overlap = {
        "train∩val": len(names["train"] & names["val"]),
        "train∩test": len(names["train"] & names["test"]),
        "val∩test": len(names["val"] & names["test"]),
    }
    union = len(names["train"] | names["val"] | names["test"])

    on_disk = set(os.listdir(images_dir))
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)

    result = {"n": n, "frac": frac, "per_class": per_class, "overlap": overlap, "union": union,
              "missing_files": len(missing), "images_on_disk": len(on_disk)}
    if verbose:
        print("Số ảnh mỗi tập:", n, "| tỉ lệ:", {k: round(v, 4) for k, v in frac.items()})
        print(per_class.to_string())
        print("Giao từng cặp:", overlap, "| hợp ba tập:", union, "| file thiếu trên đĩa:", len(missing))

    assert all(v == 0 for v in overlap.values()), f"Giao giữa các tập khác rỗng: {overlap}"
    assert union == TOTAL_IMAGES, f"Hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:5]}"
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        if abs(frac[k] - target) > 0.01:
            print(f"CẢNH BÁO: tỉ lệ {k} = {frac[k]:.4f} lệch > 1 điểm % khỏi {target}; báo giảng viên.")
    return result


def eval_resize(img_size: int) -> int:
    return int(round(img_size / 0.875))


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform.

    `aug` (trục B):
      - "basic"  : RandomResizedCrop + lật ngang
      - "flipv"  : basic + lật dọc (ảnh chụp từ trên xuống nên hướng không có ý nghĩa)
      - "color"  : basic + ColorJitter
      - "trivial": basic + TrivialAugmentWide
      - "randaug": basic + RandAugment(2, 9)
    Đánh giá: Resize(img_size/0.875) + CenterCrop(img_size), không ngẫu nhiên.
    """
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize(eval_resize(img_size)), T.CenterCrop(img_size), *norm])

    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "flipv":
        ops.append(T.RandomVerticalFlip())
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose([*ops, *norm])


class DeepWeedsDataset(Dataset):
    """Đọc ảnh theo DataFrame (Filename, Label). __getitem__ -> (tensor, label, filename).

    preload=True nạp trước bytes JPEG vào RAM (khoảng 0,5 GB) để tránh nghẽn đọc đĩa.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, preload: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()
        self._bytes = None
        if preload:
            self._bytes = [(self.images_dir / f).read_bytes() for f in self.filenames]

    def __len__(self) -> int:
        return len(self.filenames)

    def load_image(self, i: int) -> Image.Image:
        if self._bytes is not None:
            import io
            return Image.open(io.BytesIO(self._bytes[i])).convert("RGB")
        with Image.open(self.images_dir / self.filenames[i]) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, preload: bool = False, dataset: Dataset | None = None):
    """DataLoader. Eval: không shuffle, giữ thứ tự df. sampler="balanced": WeightedRandomSampler 1/n_c."""
    ds = dataset if dataset is not None else DeepWeedsDataset(df, images_dir, transform, preload=preload)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    if train and sampler == "balanced":
        labels = np.asarray(ds.labels)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        w = 1.0 / counts[labels]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels),
                                    replacement=True, generator=g)
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(ds, batch_size=batch_size, shuffle=(train and smp is None), sampler=smp,
                      drop_last=train, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=g, persistent_workers=num_workers > 0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Đảo chuẩn hoá để vẽ ảnh (C,H,W) hoặc (N,C,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)
