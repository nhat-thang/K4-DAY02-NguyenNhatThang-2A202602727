"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy được trên CPU, không cần dữ liệu:
    python -m unittest test_code -v        (từ thư mục code/)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inference as I  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
import train  # noqa: E402
from benchmark import bench  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(64, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (64,), generator=g)

    def test_focal_gamma0_equals_ce(self):
        fl = L.FocalLoss(gamma=0.0)(self.logits, self.y)
        ce = F.cross_entropy(self.logits, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_downweights_easy(self):
        self.assertLess(L.FocalLoss(2.0)(self.logits, self.y).item(), F.cross_entropy(self.logits, self.y).item())

    def test_label_smoothing_eps0_equals_ce(self):
        ls = L.LabelSmoothingCE(0.0)(self.logits, self.y)
        self.assertLess(abs(ls.item() - F.cross_entropy(self.logits, self.y).item()), 1e-6)

    def test_label_smoothing_matches_torch(self):
        ls = L.LabelSmoothingCE(0.1)(self.logits, self.y)
        ref = F.cross_entropy(self.logits, self.y, label_smoothing=0.1)
        self.assertLess(abs(ls.item() - ref.item()), 1e-5)

    def test_class_weights(self):
        w = L.class_weights([100, 10, 1000])
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertGreater(w[1], w[0])
        self.assertGreater(w[0], w[2])
        wcb = L.class_weights([100, 10, 1000], beta=0.999)
        self.assertAlmostEqual(wcb.sum().item(), 3.0, places=4)

    def test_cutmix_lambda_matches_area(self):
        x = torch.zeros(8, 3, 32, 32)
        x[4:] = 1.0
        y = torch.arange(8)  # nhãn = chỉ số, nên yb chính là hoán vị perm
        for s in range(20):
            xm, (ya, yb, lam) = L.mix_batch(x, y, 1.0, "cutmix", rng=np.random.default_rng(s))
            # tỉ lệ pixel bị thay phải bằng 1 - lam với mọi ảnh ghép từ ảnh khác giá trị
            for i in range(8):
                if x[i, 0, 0, 0] != x[yb[i], 0, 0, 0]:
                    changed = (xm[i] != x[i]).float().mean().item()
                    self.assertAlmostEqual(changed, 1 - lam, places=5)
            self.assertTrue(0.0 <= lam <= 1.0)

    def test_mixup_and_mixed_loss(self):
        x = torch.randn(8, 3, 8, 8)
        y = torch.arange(8) % 9
        xm, t = L.mix_batch(x, y, 0.4, "mixup", rng=np.random.default_rng(0))
        self.assertEqual(xm.shape, x.shape)
        logits = torch.randn(8, 9)
        ce = nn.CrossEntropyLoss()
        ref = t[2] * ce(logits, t[0]) + (1 - t[2]) * ce(logits, t[1])
        self.assertAlmostEqual(L.mixed_loss(ce, logits, t).item(), ref.item(), places=6)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 9)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.fc(self.pool(x).flatten(1))

    def get_classifier(self):
        return self.fc


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        torch.manual_seed(0)
        m = TinyCNN()
        for bn in (m.bn1, m.bn2):  # thống kê BN khác mặc định để phép gộp có ý nghĩa
            bn.running_mean.uniform_(-1, 1)
            bn.running_var.uniform_(0.5, 2)
            bn.weight.data.uniform_(0.5, 1.5)
            bn.bias.data.uniform_(-0.5, 0.5)
        fused = I.fuse_conv_bn(m)
        self.assertEqual(fused.n_fused, 2)
        x = torch.randn(4, 3, 16, 16)
        self.assertLess(I.max_abs_diff(m, fused, x), 1e-5)

    def test_temperature_recovers_scale(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 9, 5000)
        true = rng.normal(size=(5000, 9))
        true[np.arange(5000), y] += 2.0
        # sinh nhãn theo softmax(true), rồi đưa logit đã nhân 3 (quá tự tin) -> T ≈ 3
        p = I._softmax(true)
        y = np.array([rng.choice(9, p=pi) for pi in p])
        T = I.fit_temperature(true * 3, y)
        self.assertAlmostEqual(T, 3.0, delta=0.3)
        probs = I.apply_temperature(true * 3, T)
        np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-9)
        np.testing.assert_array_equal(probs.argmax(1), (true * 3).argmax(1))

    def test_aggregate_views(self):
        a, b = np.random.randn(5, 9), np.random.randn(5, 9)
        for space in ("prob", "logit"):
            p = I.aggregate_views([a, b], space)
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(I.aggregate_views([a], "prob"), I._softmax(a), atol=1e-12)

    def test_views(self):
        x = torch.randn(2, 3, 224, 224)
        self.assertTrue(torch.equal(I.view_hflip(I.view_hflip(x)), x))
        full = torch.randn(2, 3, 256, 256)
        crops = I.views_multicrop(full, 224)
        self.assertEqual(len(crops), 5)
        self.assertEqual(len(I.views_multicrop(full, 224, flip=True)), 10)
        self.assertEqual(crops[0].shape, (2, 3, 224, 224))
        self.assertTrue(torch.equal(crops[4], full[..., 16:240, 16:240]))  # crop giữa = CenterCrop(224)
        self.assertEqual(I.views_multicrop(full, 192, out_size=224)[0].shape, (2, 3, 224, 224))
        self.assertEqual([v.shape[-1] for v in I.views_multiscale(x, [192, 224, 256])], [192, 224, 256])


class TestModelAndTrain(unittest.TestCase):
    def test_param_groups_no_decay_on_norm_bias(self):
        m = TinyCNN()
        groups = M.param_groups(m, 1e-4, 1e-3, 0.05)
        n = sum(p.numel() for g in groups for p in g["params"])
        self.assertEqual(n, sum(p.numel() for p in m.parameters()))
        for g in groups:
            if g["weight_decay"] > 0:
                self.assertTrue(all(p.ndim > 1 for p in g["params"]))
            if g["name"].startswith("head"):
                self.assertEqual(g["lr"], 1e-3)
            else:
                self.assertEqual(g["lr"], 1e-4)

    def test_freeze_keeps_bn_eval(self):
        m = TinyCNN()
        M.freeze_backbone(m)
        M.set_train_mode(m)
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.fc.training)
        self.assertEqual({id(p) for p in m.parameters() if p.requires_grad}, {id(p) for p in m.fc.parameters()})
        before = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 8, 8))
        self.assertTrue(torch.equal(before, m.bn1.running_mean))

    def test_scheduler_warmup_cosine(self):
        m = TinyCNN()
        cfg = train.Config(epochs=4, warmup_epochs=1.0)
        opt = train.build_optimizer(m, cfg)
        sch = train.build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs = []
        for _ in range(40):
            opt.step()
            sch.step()
            lrs.append(opt.param_groups[0]["lr"])
        self.assertAlmostEqual(max(lrs), cfg.lr_backbone, places=10)
        self.assertEqual(int(np.argmax(lrs)), 8)  # đỉnh ở cuối warmup
        self.assertLess(lrs[-1], 1e-7)

    def test_ema(self):
        m = TinyCNN()
        ema = train.EMA(m, 0.5)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        old = [p.clone() for p in ema.module.parameters()]
        ema.update(m)
        for o, e, p in zip(old, ema.module.parameters(), m.parameters()):
            torch.testing.assert_close(e, 0.5 * o + 0.5 * p)

    def test_parse_overrides(self):
        d = train.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false", "lr_head=3e-3",
                                   "sampler=balanced", "class_weight_beta=0.999"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 3e-3,
                             "sampler": "balanced", "class_weight_beta": 0.999})
        with self.assertRaises(KeyError):
            train.parse_overrides(["nope=1"])

    def test_bench_percentiles(self):
        r = bench(lambda: sum(range(1000)), warmup=3, iters=60)
        self.assertEqual(r["n"], 60)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])


if __name__ == "__main__":
    unittest.main()
