"""Checks for the ResNet18 pipeline on a tiny synthetic dataset_v1 (no RDD2022 images needed)."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.dataset import build_transforms, compute_pos_weight, load_split_frame, PotholeDataset  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.models.train_resnet import AuxiliaryLoss, train  # noqa: E402

SPLITS = ("train", "val", "test_id", "test_ood", "test_perspective")
PER_SPLIT = 8


class ResNetPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifests, self.images_root = self.root / "manifests", self.root / "RDD2022"
        self.manifests.mkdir()
        rng = np.random.default_rng(0)
        images, splits = [], []
        for split in SPLITS:
            for index in range(PER_SPLIT):
                sid = f"{split}_{index}"
                path = Path("Country/train/images") / f"{sid}.jpg"
                (self.images_root / path.parent).mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(self.images_root / path)
                label, crack = int(index % 4 == 0), int(index % 2 == 1)
                images.append(dict(sample_id=sid, country="Country", image_path=path.as_posix(),
                                   has_D00=crack, has_D10=int(index % 4 == 1), has_D20=int(index in (2, 3)),
                                   has_D40=label, has_any_damage=int(label or crack or index in (2, 3))))
                splits.append(dict(sample_id=sid, split=split))
        pd.DataFrame(images).to_csv(self.manifests / "images.csv", index=False)
        self.write_splits(pd.DataFrame(splits))

    def write_splits(self, splits):
        splits.to_csv(self.manifests / "splits.csv", index=False)
        sha = hashlib.sha256((self.manifests / "splits.csv").read_bytes()).hexdigest()
        lock = {"version": "dataset_v1", "sha256": {"data/manifests/splits.csv": sha}}
        (self.manifests / "dataset_v1.lock.json").write_text(json.dumps(lock), encoding="utf-8")

    def test_load_split_frame_merges_and_filters(self):
        frame = load_split_frame(manifests=self.manifests)
        self.assertEqual(len(frame), PER_SPLIT * len(SPLITS))
        self.assertTrue(frame.sample_id.is_unique)
        subset = load_split_frame(["train", "val"], manifests=self.manifests)
        self.assertEqual(set(subset.split), {"train", "val"})
        self.assertTrue(subset.sample_id.str.split("_").str[0].isin(["train", "val"]).all())

    def test_modified_splits_are_rejected(self):
        (self.manifests / "splits.csv").write_text("sample_id,split\ntrain_0,val\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_split_frame(manifests=self.manifests)

    def test_pos_weight_uses_train_only(self):
        frame = load_split_frame(manifests=self.manifests)
        weight = compute_pos_weight(frame)
        self.assertAlmostEqual(weight, 6 / 2)
        frame.loc[frame.split == "val", "has_D40"] = 1
        self.assertAlmostEqual(compute_pos_weight(frame), weight)

    def test_eval_transform_is_deterministic(self):
        frame = load_split_frame(["val"], manifests=self.manifests)
        dataset = PotholeDataset(frame, build_transforms(train=False, augment=True, size=32), self.images_root)
        first, second = dataset[0][0], dataset[0][0]
        self.assertEqual(tuple(first.shape), (3, 32, 32))
        self.assertTrue(torch.equal(first, second))

    def test_model_outputs_one_logit(self):
        model = build_resnet18(pretrained=False).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 64, 64)).shape), (2, 1))

    def test_training_smoke_writes_contract_outputs(self):
        config = dict(experiment_id="smoke", epochs=1, batch_size=4, image_size=64, num_workers=0,
                      amp=False, augment=True, latency_images=4)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = train(config, manifests=self.manifests, dataset_root=self.images_root,
                            reports=self.root / "reports", checkpoints=self.root / "ckpt", device="cpu")
        out = self.root / "reports/smoke"
        for name in ("config.yaml", "history.csv", "metrics.json", "predictions.csv"):
            self.assertTrue((out / name).exists(), name)
        self.assertTrue((self.root / "ckpt/smoke/best.pt").exists())
        predictions = pd.read_csv(out / "predictions.csv")
        self.assertEqual(list(predictions.columns), ["sample_id", "split", "y_true", "y_score", "threshold",
                                                     "experiment_id", "dataset_version"])
        self.assertEqual(len(predictions), 4 * PER_SPLIT)
        self.assertNotIn("train", set(predictions.split))
        self.assertEqual(predictions.threshold.nunique(), 1)
        self.assertEqual(set(metrics["splits"]), {"val", "test_id", "test_ood", "test_perspective"})
        self.assertNotIn("auxiliary", metrics["splits"]["val"])

    def test_auxiliary_labels_put_d40_first(self):
        frame = load_split_frame(["train"], manifests=self.manifests)
        dataset = PotholeDataset(frame, build_transforms(size=32), self.images_root, ["D00", "D10", "D20"])
        labels = dataset[0][1]
        self.assertEqual(tuple(labels.shape), (4,))
        self.assertEqual(labels[0].item(), frame.has_D40.iloc[0])
        self.assertEqual(tuple(build_resnet18(outputs=4)(torch.zeros(1, 3, 64, 64)).shape), (1, 4))

    def test_auxiliary_loss_weights(self):
        logits, labels = torch.randn(6, 4), torch.randint(0, 2, (6, 4)).float()
        main = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))(logits[:, 0], labels[:, 0])
        self.assertTrue(torch.isclose(AuxiliaryLoss([3.0, 1, 1, 1], 0.0)(logits, labels), main))
        self.assertTrue(torch.isclose(AuxiliaryLoss([3.0], 0.4)(logits[:, :1], labels[:, :1]), main))
        self.assertGreater(AuxiliaryLoss([3.0, 1, 1, 1], 0.4)(logits, labels), main)

    def test_auxiliary_training_keeps_d40_contract(self):
        config = dict(experiment_id="aux", epochs=1, batch_size=4, image_size=64, num_workers=0, amp=False,
                      latency_images=2, aux_classes=["D00", "D10", "D20"], aux_weight=0.4)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = train(config, manifests=self.manifests, dataset_root=self.images_root,
                            reports=self.root / "reports", checkpoints=self.root / "ckpt", device="cpu")
        predictions = pd.read_csv(self.root / "reports/aux/predictions.csv")
        self.assertEqual(list(predictions.columns[:7]), ["sample_id", "split", "y_true", "y_score", "threshold",
                                                         "experiment_id", "dataset_version"])
        self.assertIn("y_score_D20", predictions.columns)
        images = pd.read_csv(self.manifests / "images.csv").set_index("sample_id")
        self.assertTrue((predictions.y_true.to_numpy() == images.loc[predictions.sample_id, "has_D40"].to_numpy()).all())
        self.assertEqual(set(metrics["splits"]["test_ood"]["auxiliary"]), {"D00", "D10", "D20"})


if __name__ == "__main__":
    unittest.main()
