"""Checks for the VAE training/evaluation pipeline on a tiny synthetic dataset_v1 (no RDD2022 images needed)."""
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
import yaml
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.cache_images import build_cache, cache_root  # noqa: E402
from src.evaluation.metrics import PREDICTION_COLUMNS  # noqa: E402
from src.models.train_vae import train  # noqa: E402

SPLITS = ("train", "val", "test_id", "test_ood", "test_perspective")
PER_SPLIT = 12
BASE = dict(experiment_id="smoke", latent=8, image_size=32, batch_size=4, epochs=2, warmup_epochs=1,
            num_workers=0, amp=False, use_cache=False, latency_images=2)


def status_of(index):
    # 0-3 sin dano, 4-7 otro dano, 8-11 bache
    return 0 if index < 4 else 1 if index < 8 else 2


class VAEPipelineTests(unittest.TestCase):
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
                sid, status = f"{split}_{index}", status_of(index)
                path = Path("Country/Country/train/images") / f"{sid}.jpg"
                (self.images_root / path.parent).mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(self.images_root / path)
                images.append(dict(sample_id=sid, country="A" if index % 2 else "B", image_path=path.as_posix(),
                                   has_D00=int(index in (4, 5, 8)), has_D10=0, has_D20=int(index == 6),
                                   has_D40=int(status == 2), has_any_damage=int(status > 0),
                                   other_labels="Repair;D43" if index == 7 else np.nan))
                splits.append(dict(sample_id=sid, split=split))
        self.images = pd.DataFrame(images)
        self.images.to_csv(self.manifests / "images.csv", index=False)
        pd.DataFrame(splits).to_csv(self.manifests / "splits.csv", index=False)
        sha = hashlib.sha256((self.manifests / "splits.csv").read_bytes()).hexdigest()
        lock = {"version": "dataset_v1", "sha256": {"data/manifests/splits.csv": sha}}
        (self.manifests / "dataset_v1.lock.json").write_text(json.dumps(lock), encoding="utf-8")

    def run_train(self, **overrides):
        with contextlib.redirect_stdout(io.StringIO()):
            return train({**BASE, **overrides}, manifests=self.manifests, dataset_root=self.images_root,
                         reports=self.root / "reports", checkpoints=self.root / "ckpt", device="cpu")

    def test_training_writes_contract_outputs(self):
        metrics = self.run_train()
        out = self.root / "reports/smoke"
        for name in ("config.yaml", "history.csv", "metrics.json", "predictions.csv"):
            self.assertTrue((out / name).exists(), name)
        predictions = pd.read_csv(out / "predictions.csv")
        self.assertEqual(tuple(predictions.columns[:7]), PREDICTION_COLUMNS)
        self.assertEqual(list(predictions.columns[7:]),
                         ["y_score_global", "y_score_patch", "y_score_patch_norm", "group", "country"])
        self.assertEqual(len(predictions), 4 * PER_SPLIT)
        self.assertNotIn("train", set(predictions.split))
        self.assertEqual(predictions.threshold.nunique(), 1)
        truth = self.images.set_index("sample_id").loc[predictions.sample_id, "has_D40"].to_numpy()
        self.assertTrue((predictions.y_true.to_numpy() == truth).all())
        primary = metrics["primary_score"]
        self.assertIn(primary, ("global", "patch"))
        self.assertTrue((predictions.y_score == predictions[f"y_score_{primary}"]).all())

        config = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["train_images"], 4)  # solo los sin dano de train
        self.assertEqual(config["val_no_damage_images"], 4)
        history = pd.read_csv(out / "history.csv")
        self.assertIn("beta", history.columns)
        self.assertEqual(history.beta.iloc[-1], 1.0)
        checkpoint = torch.load(self.root / "ckpt/smoke/best.pt", weights_only=True)
        self.assertIn("model", checkpoint)
        self.assertEqual(set(metrics["breakdown"]), {"val", "test_id", "test_ood", "test_perspective"})
        self.assertIn("other_damage_classes", metrics["breakdown"]["val"][primary])

    def redraw_images(self, prefix, seed):
        # cambia los pixeles de un split sin tocar etiquetas
        rng = np.random.default_rng(seed)
        for path in self.images.image_path[self.images.sample_id.str.startswith(prefix)]:
            Image.fromarray(rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(self.images_root / path)

    def test_choices_come_only_from_val(self):
        first = self.run_train()
        # test cambiado por completo (etiquetas e imagenes): nada de lo elegido debe moverse
        flipped = self.images.copy()
        is_test = flipped.sample_id.str.startswith("test_")
        flipped.loc[is_test, "has_D40"] = 1 - flipped.loc[is_test, "has_D40"]
        flipped.to_csv(self.manifests / "images.csv", index=False)
        self.redraw_images("test_", seed=1)
        second = self.run_train(experiment_id="flipped")
        self.assertEqual(first["thresholds"], second["thresholds"])
        self.assertEqual(first["primary_score"], second["primary_score"])
        self.assertEqual(first["position_stats"], second["position_stats"])

    def test_position_stats_come_only_from_train_normals(self):
        first = self.run_train()["position_stats"]
        self.assertEqual(first["n_images"], 4)
        self.assertEqual(first["grid"], [2, 2])  # 32 / 16
        # val si cambia los thresholds (es su trabajo), pero no las stats
        self.redraw_images("val_", seed=2)
        second = self.run_train(experiment_id="val_changed")["position_stats"]
        self.assertEqual(first, second)

    def test_patch_model_run(self):
        metrics = self.run_train(experiment_id="patches", image_size=64, patch_size=32, patch=16)
        out = self.root / "reports/patches"
        predictions = pd.read_csv(out / "predictions.csv")
        self.assertEqual(tuple(predictions.columns[:7]), PREDICTION_COLUMNS)
        self.assertEqual(len(predictions), 4 * PER_SPLIT)
        self.assertTrue(np.isfinite(predictions.y_score_patch_norm).all())
        self.assertEqual(metrics["position_stats"]["grid"], [4, 4])
        self.assertEqual(metrics["latency"]["patch_size"], 32)
        config = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["patch_size"], 32)
        checkpoint = torch.load(self.root / "ckpt/patches/best.pt", weights_only=True)
        self.assertEqual(checkpoint["config"]["patch_size"], 32)

    def test_cached_images_are_used_when_available(self):
        build_cache(self.images, self.images_root, cache_root(32, self.images_root), 32)
        self.run_train(experiment_id="cached", use_cache=True)
        config = yaml.safe_load((self.root / "reports/cached/config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["image_source"], "RDD2022_32")

    def test_missing_cache_fails_early(self):
        with self.assertRaises(FileNotFoundError):
            self.run_train(use_cache=True)

    def test_invalid_configs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_train(warmup_epochs=5)
        with self.assertRaises(ValueError):
            self.run_train(experiment_id="../escape")
        for bad in (dict(patch=24), dict(image_size=64, patch_size=48), dict(image_size=64, patch_size=32, patch=64)):
            with self.assertRaises(ValueError):
                self.run_train(**bad)
        no_val_normals = self.images[~(self.images.sample_id.str.startswith("val_") & (self.images.has_any_damage == 0))]
        no_val_normals.to_csv(self.manifests / "images.csv", index=False)
        with self.assertRaises(ValueError):
            self.run_train()


if __name__ == "__main__":
    unittest.main()
