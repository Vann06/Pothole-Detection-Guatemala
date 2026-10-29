"""Checks for the helpers shared by every model (transforms, loader, metrics contract, ids)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402
from src.evaluation.metrics import PREDICTION_COLUMNS, best_f1_threshold, prediction_frame  # noqa: E402
from src.utils.seed import safe_experiment_id  # noqa: E402


class SharedUtilsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.image = Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))

    def test_default_transform_matches_original_resnet_pipeline(self):
        # la version de ricardo antes del refactor, escrita a mano
        original = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.assertTrue(torch.equal(build_transforms(size=32)(self.image), original(self.image)))

    def test_unnormalized_transform_keeps_unit_range(self):
        tensor = build_transforms(size=128, normalize=False)(self.image)
        self.assertEqual(tuple(tensor.shape), (3, 128, 128))
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_prediction_frame_keeps_contract_order(self):
        frame = prediction_frame(["a", "b"], "val", np.array([0, 1]), np.array([0.2, 0.9]), 0.5, "exp", "dataset_v1")
        self.assertEqual(tuple(frame.columns), PREDICTION_COLUMNS)
        self.assertEqual(frame.split.tolist(), ["val", "val"])

    def test_best_f1_threshold_separates_perfect_scores(self):
        self.assertEqual(best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 0.8)

    def test_experiment_id_rejects_paths(self):
        self.assertEqual(safe_experiment_id("vae_base-1"), "vae_base-1")
        for bad in ("../x", "a/b", "a\\b", "", "C:x", None):
            with self.assertRaises(ValueError):
                safe_experiment_id(bad)


if __name__ == "__main__":
    unittest.main()
