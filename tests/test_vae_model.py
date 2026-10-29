"""Checks for the convolutional VAE, its loss and the anomaly scores."""
import sys
import unittest
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.models.vae import ConvVAE, anomaly_scores, vae_loss  # noqa: E402


class VAEModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(2, 3, 128, 128)

    def test_output_shape_and_range(self):
        recon, mu, logvar = ConvVAE(latent=16).eval()(self.x)
        self.assertEqual(recon.shape, self.x.shape)
        self.assertEqual(tuple(mu.shape), (2, 16))
        self.assertEqual(mu.shape, logvar.shape)
        self.assertGreaterEqual(recon.min().item(), 0.0)
        self.assertLessEqual(recon.max().item(), 1.0)

    def test_eval_is_deterministic(self):
        model = ConvVAE(latent=16).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(self.x)[0], model(self.x)[0]))

    def test_rejects_incompatible_size(self):
        with self.assertRaises(ValueError):
            ConvVAE(image_size=100)

    def test_loss_terms(self):
        zeros = torch.zeros(2, 8)
        total, reconstruction, kl = vae_loss(self.x, self.x, zeros, zeros, beta=1.0)
        self.assertEqual(kl.item(), 0.0)
        self.assertEqual(reconstruction.item(), 0.0)
        self.assertEqual(total.item(), 0.0)
        _, reconstruction, kl = vae_loss(torch.zeros_like(self.x), self.x, torch.ones(2, 8), zeros, beta=1.0)
        # suma por imagen, promedio en el batch
        self.assertAlmostEqual(reconstruction.item(), self.x.pow(2).flatten(1).sum(1).mean().item(), places=2)
        self.assertAlmostEqual(kl.item(), 4.0, places=5)

    def test_loss_rejects_non_finite(self):
        bad = self.x.clone()
        bad[0, 0, 0, 0] = float("nan")
        zeros = torch.zeros(2, 8)
        with self.assertRaises(FloatingPointError):
            vae_loss(bad, self.x, zeros, zeros, beta=1.0)

    def test_patch_score_highlights_local_error(self):
        x, recon = torch.zeros(1, 3, 128, 128), torch.zeros(1, 3, 128, 128)
        x[..., 32:48, 64:80] = 1.0  # un bache de 16x16
        global_score, patch_score = anomaly_scores(x, recon)
        self.assertGreaterEqual(patch_score.item(), 10 * global_score.item())
        global_score, patch_score = anomaly_scores(torch.rand(8, 3, 128, 128), torch.rand(8, 3, 128, 128))
        self.assertTrue(bool((patch_score >= global_score).all()))

    def test_mixed_precision_loss_is_finite(self):
        model = ConvVAE(latent=16).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            recon, mu, logvar = model(self.x)
        total, _, _ = vae_loss(recon, self.x, mu, logvar, beta=1.0)
        self.assertTrue(torch.isfinite(total))
        total.backward()


if __name__ == "__main__":
    unittest.main()
