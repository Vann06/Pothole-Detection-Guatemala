"""Checks for the convolutional VAE, its loss and the anomaly scores."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.models.vae import (ConvVAE, PatchVAE, anomaly_scores, build_vae, from_tiles, normalized_max,  # noqa: E402
                            position_stats, to_tiles, vae_loss, vae_terms)


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
        global_score, patch_score, grid = anomaly_scores(x, recon)
        self.assertGreaterEqual(patch_score.item(), 10 * global_score.item())
        self.assertEqual(tuple(grid.shape), (1, 64))
        self.assertEqual(int(grid.argmax()), 2 * 8 + 4)  # fila 2, columna 4 del grid 8x8
        global_score, patch_score, grid = anomaly_scores(torch.rand(8, 3, 128, 128), torch.rand(8, 3, 128, 128))
        self.assertTrue(bool((patch_score >= global_score).all()))
        self.assertTrue(torch.equal(patch_score, grid.amax(1)))

    def test_tiles_round_trip_in_row_major_order(self):
        x = torch.rand(2, 3, 128, 128)
        tiles = to_tiles(x, 64)
        self.assertEqual(tuple(tiles.shape), (8, 3, 64, 64))
        self.assertTrue(torch.equal(tiles[1], x[0, :, :64, 64:]))  # imagen 0, fila 0, columna 1
        self.assertTrue(torch.equal(tiles[6], x[1, :, 64:, :64]))  # imagen 1, fila 1, columna 0
        self.assertTrue(torch.equal(from_tiles(tiles, 2, 128, 128), x))

    def test_patch_vae_shapes_and_tile_independence(self):
        model = PatchVAE(latent=8, image_size=128, patch_size=64).eval()
        with torch.no_grad():
            recon, mu, logvar = model(self.x)
            self.assertEqual(recon.shape, self.x.shape)
            self.assertEqual(tuple(mu.shape), (2, 4 * 8))
            self.assertGreaterEqual(recon.min().item(), 0.0)
            self.assertLessEqual(recon.max().item(), 1.0)
            # en eval cada parche se reconstruye solo: cambiar uno no toca a los demas
            changed = self.x.clone()
            changed[0, :, :64, :64] = 0.0
            other = model(changed)[0]
        self.assertTrue(torch.allclose(recon[0, :, 64:, :], other[0, :, 64:, :]))
        self.assertTrue(torch.allclose(recon[1], other[1]))

    def test_patch_vae_kl_is_sum_over_tiles(self):
        model = PatchVAE(latent=8, image_size=128, patch_size=64).eval()
        with torch.no_grad():
            recon, mu, logvar = model(self.x)
            tile_mu, tile_logvar = ConvVAE.encode(model, to_tiles(self.x, 64))
        _, kl = vae_terms(recon, self.x, mu, logvar)
        _, tile_kl = vae_terms(to_tiles(recon, 64), to_tiles(self.x, 64), tile_mu, tile_logvar)
        self.assertTrue(torch.allclose(kl, tile_kl.view(2, 4).sum(1)))

    def test_build_vae_keeps_whole_image_checkpoints_compatible(self):
        self.assertIs(type(build_vae(16, 128)), ConvVAE)
        reference = ConvVAE(latent=16, image_size=128).state_dict()
        built = build_vae(16, 128).state_dict()
        self.assertEqual({k: v.shape for k, v in reference.items()}, {k: v.shape for k, v in built.items()})
        self.assertIsInstance(build_vae(16, 512, 64), PatchVAE)
        with self.assertRaises(ValueError):
            build_vae(16, 500, 64)
        with self.assertRaises(ValueError):
            build_vae(16, 480, 48)  # divide la imagen pero no pasa por las 5 etapas

    def test_position_stats_and_normalized_max(self):
        grids = np.array([[1.0, 10.0, 5.0], [3.0, 10.0, 7.0]])
        stats = position_stats(grids, floor_frac=0.5)
        np.testing.assert_allclose(stats["mean"], [2.0, 10.0, 6.0])
        np.testing.assert_allclose(stats["std"], [1.0, 0.0, 1.0])
        # la mediana de std es 1, asi que el piso 0.5 evita dividir entre cero en la posicion constante
        np.testing.assert_allclose(stats["std_eff"], [1.0, 0.5, 1.0])
        self.assertEqual((stats["n_floored"], stats["n_images"]), (1, 2))
        scores = normalized_max(np.array([[2.0, 11.0, 6.0]]), stats["mean"], stats["std_eff"])
        np.testing.assert_allclose(scores, [2.0])  # (11 - 10) / 0.5

    def test_mixed_precision_loss_is_finite(self):
        model = ConvVAE(latent=16).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            recon, mu, logvar = model(self.x)
        total, _, _ = vae_loss(recon, self.x, mu, logvar, beta=1.0)
        self.assertTrue(torch.isfinite(total))
        total.backward()


if __name__ == "__main__":
    unittest.main()
