"""Convolutional VAE for road images in [0, 1]; the anomaly score is the reconstruction error.

ConvVAE models a whole (small) image. PatchVAE runs the same network on non-overlapping tiles of a larger
image, so small damage covers a bigger share of what the model sees.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LOGVAR_RANGE = (-30.0, 20.0)


def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


def up(cin, cout):
    # convtranspose y no upsample bilineal: su backward en cuda es determinista
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


class ConvVAE(nn.Module):
    """Each stage halves the resolution: 128 -> 4 with the default 5 stages. Uses z = mu in eval mode."""

    def __init__(self, latent=128, image_size=128, channels=(32, 64, 128, 256, 256)):
        super().__init__()
        if image_size % 2 ** len(channels):
            raise ValueError(f"image_size {image_size} must be divisible by {2 ** len(channels)}")
        self.channels, self.grid = channels, image_size // 2 ** len(channels)
        flat = channels[-1] * self.grid ** 2
        stages = [down(cin, cout) for cin, cout in zip((3, *channels), channels)]
        self.encoder = nn.Sequential(*stages, nn.Flatten())
        self.mu, self.logvar = nn.Linear(flat, latent), nn.Linear(flat, latent)
        self.project = nn.Linear(latent, flat)
        back = channels[::-1]
        self.decoder = nn.Sequential(*(up(cin, cout) for cin, cout in zip(back, back[1:])),
                                     nn.ConvTranspose2d(back[-1], 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden).clamp(*LOGVAR_RANGE)

    def decode(self, z):
        hidden = F.leaky_relu(self.project(z), 0.2)
        return self.decoder(hidden.view(-1, self.channels[-1], self.grid, self.grid))

    def forward(self, x):
        mu, logvar = self.encode(x)
        # en eval no se muestrea, asi el score es reproducible
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decode(z), mu, logvar


def to_tiles(x, size):
    """(B, C, H, W) -> (B * rows * cols, C, size, size), tiles grouped image by image in row-major order."""
    batch, channels, height, width = x.shape
    rows, cols = height // size, width // size
    tiles = x.reshape(batch, channels, rows, size, cols, size).permute(0, 2, 4, 1, 3, 5)
    return tiles.reshape(batch * rows * cols, channels, size, size)


def from_tiles(tiles, batch, height, width):
    """Inverse of to_tiles."""
    channels, size = tiles.shape[1], tiles.shape[-1]
    rows, cols = height // size, width // size
    image = tiles.reshape(batch, rows, cols, channels, size, size).permute(0, 3, 1, 4, 2, 5)
    return image.reshape(batch, channels, height, width)


class PatchVAE(ConvVAE):
    """ConvVAE applied to non-overlapping patch_size tiles; reconstructs the full image.

    mu/logvar come back as (B, tiles * latent), so vae_terms sums the KL of every tile of an image.
    """

    def __init__(self, latent=128, image_size=512, patch_size=64, channels=(32, 64, 128, 256, 256)):
        if image_size % patch_size:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        super().__init__(latent, patch_size, channels)
        self.patch_size = patch_size

    def forward(self, x):
        batch, _, height, width = x.shape
        recon, mu, logvar = super().forward(to_tiles(x, self.patch_size))
        return from_tiles(recon, batch, height, width), mu.reshape(batch, -1), logvar.reshape(batch, -1)


def build_vae(latent, image_size, patch_size=None):
    """Whole-image ConvVAE, or PatchVAE when patch_size is given."""
    if patch_size is None:
        return ConvVAE(latent=latent, image_size=image_size)
    return PatchVAE(latent=latent, image_size=image_size, patch_size=patch_size)


def vae_terms(recon, x, mu, logvar):
    """Per-image squared error (summed over pixels) and KL (summed over latent dims), in fp32."""
    recon, x, mu, logvar = recon.float(), x.float(), mu.float(), logvar.float()
    reconstruction = (recon - x).pow(2).flatten(1).sum(1)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1)
    return reconstruction, kl


def vae_loss(recon, x, mu, logvar, beta):
    """(total, reconstruction, KL) averaged over the batch; sums per image keep KL from being overweighted."""
    reconstruction, kl = (term.mean() for term in vae_terms(recon, x, mu, logvar))
    total = reconstruction + beta * kl
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite VAE loss")
    return total, reconstruction, kl


def error_map(x, recon):
    """Per-pixel squared error averaged over channels, shape (B, 1, H, W)."""
    return (x.float() - recon.float()).pow(2).mean(1, keepdim=True)


def patch_grid(errors, patch, stride):
    """Patch-averaged error per image, flattened in row-major order: (B, cells)."""
    return F.avg_pool2d(errors, patch, stride).flatten(1)


def anomaly_scores(x, recon, patch=16, stride=16):
    """(global, patch, grid): mean image error, max patch error and the patch grid it comes from."""
    errors = error_map(x, recon)
    grid = patch_grid(errors, patch, stride)
    return errors.mean((1, 2, 3)), grid.amax(1), grid


def position_stats(grids, floor_frac):
    """Per-position mean/std of the patch grid over normal images.

    The std gets a floor relative to its median: near-constant positions (sky, hood) would otherwise
    produce huge z-scores and dominate the max.
    """
    grids = np.asarray(grids, dtype=np.float64)
    mean, std = grids.mean(0), grids.std(0)
    floor = floor_frac * float(np.median(std))
    return dict(mean=mean, std=std, std_eff=np.maximum(std, floor), floor=floor,
                n_floored=int((std < floor).sum()), n_images=int(len(grids)))


def normalized_max(grids, mean, std_eff):
    """Max over positions of (patch error - position mean) / position std."""
    return ((np.asarray(grids, dtype=np.float64) - mean) / std_eff).max(1)
