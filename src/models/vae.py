"""Convolutional VAE for road images in [0, 1]; the anomaly score is the reconstruction error."""
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


def anomaly_scores(x, recon, patch=16, stride=16):
    """(global, patch) scores per image: mean error, and the max of the patch-averaged error."""
    errors = error_map(x, recon)
    return errors.mean((1, 2, 3)), F.avg_pool2d(errors, patch, stride).amax((1, 2, 3))
