"""VAE trained on a single forgery, with optional SSIM-controlled sampling."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import preprocess_img, ssim, tensor_to_uint8


@dataclass
class VAEConfig:
    n_epochs: int = 799
    lr: float = 1e-3
    h_dim1: int = 512
    h_dim2: int = 256
    z_dim: int = 2


@dataclass
class SSIMConfig:
    """Settings for VAE-SSIM: generated forgeries must fall inside an SSIM band."""
    n_epochs: int = 200
    lr: float = 5e-4
    ssim_min: float = 0.03
    ssim_max: float = 0.2
    check_every: int = 4
    max_tries: int = 100


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super().__init__()
        self.z_dim = z_dim
        # encoder
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(x.size(0), -1))
        return self.decoder(self.sampling(mu, log_var)), mu, log_var


def loss_function(recon_x, x, mu, log_var):
    """Reconstruction (BCE) + KL divergence."""
    bce = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld


@torch.no_grad()
def sample_vae(vae, img_shape, n_samples, device):
    """Decode `n_samples` random latents into images in [0, 1]."""
    z = torch.randn(n_samples, vae.z_dim, device=device)
    return vae.decoder(z).view(n_samples, *img_shape)


def in_ssim_band(sample, reference, cfg):
    score = ssim(reference, preprocess_img(tensor_to_uint8(sample)))
    return cfg.ssim_min <= score <= cfg.ssim_max


def train_vae(img, cfg, device, reference=None, ssim_cfg=None):
    """Train a VAE on one signature of shape (1, H, W).

    With `ssim_cfg`, training stops early once a sample's SSIM to
    `reference` (the preprocessed forgery) falls inside the SSIM band.
    """
    img_shape = tuple(img.shape)
    x = img.unsqueeze(0).to(device)
    vae = VAE(x.numel(), cfg.h_dim1, cfg.h_dim2, cfg.z_dim).to(device)
    n_epochs = ssim_cfg.n_epochs if ssim_cfg else cfg.n_epochs
    lr = ssim_cfg.lr if ssim_cfg else cfg.lr
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(n_epochs):
        vae.train()
        optimizer.zero_grad()
        recon, mu, log_var = vae(x)
        loss = loss_function(recon, x, mu, log_var)
        loss.backward()
        optimizer.step()

        if ssim_cfg and (epoch + 1) % ssim_cfg.check_every == 0:
            sample = sample_vae(vae, img_shape, 1, device)[0]
            if in_ssim_band(sample, reference, ssim_cfg):
                print(f"  SSIM band reached at epoch {epoch + 1}")
                break

    return vae


def sample_vae_ssim(vae, img_shape, n_samples, reference, cfg, device):
    """Rejection-sample images whose SSIM to `reference` is inside the band.

    Falls back to the last draw when no sample hits the band within
    `cfg.max_tries` attempts.
    """
    samples, misses = [], 0
    for _ in range(n_samples):
        for _ in range(cfg.max_tries):
            sample = sample_vae(vae, img_shape, 1, device)[0]
            if in_ssim_band(sample, reference, cfg):
                break
        else:
            misses += 1
        samples.append(sample)
    if misses:
        print(f"  warning: {misses}/{n_samples} samples outside the SSIM band")
    return torch.stack(samples)
