"""Conditional GAN trained on a single forgery to produce synthetic forgeries."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CGANConfig:
    n_epochs: int = 800
    latent_dim: int = 20
    n_classes: int = 2
    lr: float = 1e-3
    b1: float = 0.5
    b2: float = 0.999


class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        return self.model(gen_input).view(noise.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super().__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        return self.model(d_in)


def train_cgan(img, cfg, device):
    """Train a CGAN (LSGAN loss) on one signature of shape (1, H, W)."""
    img_shape = tuple(img.shape)
    generator = Generator(cfg.latent_dim, cfg.n_classes, img_shape).to(device)
    discriminator = Discriminator(cfg.n_classes, img_shape).to(device)
    adversarial_loss = nn.MSELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

    # BatchNorm needs more than one sample, so the forgery is repeated once
    # per class label to form the real batch.
    n = cfg.n_classes
    real_imgs = img.unsqueeze(0).repeat(n, 1, 1, 1).to(device)
    labels = torch.arange(n, device=device)
    valid = torch.ones(n, 1, device=device)
    fake = torch.zeros(n, 1, device=device)

    for epoch in range(cfg.n_epochs):
        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(n, cfg.latent_dim, device=device)
        gen_labels = torch.randint(0, n, (n,), device=device)
        gen_imgs = generator(z, gen_labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train discriminator
        optimizer_D.zero_grad()
        d_real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        if (epoch + 1) % 100 == 0:
            print(f"  [Epoch {epoch + 1}/{cfg.n_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    return generator


@torch.no_grad()
def sample_cgan(generator, cfg, n_samples, device):
    """Draw `n_samples` images in [-1, 1], cycling through the class labels.

    The generator stays in training mode, as in the original experiments, so
    BatchNorm uses the statistics of the sampled batch.
    """
    z = torch.randn(n_samples, cfg.latent_dim, device=device)
    labels = torch.arange(n_samples, device=device) % cfg.n_classes
    return generator(z, labels)
