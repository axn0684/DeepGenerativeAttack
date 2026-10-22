"""Generate synthetic forgeries with CGAN, VAE or VAE-SSIM.

A separate generative model is trained on each forgery image in --input,
and --n-samples images are saved as generated_<name>_<k>.png in --output.

Example:
    python generate.py --method cgan --input data/CEDAR/full_forg --output generated/cgan_CEDAR_forg
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from sigattack.cgan import CGANConfig, sample_cgan, train_cgan
from sigattack.data import get_device, load_signature, preprocess_img
from sigattack.vae import SSIMConfig, VAEConfig, sample_vae, sample_vae_ssim, train_vae


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["cgan", "vae", "vae-ssim"], required=True)
    p.add_argument("--input", type=Path, required=True, help="folder of forgery images (forgeries_*.png)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=9, help="synthetic images per forgery")
    p.add_argument("--pattern", default="forgeries_*.png", help="glob for input images")
    p.add_argument("--max-size", type=int, default=512, help="larger images are resized to this square size")
    p.add_argument("--epochs", type=int, help="override the default number of training epochs")
    p.add_argument("--ssim-min", type=float, default=SSIMConfig.ssim_min)
    p.add_argument("--ssim-max", type=float, default=SSIMConfig.ssim_max)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def generate(path, args, device):
    # The CGAN's tanh output matches [-1, 1] inputs; the VAE's BCE loss needs [0, 1].
    img = load_signature(path, args.max_size, signed=args.method == "cgan")

    if args.method == "cgan":
        cfg = CGANConfig()
        if args.epochs:
            cfg.n_epochs = args.epochs
        generator = train_cgan(img, cfg, device)
        # CGAN outputs are in [-1, 1]; normalize=True rescales each image to [0, 1].
        return sample_cgan(generator, cfg, args.n_samples, device), True

    cfg = VAEConfig()
    if args.method == "vae":
        if args.epochs:
            cfg.n_epochs = args.epochs
        vae = train_vae(img, cfg, device)
        return sample_vae(vae, img.shape, args.n_samples, device), False

    ssim_cfg = SSIMConfig(ssim_min=args.ssim_min, ssim_max=args.ssim_max)
    if args.epochs:
        ssim_cfg.n_epochs = args.epochs
    reference = preprocess_img(path)
    vae = train_vae(img, cfg, device, reference=reference, ssim_cfg=ssim_cfg)
    return sample_vae_ssim(vae, img.shape, args.n_samples, reference, ssim_cfg, device), False


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    args.output.mkdir(parents=True, exist_ok=True)

    images = sorted(args.input.glob(args.pattern))
    if not images:
        raise SystemExit(f"No images matching {args.pattern} in {args.input}")

    for i, path in enumerate(images, 1):
        out_paths = [args.output / f"generated_{path.stem}_{k}.png" for k in range(args.n_samples)]
        if all(p.exists() for p in out_paths):
            continue  # resume an interrupted run
        print(f"[{i}/{len(images)}] {args.method}: {path.name}")
        samples, normalize = generate(path, args, device)
        for sample, out in zip(samples, out_paths):
            save_image(sample, out, normalize=normalize)


if __name__ == "__main__":
    main()
