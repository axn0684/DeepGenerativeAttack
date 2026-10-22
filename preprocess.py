"""Preprocess CEDAR (and optionally synthetic forgeries) into .npy arrays.

Outputs, ordered writer by writer, as uint8 arrays of shape (N, 224, 224, 3):
    <out>/CEDAR_genuine.npy   genuine signatures         (writers * samples)
    <out>/CEDAR_forg.npy      skilled forgeries          (writers * samples)
    <out>/<name>.npy          synthetic forgeries        (writers * samples * n_generated)

Example:
    python preprocess.py --cedar data/CEDAR --generated generated/cgan_CEDAR_forg --out data
"""
import argparse
from pathlib import Path

import numpy as np

from sigattack.data import preprocess_img, ssim, to_rgb


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cedar", type=Path, required=True, help="folder containing full_org/ and full_forg/")
    p.add_argument("--generated", type=Path, help="folder of generated_forgeries_<w>_<s>_<k>.png images")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--name", help="output name for the synthetic array (default: generated folder name)")
    p.add_argument("--writers", type=int, default=55)
    p.add_argument("--samples", type=int, default=24, help="signatures per writer")
    p.add_argument("--n-generated", type=int, default=9, help="synthetic images per forgery")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    genuine, forgeries, generated, ssims = [], [], [], []
    for w in range(1, args.writers + 1):
        for s in range(1, args.samples + 1):
            real = preprocess_img(args.cedar / "full_org" / f"original_{w}_{s}.png")
            forg = preprocess_img(args.cedar / "full_forg" / f"forgeries_{w}_{s}.png")
            genuine.append(to_rgb(real))
            forgeries.append(to_rgb(forg))
            if args.generated:
                for k in range(args.n_generated):
                    gen = preprocess_img(args.generated / f"generated_forgeries_{w}_{s}_{k}.png")
                    ssims.append(ssim(forg, gen))
                    generated.append(to_rgb(gen))

    np.save(args.out / "CEDAR_genuine.npy", np.stack(genuine))
    np.save(args.out / "CEDAR_forg.npy", np.stack(forgeries))
    print(f"Saved {len(genuine)} genuine and {len(forgeries)} forged signatures to {args.out}")

    if args.generated:
        name = args.name or args.generated.name
        np.save(args.out / f"{name}.npy", np.stack(generated))
        print(f"Saved {len(generated)} synthetic forgeries to {args.out / name}.npy")
        print(f"Average SSIM to the source forgery: {np.mean(ssims):.4f}")


if __name__ == "__main__":
    main()
