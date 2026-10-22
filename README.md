# Deep Generative Attacks and Countermeasures for Offline Signature Verification

Code for the IJCB 2024 paper **Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification**
([IEEE](https://ieeexplore.ieee.org/document/10744496) · [arXiv](https://arxiv.org/abs/2312.00987)).

An Ngo, Rajesh Kumar, Phuong Cao

## Overview

Deep learning-based signature verification (DASV) systems are usually tested against human forgeries. We ask how they hold up against forgeries made by deep generative models, and how to defend them.

- **Attack.** We train a Conditional GAN (CGAN) and a Variational Autoencoder (VAE) on individual skilled forgeries from CEDAR, BHSig260-Bengali and BHSig260-Hindi to produce synthetic forgeries. Against DenseNet201, ResNet152V2 and Xception verifiers, these attacks raise the false acceptance rate (FAR) from about 3% to about 25%.
- **Explanation.** The structural similarity (SSIM) between a synthetic forgery and its source is strongly negatively correlated with FAR. We use this to control attack strength (VAE-SSIM).
- **Countermeasure.** Retraining verifiers with synthetic forgeries brings the attack success rate back under 3%.

## Repository layout

```
sigattack/
  data.py        image loading, Otsu binarization, SSIM
  cgan.py        conditional GAN (generator, discriminator, training)
  vae.py         VAE and SSIM-controlled sampling
generate.py      train a generative model per forgery and save synthetic forgeries
preprocess.py    turn CEDAR + synthetic images into .npy arrays
evaluate.py      train writer-dependent verifiers and report FAR / FRR
```

## Setup

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended. The generative models are fully connected and run on signatures up to 512x512 pixels.

## Data

Download CEDAR from [Kaggle](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) and arrange it as:

```
data/CEDAR/
  full_org/   original_<writer>_<sample>.png    (55 writers x 24 genuine)
  full_forg/  forgeries_<writer>_<sample>.png   (55 writers x 24 skilled forgeries)
```

The synthetic forgeries used in the paper (CEDAR and BHSig260) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1KbbJ5pCx9CVjlFgt28j4bY9KaFcELHb_), so you can skip step 1 below. Each folder contains 9 synthetic images per skilled forgery.

## Usage

### 1. Generate synthetic forgeries

A separate model is trained on each skilled forgery, then sampled 9 times.

```bash
python generate.py --method cgan     --input data/CEDAR/full_forg --output generated/cgan_CEDAR_forg
python generate.py --method vae      --input data/CEDAR/full_forg --output generated/vae_CEDAR_forg
python generate.py --method vae-ssim --input data/CEDAR/full_forg --output generated/vaessim_CEDAR_forg
```

`vae-ssim` stops training once a sample's SSIM to the source forgery falls inside `[--ssim-min, --ssim-max]` (default 0.03–0.2). It then keeps only samples inside that band. Interrupted runs resume where they stopped.

### 2. Preprocess

Signatures are Otsu-binarized, resized to 224x224 and stacked into arrays:

```bash
python preprocess.py --cedar data/CEDAR --generated generated/cgan_CEDAR_forg --out data
```

This writes `data/CEDAR_genuine.npy`, `data/CEDAR_forg.npy` and `data/cgan_CEDAR_forg.npy`, and prints the average SSIM between synthetic forgeries and their sources.

### 3. Evaluate attacks and the countermeasure

For each writer, a frozen ImageNet backbone with a dense head is trained on 16 genuine signatures and 16 skilled forgeries. It is then tested on the remaining 8 + 8 and on the 72 synthetic forgeries made from the held-out forgeries.

```bash
# Baseline verifier under attack
python evaluate.py --synthetic data/cgan_CEDAR_forg.npy --out results/cgan.csv

# Countermeasure: add one synthetic forgery per training forgery as a negative
python evaluate.py --synthetic data/cgan_CEDAR_forg.npy --retrain --out results/cgan_retrain.csv
```

Use `--backbone resnet152v2` or `--backbone xception` for the other verifiers. The CSV lists FAR, FRR and synthetic FAR per writer at `--threshold` (default 0.5), plus the mean.

The same protocol applies to BHSig260 after arranging its images with the same naming scheme. Pass `--writers` and `--samples` to match.

## Citation

```bibtex
@inproceedings{ngo2024deep,
  title     = {Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification},
  author    = {Ngo, An and Kumar, Rajesh and Cao, Phuong},
  booktitle = {2024 IEEE International Joint Conference on Biometrics (IJCB)},
  pages     = {1--10},
  year      = {2024},
  doi       = {10.1109/IJCB62174.2024.10744496}
}
```
