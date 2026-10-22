"""Image loading, preprocessing and SSIM utilities shared by all scripts."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision import transforms

# Input size of the verification models (DenseNet201, ResNet152V2, Xception).
VERIFIER_SIZE = 224


def preprocess_img(img):
    """Otsu-binarize a signature and resize it to the verifier input size.

    `img` is a file path, a BGR image, or a grayscale image (uint8).
    """
    if isinstance(img, (str, Path)):
        img = cv2.imread(str(img))
        if img is None:
            raise FileNotFoundError(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(img, (VERIFIER_SIZE, VERIFIER_SIZE), interpolation=cv2.INTER_CUBIC)


def to_rgb(img):
    """Stack a grayscale image into 3 channels for ImageNet backbones."""
    return cv2.merge([img, img, img])


def ssim(reference, img):
    """SSIM between two preprocessed (binary, 224x224) signatures."""
    return structural_similarity(reference, img, data_range=255)


def tensor_to_uint8(t):
    """Convert a [0, 1] image tensor of shape (1, H, W) to a uint8 array (H, W)."""
    return t.squeeze(0).mul(255).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()


def load_signature(path, max_size=512, signed=True):
    """Load a signature as a (1, H, W) tensor in [-1, 1] (or [0, 1] if not `signed`).

    Images larger than `max_size` on either side are resized to
    max_size x max_size to keep the fully connected models in memory.
    """
    width, height = Image.open(path).size
    tf = [transforms.Grayscale(), transforms.ToTensor()]
    if signed:
        tf.append(transforms.Normalize((0.5,), (0.5,)))
    if max(width, height) > max_size:
        tf.append(transforms.Resize((max_size, max_size)))
    return transforms.Compose(tf)(Image.open(path))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
