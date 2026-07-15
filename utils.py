from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import IMG_SIZE, MEAN, STD
from model import load_model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)

    mean = np.array(MEAN, dtype=np.float32)[:, None, None]
    std = np.array(STD, dtype=np.float32)[:, None, None]

    arr = (arr - mean) / std

    return torch.from_numpy(arr).float().unsqueeze(0)


def predict_image(image_path, checkpoint_path=None, device="cpu"):
    model = load_model(checkpoint_path=checkpoint_path, device=device)
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return float(output.item())
