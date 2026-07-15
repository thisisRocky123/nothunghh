import argparse
import json
from pathlib import Path

import torch

from config import CLASS_NAMES
from utils import predict_image


def main():
    parser = argparse.ArgumentParser(description="Run HQNN inference on a chest X-ray image")
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent / "model.pth"), help="Path to the model checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device(args.device)
    prob = predict_image(args.image, checkpoint_path=args.checkpoint, device=str(device))
    pred_idx = int(prob >= 0.5)

    result = {
        "image": args.image,
        "predicted_class": CLASS_NAMES[pred_idx],
        "probability_pneumonia": round(float(prob), 6),
        "probability_normal": round(float(1.0 - prob), 6),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
