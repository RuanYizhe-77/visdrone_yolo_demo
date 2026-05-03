import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.infer import save_prediction_image
from src.modeling.detector import FCOSLiteDetector
from src.common.checkpoint import load_model_weights


def iter_images(source):
    p = Path(source)
    if p.is_file():
        return [p]
    return sorted(x for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--output-dir", default="assets/predictions")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = FCOSLiteDetector().to(device)
    load_model_weights(model, args.weights, device)
    for image_path in iter_images(args.source):
        out_path, pred = save_prediction_image(model, image_path, args.output_dir, args.img_size, device, args.conf, args.iou)
        print(f"saved {out_path} ({len(pred['boxes'])} detections, {pred['infer_ms']:.1f} ms)")


if __name__ == "__main__":
    main()
