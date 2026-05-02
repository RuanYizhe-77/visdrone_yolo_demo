import argparse
from pathlib import Path

import cv2
import torch

from src.decode import decode_predictions
from src.model import TinyCenterNet
from src.utils import ensure_dir, get_device, image_files, load_checkpoint, preprocess_bgr, scale_boxes_to_original
from src.visualize import draw_detections


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="checkpoints/best.pt")
    p.add_argument("--source", required=True, help="Image file or folder")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="outputs/images")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    model = TinyCenterNet(num_classes=10).to(device)
    load_checkpoint(model, args.weights, device)
    model.eval()
    ensure_dir(args.output_dir)

    with torch.no_grad():
        for img_path in image_files(args.source):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Skipping unreadable image: {img_path}")
                continue
            tensor, scale = preprocess_bgr(image, args.img_size, device)
            det = decode_predictions(model(tensor), args.conf, args.iou, model.stride)[0]
            boxes = scale_boxes_to_original(det["boxes"].cpu(), scale).numpy()
            drawn = draw_detections(image, boxes, det["scores"].cpu().numpy(), det["labels"].cpu().numpy())
            out_path = Path(args.output_dir) / img_path.name
            cv2.imwrite(str(out_path), drawn)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
