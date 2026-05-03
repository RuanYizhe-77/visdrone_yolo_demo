import argparse
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.infer import predict_image
from src.modeling.detector import FCOSLiteDetector
from src.common.checkpoint import load_model_weights
from src.visualization.draw import draw_boxes, make_side_by_side


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours-weights", required=True)
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--output-dir", default="assets/comparisons")
    args = p.parse_args()
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("YOLO comparison is optional and requires the ultralytics package.")
        print(f"Import failed: {exc}")
        return

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    ours = FCOSLiteDetector().to(device)
    load_model_weights(ours, args.ours_weights, device)
    yolo = YOLO(args.yolo_weights)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [Path(args.source)] if Path(args.source).is_file() else sorted(Path(args.source).glob("*.jpg"))
    for image_path in sources:
        image = cv2.imread(str(image_path))
        own = predict_image(ours, image, args.img_size, device, args.conf)
        own_vis = draw_boxes(image, own["boxes"].numpy(), own["labels"].numpy(), own["scores"].numpy())
        result = yolo.predict(str(image_path), imgsz=args.img_size, conf=args.conf, verbose=False)[0]
        yolo_vis = result.plot()
        side = make_side_by_side(own_vis, yolo_vis, "FCOS-lite from scratch", "Pretrained YOLO baseline")
        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), side)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
