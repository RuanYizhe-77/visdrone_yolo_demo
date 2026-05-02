import argparse
import time
from pathlib import Path

import cv2
import torch

from src.decode import decode_predictions
from src.model import TinyCenterNet
from src.utils import ensure_dir, get_device, load_checkpoint, preprocess_bgr, scale_boxes_to_original
from src.visualize import draw_detections


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="checkpoints/best.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="outputs/video_detected.mp4")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    model = TinyCenterNet(num_classes=10).to(device)
    load_checkpoint(model, args.weights, device)
    model.eval()
    ensure_dir(Path(args.output).parent)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, infer_time = 0, 0.0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            tensor, scale = preprocess_bgr(frame, args.img_size, device)
            det = decode_predictions(model(tensor), args.conf, args.iou, model.stride)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.perf_counter() - t0
            boxes = scale_boxes_to_original(det["boxes"].cpu(), scale).numpy()
            out = draw_detections(frame, boxes, det["scores"].cpu().numpy(), det["labels"].cpu().numpy())
            writer.write(out)
            frames += 1

    cap.release()
    writer.release()
    avg = infer_time / max(frames, 1)
    print(f"saved {args.output}")
    print(f"frames={frames} avg_inference={avg:.4f}s fps={1.0 / max(avg, 1e-9):.2f}")


if __name__ == "__main__":
    main()
