import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.infer import predict_image
from src.modeling.detector import FCOSLiteDetector
from src.tracking.iou_tracker import SimpleIoUTracker
from src.common.checkpoint import load_model_weights
from src.visdrone.vid_dataset import list_vid_sequences
from src.visualization.draw import draw_boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--sequence", default=None)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--output-dir", default="assets/tracking")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = FCOSLiteDetector().to(device)
    load_model_weights(model, args.weights, device)
    sequences = list_vid_sequences(args.data_root, args.split)
    if args.sequence:
        sequences = [s for s in sequences if s.name == args.sequence]
    if not sequences:
        raise FileNotFoundError("No matching VisDrone-VID sequence found")
    seq = sequences[0]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{seq.name}_tracked.mp4"
    out_csv = out_dir / f"{seq.name}_tracks.csv"
    tracker = SimpleIoUTracker()
    writer = None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "track_id", "class_id", "score", "x1", "y1", "x2", "y2"])
        for frame_idx, _, frame, _ in seq.iter_frames():
            pred = predict_image(model, frame, args.img_size, device, args.conf, args.iou)
            tracks = tracker.update({k: pred[k].numpy() for k in ("boxes", "scores", "labels")})
            boxes = [t["bbox"] for t in tracks]
            labels = [t["class_id"] for t in tracks]
            scores = [t["score"] for t in tracks]
            ids = [t["track_id"] for t in tracks]
            vis = draw_boxes(frame, boxes, labels, scores, ids)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            writer.write(vis)
            for t in tracks:
                csv_writer.writerow([frame_idx, t["track_id"], t["class_id"], f"{t['score']:.4f}", *[f"{v:.1f}" for v in t["bbox"]]])
            if args.max_frames and frame_idx >= args.max_frames:
                break
    if writer:
        writer.release()
    print(f"saved {out_video}")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
