import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.checkpoint import load_model_weights
from src.engine.infer import predict_image
from src.modeling.detector import FCOSLiteDetector
from src.tracking.bytetrack_lite import ByteTrackLite
from src.tracking.iou_tracker import SimpleIoUTracker
from src.visdrone.vid_dataset import list_vid_sequences
from src.visualization.draw import draw_boxes, make_side_by_side


def make_tracker(name, conf, low_conf, new_track_conf, track_iou):
    if name == "bytetrack":
        return ByteTrackLite(high_thresh=conf, low_thresh=low_conf, new_track_thresh=new_track_conf, iou_thresh=track_iou)
    return SimpleIoUTracker(iou_thresh=track_iou)


def load_detector(weights, width, fpn_channels, head_convs, use_p2, reg_activation, regress_normalized, device):
    model = FCOSLiteDetector(
        width=width,
        fpn_channels=fpn_channels,
        head_convs=head_convs,
        use_p2=use_p2,
        reg_activation=reg_activation,
        regress_normalized=regress_normalized,
    ).to(device)
    load_model_weights(model, weights, device)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--split", default="val")
    p.add_argument("--sequence", default=None)
    p.add_argument("--left-weights", required=True)
    p.add_argument("--right-weights", required=True)
    p.add_argument("--left-title", default="baseline")
    p.add_argument("--right-title", default="improved")
    p.add_argument("--left-width", type=int, default=64)
    p.add_argument("--right-width", type=int, default=96)
    p.add_argument("--left-fpn-channels", type=int, default=160)
    p.add_argument("--right-fpn-channels", type=int, default=224)
    p.add_argument("--left-head-convs", type=int, default=3)
    p.add_argument("--right-head-convs", type=int, default=4)
    p.add_argument("--left-use-p2", action="store_true")
    p.add_argument("--right-use-p2", action="store_true")
    p.add_argument("--left-reg-activation", choices=["relu", "softplus"], default="relu")
    p.add_argument("--right-reg-activation", choices=["relu", "softplus"], default="relu")
    p.add_argument("--left-regress-normalized", action="store_true")
    p.add_argument("--right-regress-normalized", action="store_true")
    p.add_argument("--left-tracker", choices=["iou", "bytetrack"], default="iou")
    p.add_argument("--right-tracker", choices=["iou", "bytetrack"], default="bytetrack")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--track-low-conf", type=float, default=0.08)
    p.add_argument("--new-track-conf", type=float, default=0.45)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--track-iou", type=float, default=0.3)
    p.add_argument("--max-detections", type=int, default=120)
    p.add_argument("--left-nms-mode", choices=["classwise", "agnostic", "classwise_then_agnostic"], default="classwise")
    p.add_argument("--right-nms-mode", choices=["classwise", "agnostic", "classwise_then_agnostic"], default="classwise")
    p.add_argument("--agnostic-nms-iou", type=float, default=0.7)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--output-dir", default="assets/tracking_comparisons")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    left = load_detector(
        args.left_weights,
        args.left_width,
        args.left_fpn_channels,
        args.left_head_convs,
        args.left_use_p2,
        args.left_reg_activation,
        args.left_regress_normalized,
        device,
    )
    right = load_detector(
        args.right_weights,
        args.right_width,
        args.right_fpn_channels,
        args.right_head_convs,
        args.right_use_p2,
        args.right_reg_activation,
        args.right_regress_normalized,
        device,
    )
    left_tracker = make_tracker(args.left_tracker, args.conf, args.track_low_conf, args.new_track_conf, args.track_iou)
    right_tracker = make_tracker(args.right_tracker, args.conf, args.track_low_conf, args.new_track_conf, args.track_iou)

    sequences = list_vid_sequences(args.data_root, args.split)
    if args.sequence:
        sequences = [s for s in sequences if s.name == args.sequence]
    if not sequences:
        raise FileNotFoundError("No matching VisDrone-VID sequence found")
    seq = sequences[0]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{seq.name}_{args.left_title}_vs_{args.right_title}.mp4"
    out_csv = out_dir / f"{seq.name}_{args.left_title}_vs_{args.right_title}.csv"

    writer = None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["side", "frame", "track_id", "class_id", "score", "x1", "y1", "x2", "y2"])
        for frame_idx, _, frame, _ in seq.iter_frames():
            low_conf = args.track_low_conf
            left_decode_conf = low_conf if args.left_tracker == "bytetrack" else args.conf
            right_decode_conf = low_conf if args.right_tracker == "bytetrack" else args.conf
            left_pred = predict_image(
                left,
                frame,
                args.img_size,
                device,
                left_decode_conf,
                args.iou,
                args.max_detections,
                args.left_nms_mode,
                args.agnostic_nms_iou,
            )
            right_pred = predict_image(
                right,
                frame,
                args.img_size,
                device,
                right_decode_conf,
                args.iou,
                args.max_detections,
                args.right_nms_mode,
                args.agnostic_nms_iou,
            )
            left_tracks = left_tracker.update({k: left_pred[k].numpy() for k in ("boxes", "scores", "labels")})
            right_tracks = right_tracker.update({k: right_pred[k].numpy() for k in ("boxes", "scores", "labels")})
            left_vis = draw_boxes(
                frame,
                [t["bbox"] for t in left_tracks],
                [t["class_id"] for t in left_tracks],
                [t["score"] for t in left_tracks],
                [t["track_id"] for t in left_tracks],
            )
            right_vis = draw_boxes(
                frame,
                [t["bbox"] for t in right_tracks],
                [t["class_id"] for t in right_tracks],
                [t["score"] for t in right_tracks],
                [t["track_id"] for t in right_tracks],
            )
            side = make_side_by_side(left_vis, right_vis, args.left_title, args.right_title)
            if writer is None:
                h, w = side.shape[:2]
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            writer.write(side)
            for side_name, tracks in ((args.left_title, left_tracks), (args.right_title, right_tracks)):
                for t in tracks:
                    csv_writer.writerow(
                        [side_name, frame_idx, t["track_id"], t["class_id"], f"{t['score']:.4f}", *[f"{v:.1f}" for v in t["bbox"]]]
                    )
            if args.max_frames and frame_idx >= args.max_frames:
                break

    if writer:
        writer.release()
    print(f"saved {out_video}")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
