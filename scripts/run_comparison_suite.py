import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.checkpoint import load_model_weights
from src.common.metrics import evaluate_map50
from src.engine.benchmark import benchmark_model
from src.engine.evaluate import evaluate_detector
from src.engine.infer import predict_image
from src.engine.yolo_infer import empty_np_prediction, predict_yolo, yolo_label_mode
from src.modeling.detector import FCOSLiteDetector
from src.tracking.bytetrack_lite import ByteTrackLite
from src.tracking.iou_tracker import SimpleIoUTracker
from src.visdrone.det_dataset import VisDroneDETDataset, detection_collate
from src.visdrone.vid_dataset import list_vid_sequences
from src.visualization.draw import draw_boxes


@dataclass
class CustomSpec:
    name: str
    weights: str
    width: int
    fpn_channels: int
    head_convs: int
    tracker: str
    use_p2: bool = False
    reg_activation: str = "relu"
    regress_normalized: bool = False


def load_custom(spec, device):
    model = FCOSLiteDetector(
        width=spec.width,
        fpn_channels=spec.fpn_channels,
        head_convs=spec.head_convs,
        use_p2=spec.use_p2,
        reg_activation=spec.reg_activation,
        regress_normalized=spec.regress_normalized,
    ).to(device)
    load_model_weights(model, spec.weights, device)
    model.eval()
    return model


def load_yolo(weights):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"YOLO comparison skipped; ultralytics import failed: {exc}")
        return None
    return YOLO(weights)


def make_grid(images, titles):
    h = max(img.shape[0] for img in images)
    total_w = sum(img.shape[1] for img in images)
    canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
    x = 0
    for image, title in zip(images, titles):
        canvas[: image.shape[0], x : x + image.shape[1]] = image
        label = title.replace("_", " ")
        cv2.rectangle(canvas, (x, 0), (x + image.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2, cv2.LINE_AA)
        x += image.shape[1]
    return canvas


def make_tracker(name, conf, low_conf, new_track_conf, track_iou):
    if name == "bytetrack":
        return ByteTrackLite(high_thresh=conf, low_thresh=low_conf, new_track_thresh=new_track_conf, iou_thresh=track_iou)
    return SimpleIoUTracker(iou_thresh=track_iou)


def draw_prediction(image, pred, track_ids=None):
    return draw_boxes(image, pred["boxes"], pred["labels"], pred["scores"], track_ids=track_ids)


def run_image_comparison(args, old_model, new_model, yolo):
    out_dir = Path(args.output_dir) / "images_old_new_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.image_source)
    images = [source] if source.is_file() else sorted(source.glob("*.jpg"))
    if args.max_images:
        images = images[: args.max_images]
    rows = []
    for image_path in tqdm(images, desc="image comparison"):
        image = cv2.imread(str(image_path))
        old_pred = predict_image(
            old_model,
            image,
            args.img_size,
            args.device,
            args.conf,
            args.iou,
            args.max_detections,
            args.old_nms_mode,
            args.agnostic_nms_iou,
        )
        new_pred = predict_image(
            new_model,
            image,
            args.img_size,
            args.device,
            args.conf,
            args.iou,
            args.max_detections,
            args.new_nms_mode,
            args.agnostic_nms_iou,
        )
        yolo_pred = predict_yolo(yolo, image, args.img_size, args.conf, args.device, args.iou, args.max_detections)
        panels = [
            draw_prediction(image, {k: old_pred[k].cpu().numpy() for k in ("boxes", "scores", "labels")}),
            draw_prediction(image, {k: new_pred[k].cpu().numpy() for k in ("boxes", "scores", "labels")}),
            draw_prediction(image, yolo_pred),
        ]
        titles = [args.old_name, args.new_name, "pretrained_yolo"]
        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), make_grid(panels, titles))
        rows.append(
            {
                "image": image_path.name,
                args.old_name: int(len(old_pred["boxes"])),
                args.new_name: int(len(new_pred["boxes"])),
                "pretrained_yolo": int(len(yolo_pred["boxes"])),
            }
        )
    write_csv(out_dir / "image_detection_counts.csv", rows, ["image", args.old_name, args.new_name, "pretrained_yolo"])
    return {"output_dir": str(out_dir), "num_images": len(images)}


def run_custom_metrics(args, model_name, model):
    ds = VisDroneDETDataset(args.data_root, "val", args.img_size, args.max_eval_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate)
    nms_mode = args.old_nms_mode if model_name == args.old_name else args.new_nms_mode
    return evaluate_detector(
        model,
        loader,
        device=args.device,
        conf=args.eval_conf,
        iou=args.iou,
        max_detections=args.max_detections,
        nms_mode=nms_mode,
        agnostic_nms_thresh=args.agnostic_nms_iou,
    )


def run_yolo_metrics(args, yolo):
    if yolo is None:
        return {"error": "YOLO unavailable"}
    ds = VisDroneDETDataset(args.data_root, "val", args.img_size, args.max_eval_samples)
    predictions, targets = [], []
    for _, target in tqdm(ds, desc="evaluating yolo"):
        image = cv2.imread(target["path"])
        pred = predict_yolo(yolo, image, args.img_size, args.eval_conf, args.device, args.iou, args.max_detections)
        h0, w0 = target["orig_size"].tolist()
        if len(pred["boxes"]):
            pred["boxes"][:, [0, 2]] *= args.img_size / float(w0)
            pred["boxes"][:, [1, 3]] *= args.img_size / float(h0)
        predictions.append({k: torch.as_tensor(v) for k, v in pred.items()})
        targets.append({"boxes": target["boxes"], "labels": target["labels"]})
    metrics = evaluate_map50(predictions, targets)
    mode = yolo_label_mode(yolo)
    if mode == "visdrone":
        metrics["note"] += " YOLO labels are native VisDrone labels."
    elif mode == "coco":
        metrics["note"] += " YOLO COCO labels are mapped approximately to VisDrone labels."
    else:
        metrics["note"] += " YOLO labels were treated as class IDs 0-9; verify class names."
    return metrics


def run_score_comparison(args, old_model, new_model, yolo):
    out_dir = Path(args.output_dir) / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        args.old_name: run_custom_metrics(args, args.old_name, old_model),
        args.new_name: run_custom_metrics(args, args.new_name, new_model),
        "pretrained_yolo": run_yolo_metrics(args, yolo),
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    rows = []
    for name, item in metrics.items():
        rows.append(
            {
                "model": name,
                "precision": item.get("precision", ""),
                "recall": item.get("recall", ""),
                "mAP50_approx": item.get("mAP50_approx", ""),
                "avg_predictions_per_image": item.get("avg_predictions_per_image", ""),
                "note": item.get("note", item.get("error", "")),
            }
        )
    write_csv(out_dir / "metrics_summary.csv", rows, ["model", "precision", "recall", "mAP50_approx", "avg_predictions_per_image", "note"])
    return {"output_dir": str(out_dir), "metrics": metrics}


def benchmark_yolo(args, yolo):
    if yolo is None:
        return {"error": "YOLO unavailable"}
    image = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
    for _ in range(args.benchmark_warmup):
        yolo.predict(image, imgsz=args.img_size, conf=args.conf, device=args.device, verbose=False)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.benchmark_iters):
        yolo.predict(image, imgsz=args.img_size, conf=args.conf, device=args.device, verbose=False)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0 / args.benchmark_iters
    return {"batch_size": 1, "img_size": args.img_size, "latency_ms": latency_ms, "fps": 1000.0 / latency_ms}


def run_benchmark_comparison(args, old_model, new_model, yolo):
    out_dir = Path(args.output_dir) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        args.old_name: benchmark_model(
            old_model, args.img_size, args.benchmark_batch_size, args.device, args.benchmark_warmup, args.benchmark_iters
        ),
        args.new_name: benchmark_model(
            new_model, args.img_size, args.benchmark_batch_size, args.device, args.benchmark_warmup, args.benchmark_iters
        ),
        "pretrained_yolo": benchmark_yolo(args, yolo),
    }
    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    rows = []
    for name, item in results.items():
        rows.append(
            {
                "model": name,
                "batch_size": item.get("batch_size", ""),
                "img_size": item.get("img_size", ""),
                "latency_ms": item.get("latency_ms", ""),
                "fps": item.get("fps", ""),
                "error": item.get("error", ""),
            }
        )
    write_csv(out_dir / "benchmark_summary.csv", rows, ["model", "batch_size", "img_size", "latency_ms", "fps", "error"])
    return {"output_dir": str(out_dir), "benchmarks": results}


def tracks_to_prediction(tracks):
    return {
        "boxes": np.asarray([t["bbox"] for t in tracks], dtype=np.float32),
        "labels": np.asarray([t["class_id"] for t in tracks], dtype=np.int64),
        "scores": np.asarray([t["score"] for t in tracks], dtype=np.float32),
        "track_ids": np.asarray([t["track_id"] for t in tracks], dtype=np.int64),
    }


def run_video_comparison(args, old_model, new_model, yolo):
    out_dir = Path(args.output_dir) / "tracking_old_new_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = list_vid_sequences(args.data_root, args.video_split)
    if args.sequence:
        sequences = [s for s in sequences if s.name == args.sequence]
    if not sequences:
        raise FileNotFoundError("No matching VisDrone-VID sequence found")
    seq = sequences[0]
    out_video = out_dir / f"{seq.name}_{args.old_name}_vs_{args.new_name}_vs_yolo.mp4"
    out_csv = out_dir / f"{seq.name}_tracks.csv"
    old_tracker = make_tracker(args.old_tracker, args.conf, args.track_low_conf, args.new_track_conf, args.track_iou)
    new_tracker = make_tracker(args.new_tracker, args.conf, args.track_low_conf, args.new_track_conf, args.track_iou)
    yolo_tracker = make_tracker("bytetrack", args.conf, args.track_low_conf, args.new_track_conf, args.track_iou)
    writer = None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["model", "frame", "track_id", "class_id", "score", "x1", "y1", "x2", "y2"])
        for frame_idx, _, frame, _ in tqdm(seq.iter_frames(), desc="video comparison"):
            old_conf = args.track_low_conf if args.old_tracker == "bytetrack" else args.conf
            new_conf = args.track_low_conf if args.new_tracker == "bytetrack" else args.conf
            old_pred = predict_image(
                old_model,
                frame,
                args.img_size,
                args.device,
                old_conf,
                args.iou,
                args.max_detections,
                args.old_nms_mode,
                args.agnostic_nms_iou,
            )
            new_pred = predict_image(
                new_model,
                frame,
                args.img_size,
                args.device,
                new_conf,
                args.iou,
                args.max_detections,
                args.new_nms_mode,
                args.agnostic_nms_iou,
            )
            yolo_pred = predict_yolo(yolo, frame, args.img_size, args.track_low_conf, args.device, args.iou, args.max_detections)
            old_tracks = old_tracker.update({k: old_pred[k].cpu().numpy() for k in ("boxes", "scores", "labels")})
            new_tracks = new_tracker.update({k: new_pred[k].cpu().numpy() for k in ("boxes", "scores", "labels")})
            yolo_tracks = yolo_tracker.update(yolo_pred)
            old_vis = tracks_to_prediction(old_tracks)
            new_vis = tracks_to_prediction(new_tracks)
            yolo_vis = tracks_to_prediction(yolo_tracks)
            panels = [
                draw_boxes(frame, old_vis["boxes"], old_vis["labels"], old_vis["scores"], old_vis["track_ids"]),
                draw_boxes(frame, new_vis["boxes"], new_vis["labels"], new_vis["scores"], new_vis["track_ids"]),
                draw_boxes(frame, yolo_vis["boxes"], yolo_vis["labels"], yolo_vis["scores"], yolo_vis["track_ids"]),
            ]
            side = make_grid(panels, [f"{args.old_name}_{args.old_tracker}", f"{args.new_name}_{args.new_tracker}", "pretrained_yolo_bytetrack"])
            if writer is None:
                h, w = side.shape[:2]
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            writer.write(side)
            for name, tracks in ((args.old_name, old_tracks), (args.new_name, new_tracks), ("pretrained_yolo", yolo_tracks)):
                for t in tracks:
                    csv_writer.writerow([name, frame_idx, t["track_id"], t["class_id"], f"{t['score']:.4f}", *[f"{v:.1f}" for v in t["bbox"]]])
            if args.max_frames and frame_idx >= args.max_frames:
                break
    if writer:
        writer.release()
    return {"video": str(out_video), "tracks_csv": str(out_csv), "sequence": seq.name}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(args, summary):
    path = Path(args.output_dir) / "comparison_report.md"
    metrics = summary.get("scores", {}).get("metrics", {})
    benches = summary.get("benchmarks", {}).get("benchmarks", {})
    lines = [
        "# Comparison Suite Report",
        "",
        "This report compares the old custom detector, the new custom detector, and an optional pretrained YOLO baseline.",
        "",
        f"YOLO label mode: `{summary.get('yolo_label_mode', 'unknown')}`.",
        "",
        "## Outputs",
        "",
        f"- Image comparisons: `{summary.get('images', {}).get('output_dir', '')}`",
        f"- Video comparison: `{summary.get('video', {}).get('video', '')}`",
        f"- Track CSV: `{summary.get('video', {}).get('tracks_csv', '')}`",
        f"- Score tables: `{summary.get('scores', {}).get('output_dir', '')}`",
        f"- Benchmark tables: `{summary.get('benchmarks', {}).get('output_dir', '')}`",
        "",
        "## Detection Scores",
        "",
        "| model | precision | recall | mAP50 approx | avg boxes/image |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in metrics.items():
        lines.append(
            f"| {name} | {fmt(item.get('precision'))} | {fmt(item.get('recall'))} | {fmt(item.get('mAP50_approx'))} | {fmt(item.get('avg_predictions_per_image'))} |"
        )
    lines.extend(["", "## Speed", "", "| model | latency ms | fps |", "| --- | ---: | ---: |"])
    for name, item in benches.items():
        lines.append(f"| {name} | {fmt(item.get('latency_ms'))} | {fmt(item.get('fps'))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def fmt(value):
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--output-dir", default="assets/comparison_suite")
    p.add_argument("--old-name", default="old_fcos_lite_50e")
    p.add_argument("--new-name", default="new_fcos_lite_w96_100e")
    p.add_argument("--old-weights", required=True)
    p.add_argument("--new-weights", required=True)
    p.add_argument("--old-width", type=int, default=64)
    p.add_argument("--new-width", type=int, default=96)
    p.add_argument("--old-fpn-channels", type=int, default=160)
    p.add_argument("--new-fpn-channels", type=int, default=224)
    p.add_argument("--old-head-convs", type=int, default=3)
    p.add_argument("--new-head-convs", type=int, default=4)
    p.add_argument("--old-use-p2", action="store_true")
    p.add_argument("--new-use-p2", action="store_true")
    p.add_argument("--old-reg-activation", choices=["relu", "softplus"], default="relu")
    p.add_argument("--new-reg-activation", choices=["relu", "softplus"], default="relu")
    p.add_argument("--old-regress-normalized", action="store_true")
    p.add_argument("--new-regress-normalized", action="store_true")
    p.add_argument("--old-tracker", choices=["iou", "bytetrack"], default="iou")
    p.add_argument("--new-tracker", choices=["iou", "bytetrack"], default="bytetrack")
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--image-source", default="datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/images")
    p.add_argument("--video-split", default="val")
    p.add_argument("--sequence", default=None)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--eval-conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--track-low-conf", type=float, default=0.08)
    p.add_argument("--new-track-conf", type=float, default=0.45)
    p.add_argument("--track-iou", type=float, default=0.3)
    p.add_argument("--max-detections", type=int, default=120)
    p.add_argument("--old-nms-mode", choices=["classwise", "agnostic", "classwise_then_agnostic"], default="classwise")
    p.add_argument("--new-nms-mode", choices=["classwise", "agnostic", "classwise_then_agnostic"], default="classwise")
    p.add_argument("--agnostic-nms-iou", type=float, default=0.7)
    p.add_argument("--max-images", type=int, default=120)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--benchmark-batch-size", type=int, default=1)
    p.add_argument("--benchmark-warmup", type=int, default=10)
    p.add_argument("--benchmark-iters", type=int, default=50)
    p.add_argument("--skip-images", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-scores", action="store_true")
    p.add_argument("--skip-benchmarks", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    old_spec = CustomSpec(
        args.old_name,
        args.old_weights,
        args.old_width,
        args.old_fpn_channels,
        args.old_head_convs,
        args.old_tracker,
        args.old_use_p2,
        args.old_reg_activation,
        args.old_regress_normalized,
    )
    new_spec = CustomSpec(
        args.new_name,
        args.new_weights,
        args.new_width,
        args.new_fpn_channels,
        args.new_head_convs,
        args.new_tracker,
        args.new_use_p2,
        args.new_reg_activation,
        args.new_regress_normalized,
    )
    old_model = load_custom(old_spec, args.device)
    new_model = load_custom(new_spec, args.device)
    yolo = load_yolo(args.yolo_weights)
    summary = {
        "old": old_spec.__dict__,
        "new": new_spec.__dict__,
        "yolo_weights": args.yolo_weights,
        "yolo_label_mode": yolo_label_mode(yolo) if yolo is not None else "unavailable",
        "note": "YOLO labels are native VisDrone labels when a VisDrone-trained checkpoint is used; COCO labels are mapped only for COCO checkpoints.",
    }
    if not args.skip_images:
        summary["images"] = run_image_comparison(args, old_model, new_model, yolo)
    if not args.skip_video:
        summary["video"] = run_video_comparison(args, old_model, new_model, yolo)
    if not args.skip_scores:
        summary["scores"] = run_score_comparison(args, old_model, new_model, yolo)
    if not args.skip_benchmarks:
        summary["benchmarks"] = run_benchmark_comparison(args, old_model, new_model, yolo)
    summary["report"] = write_report(args, summary)
    with open(Path(args.output_dir) / "comparison_suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
