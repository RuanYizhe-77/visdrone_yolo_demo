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
from src.engine.yolo_infer import predict_yolo, yolo_label_mode
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
    width: int = 64
    fpn_channels: int = 160
    head_convs: int = 3
    tracker: str = "iou"
    nms_mode: str = "classwise"
    use_p2: bool = False
    reg_activation: str = "relu"
    regress_normalized: bool = False


def load_specs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CustomSpec(**item) for item in data]


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
        print(f"pretrained YOLO comparison unavailable: {exc}")
        return None
    return YOLO(weights)


def make_grid(images, titles):
    h = max(img.shape[0] for img in images)
    w = sum(img.shape[1] for img in images)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x = 0
    for image, title in zip(images, titles):
        canvas[: image.shape[0], x : x + image.shape[1]] = image
        draw_panel_title(canvas, title, x, image.shape[1])
        x += image.shape[1]
    return canvas


def draw_panel_title(canvas, title, x, panel_width):
    label = title.replace("_", " ")
    bar_h = 44
    cv2.rectangle(canvas, (x, 0), (x + panel_width, bar_h), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.78
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    if tw > panel_width - 24:
        scale = max(0.48, scale * (panel_width - 24) / max(tw, 1))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    text_x = x + max(12, (panel_width - tw) // 2)
    text_y = 30
    cv2.putText(canvas, label, (text_x + 1, text_y + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (text_x, text_y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)


def resize_video_frame(frame, max_width):
    if not max_width or frame.shape[1] <= max_width:
        h, w = frame.shape[:2]
        even_w = w - (w % 2)
        even_h = h - (h % 2)
        return frame[:even_h, :even_w] if (even_w != w or even_h != h) else frame
    scale = max_width / float(frame.shape[1])
    new_w = int(frame.shape[1] * scale)
    new_h = int(frame.shape[0] * scale)
    new_w -= new_w % 2
    new_h -= new_h % 2
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_tracker(name, conf, low_conf, new_track_conf, track_iou):
    if name == "stable_bytetrack":
        return ByteTrackLite(
            high_thresh=conf,
            low_thresh=low_conf,
            new_track_thresh=new_track_conf,
            iou_thresh=track_iou,
            max_age=45,
            min_hits=3,
            smooth=0.65,
            use_motion=True,
            display_max_age=2,
        )
    if name == "bytetrack":
        return ByteTrackLite(high_thresh=conf, low_thresh=low_conf, new_track_thresh=new_track_conf, iou_thresh=track_iou)
    return SimpleIoUTracker(iou_thresh=track_iou)


def np_from_custom(pred):
    return {k: pred[k].detach().cpu().numpy() for k in ("boxes", "scores", "labels")}


def tracks_to_prediction(tracks):
    return {
        "boxes": np.asarray([t["bbox"] for t in tracks], dtype=np.float32),
        "labels": np.asarray([t["class_id"] for t in tracks], dtype=np.int64),
        "scores": np.asarray([t["score"] for t in tracks], dtype=np.float32),
        "track_ids": np.asarray([t["track_id"] for t in tracks], dtype=np.int64),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_images(args, custom_items, yolo):
    out_dir = Path(args.output_dir) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.image_source)
    images = [source] if source.is_file() else sorted(source.glob("*.jpg"))
    images = images[: args.max_images] if args.max_images else images
    rows = []
    custom_names = [spec.name for spec, _ in custom_items]
    names = ["pretrained_yolo"] + custom_names if args.yolo_first else custom_names + ["pretrained_yolo"]
    for image_path in tqdm(images, desc="image comparison", disable=args.no_progress):
        image = cv2.imread(str(image_path))
        panels, row = [], {"image": image_path.name}
        for spec, model in custom_items:
            pred = predict_image(
                model,
                image,
                args.img_size,
                args.device,
                args.conf,
                args.iou,
                args.max_detections,
                spec.nms_mode,
                args.agnostic_nms_iou,
            )
            np_pred = np_from_custom(pred)
            panels.append(draw_boxes(image, np_pred["boxes"], np_pred["labels"], np_pred["scores"]))
            row[spec.name] = int(len(np_pred["boxes"]))
        yolo_pred = predict_yolo(yolo, image, args.img_size, args.conf, args.device, args.iou, args.max_detections)
        yolo_panel = draw_boxes(image, yolo_pred["boxes"], yolo_pred["labels"], yolo_pred["scores"])
        panels = [yolo_panel] + panels if args.yolo_first else panels + [yolo_panel]
        row["pretrained_yolo"] = int(len(yolo_pred["boxes"]))
        cv2.imwrite(str(out_dir / image_path.name), make_grid(panels, names))
        rows.append(row)
    write_csv(out_dir / "image_detection_counts.csv", rows, ["image"] + names)
    return {"output_dir": str(out_dir), "num_images": len(images)}


def run_custom_metrics(args, spec, model):
    ds = VisDroneDETDataset(args.data_root, "val", args.img_size, args.max_eval_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate)
    return evaluate_detector(
        model,
        loader,
        device=args.device,
        conf=args.eval_conf,
        iou=args.iou,
        max_detections=args.max_detections,
        nms_mode=spec.nms_mode,
        agnostic_nms_thresh=args.agnostic_nms_iou,
        progress=not args.no_progress,
    )


def run_yolo_metrics(args, yolo):
    if yolo is None:
        return {"error": "YOLO unavailable"}
    ds = VisDroneDETDataset(args.data_root, "val", args.img_size, args.max_eval_samples)
    predictions, targets = [], []
    for _, target in tqdm(ds, desc="evaluating pretrained yolo", disable=args.no_progress):
        image = cv2.imread(target["path"])
        pred = predict_yolo(yolo, image, args.img_size, args.eval_conf, args.device, args.iou, args.max_detections)
        h0, w0 = target["orig_size"].tolist()
        if len(pred["boxes"]):
            pred["boxes"][:, [0, 2]] *= args.img_size / float(w0)
            pred["boxes"][:, [1, 3]] *= args.img_size / float(h0)
        predictions.append({k: torch.as_tensor(v) for k, v in pred.items()})
        targets.append({"boxes": target["boxes"], "labels": target["labels"]})
    return evaluate_map50(predictions, targets)


def run_scores(args, custom_items, yolo):
    out_dir = Path(args.output_dir) / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    custom_metrics = {spec.name: run_custom_metrics(args, spec, model) for spec, model in custom_items}
    yolo_metrics = {"pretrained_yolo": run_yolo_metrics(args, yolo)}
    metrics = {**yolo_metrics, **custom_metrics} if args.yolo_first else {**custom_metrics, **yolo_metrics}
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
                "num_predictions": item.get("num_predictions", ""),
                "num_targets": item.get("num_targets", ""),
                "note": item.get("note", item.get("error", "")),
            }
        )
    write_csv(
        out_dir / "metrics_summary.csv",
        rows,
        ["model", "precision", "recall", "mAP50_approx", "avg_predictions_per_image", "num_predictions", "num_targets", "note"],
    )
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


def run_benchmarks(args, custom_items, yolo):
    out_dir = Path(args.output_dir) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    custom_results = {
        spec.name: benchmark_model(
            model,
            args.img_size,
            args.benchmark_batch_size,
            args.device,
            args.benchmark_warmup,
            args.benchmark_iters,
        )
        for spec, model in custom_items
    }
    yolo_results = {"pretrained_yolo": benchmark_yolo(args, yolo)}
    results = {**yolo_results, **custom_results} if args.yolo_first else {**custom_results, **yolo_results}
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


def run_video(args, custom_items, yolo):
    out_dir = Path(args.output_dir) / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = list_vid_sequences(args.data_root, args.video_split)
    if args.sequence:
        sequences = [s for s in sequences if s.name == args.sequence]
    if not sequences:
        raise FileNotFoundError("No matching VisDrone-VID sequence found")
    seq = sequences[0]
    custom_names = [spec.name for spec, _ in custom_items]
    ordered_names = ["pretrained_yolo"] + custom_names if args.yolo_first else custom_names + ["pretrained_yolo"]
    title = "_vs_".join(ordered_names)
    out_video = out_dir / f"{seq.name}_{title}.mp4"
    out_csv = out_dir / f"{seq.name}_tracks.csv"
    trackers = {
        spec.name: make_tracker(spec.tracker, args.conf, args.track_low_conf, args.new_track_conf, args.track_iou)
        for spec, _ in custom_items
    }
    yolo_tracker = make_tracker("bytetrack", args.conf, args.track_low_conf, args.new_track_conf, args.track_iou)
    writer = None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["model", "frame", "track_id", "class_id", "score", "x1", "y1", "x2", "y2"])
        written_frames = 0
        for frame_idx, _, frame, _ in tqdm(seq.iter_frames(), desc="video comparison", disable=args.no_progress):
            if frame_idx < args.start_frame:
                continue
            panels, titles = [], []
            all_tracks = {}
            for spec, model in custom_items:
                decode_conf = args.track_low_conf if spec.tracker == "bytetrack" else args.conf
                pred = predict_image(
                    model,
                    frame,
                    args.img_size,
                    args.device,
                    decode_conf,
                    args.iou,
                    args.max_detections,
                    spec.nms_mode,
                    args.agnostic_nms_iou,
                )
                tracks = trackers[spec.name].update(np_from_custom(pred))
                vis = tracks_to_prediction(tracks)
                panels.append(draw_boxes(frame, vis["boxes"], vis["labels"], vis["scores"], vis["track_ids"]))
                titles.append(f"{spec.name}_{spec.tracker}")
                all_tracks[spec.name] = tracks
            yolo_pred = predict_yolo(yolo, frame, args.img_size, args.track_low_conf, args.device, args.iou, args.max_detections)
            yolo_tracks = yolo_tracker.update(yolo_pred)
            yolo_vis = tracks_to_prediction(yolo_tracks)
            yolo_panel = draw_boxes(frame, yolo_vis["boxes"], yolo_vis["labels"], yolo_vis["scores"], yolo_vis["track_ids"])
            yolo_title = "pretrained_yolo_bytetrack"
            if args.yolo_first:
                panels = [yolo_panel] + panels
                titles = [yolo_title] + titles
            else:
                panels.append(yolo_panel)
                titles.append(yolo_title)
            if writer is None:
                side = make_grid(panels, titles)
                side = resize_video_frame(side, args.video_max_width)
                h, w = side.shape[:2]
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            else:
                side = make_grid(panels, titles)
                side = resize_video_frame(side, args.video_max_width)
            writer.write(side)
            all_tracks["pretrained_yolo"] = yolo_tracks
            for name, tracks in all_tracks.items():
                for t in tracks:
                    csv_writer.writerow([name, frame_idx, t["track_id"], t["class_id"], f"{t['score']:.4f}", *[f"{v:.1f}" for v in t["bbox"]]])
            written_frames += 1
            if args.max_frames and written_frames >= args.max_frames:
                break
    if writer:
        writer.release()
    return {"video": str(out_video), "tracks_csv": str(out_csv), "sequence": seq.name}


def fmt(value):
    return f"{value:.4f}" if isinstance(value, (float, int)) else ""


def write_report(args, summary, specs, yolo):
    path = Path(args.output_dir) / "comparison_report.md"
    metrics = summary.get("scores", {}).get("metrics", {})
    benches = summary.get("benchmarks", {}).get("benchmarks", {})
    lines = [
        "# Final Comparison Report",
        "",
        "Custom models were trained with the repository's low-level PyTorch FCOS-style code. The YOLO checkpoint is used only as a pretrained VisDrone reference for inference/evaluation.",
        "",
        f"YOLO label mode: `{yolo_label_mode(yolo) if yolo is not None else 'unavailable'}`.",
        "",
        "## Custom Models",
        "",
        "| model | weights | tracker | nms | width | fpn | heads | p2 | normalized boxes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for spec in specs:
        lines.append(
            f"| {spec.name} | `{spec.weights}` | {spec.tracker} | {spec.nms_mode} | {spec.width} | {spec.fpn_channels} | {spec.head_convs} | {spec.use_p2} | {spec.regress_normalized} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Images: `{summary.get('images', {}).get('output_dir', '')}`",
            f"- Video: `{summary.get('video', {}).get('video', '')}`",
            f"- Tracks CSV: `{summary.get('video', {}).get('tracks_csv', '')}`",
            f"- Scores: `{summary.get('scores', {}).get('output_dir', '')}`",
            f"- Benchmarks: `{summary.get('benchmarks', {}).get('output_dir', '')}`",
            "",
            "## Scores",
            "",
            "| model | precision | recall | mAP50 approx | avg boxes/image |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in metrics.items():
        lines.append(
            f"| {name} | {fmt(item.get('precision'))} | {fmt(item.get('recall'))} | {fmt(item.get('mAP50_approx'))} | {fmt(item.get('avg_predictions_per_image'))} |"
        )
    lines.extend(["", "## Speed", "", "| model | latency ms | fps |", "| --- | ---: | ---: |"])
    for name, item in benches.items():
        lines.append(f"| {name} | {fmt(item.get('latency_ms'))} | {fmt(item.get('fps'))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--custom-specs", required=True, help="JSON list of custom model specs.")
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--output-dir", default="assets/final_comparison")
    p.add_argument("--image-source", default="datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/images")
    p.add_argument("--video-split", default="val")
    p.add_argument("--sequence", default=None)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--eval-conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--agnostic-nms-iou", type=float, default=0.6)
    p.add_argument("--track-low-conf", type=float, default=0.08)
    p.add_argument("--new-track-conf", type=float, default=0.45)
    p.add_argument("--track-iou", type=float, default=0.3)
    p.add_argument("--max-detections", type=int, default=120)
    p.add_argument("--max-images", type=int, default=80)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--video-max-width", type=int, default=3840)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--benchmark-batch-size", type=int, default=1)
    p.add_argument("--benchmark-warmup", type=int, default=10)
    p.add_argument("--benchmark-iters", type=int, default=50)
    p.add_argument("--skip-images", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-scores", action="store_true")
    p.add_argument("--skip-benchmarks", action="store_true")
    p.add_argument("--yolo-first", action="store_true", help="Put pretrained YOLO first in image/video/table outputs.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    return p.parse_args()


def main():
    args = parse_args()
    args.device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    specs = load_specs(args.custom_specs)
    custom_items = [(spec, load_custom(spec, args.device)) for spec in specs]
    yolo = load_yolo(args.yolo_weights)
    summary = {
        "custom_specs": [spec.__dict__ for spec in specs],
        "yolo_weights": args.yolo_weights,
        "yolo_label_mode": yolo_label_mode(yolo) if yolo is not None else "unavailable",
    }
    if not args.skip_images:
        summary["images"] = run_images(args, custom_items, yolo)
    if not args.skip_video:
        summary["video"] = run_video(args, custom_items, yolo)
    if not args.skip_scores:
        summary["scores"] = run_scores(args, custom_items, yolo)
    if not args.skip_benchmarks:
        summary["benchmarks"] = run_benchmarks(args, custom_items, yolo)
    summary["report"] = write_report(args, summary, specs, yolo)
    with open(Path(args.output_dir) / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
