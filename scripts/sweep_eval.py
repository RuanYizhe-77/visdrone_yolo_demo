import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.checkpoint import load_model_weights
from src.common.metrics import evaluate_map50
from src.engine.evaluate import evaluate_detector
from src.engine.yolo_infer import predict_yolo, yolo_label_mode
from src.modeling.detector import FCOSLiteDetector
from src.visdrone.det_dataset import VisDroneDETDataset, detection_collate


def parse_thresholds(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def eval_yolo(yolo, ds, args, conf):
    predictions, targets = [], []
    for _, target in tqdm(ds, desc=f"yolo conf={conf:.2f}"):
        image = cv2.imread(target["path"])
        pred = predict_yolo(yolo, image, args.img_size, conf, args.device, args.iou, args.max_detections)
        h0, w0 = target["orig_size"].tolist()
        if len(pred["boxes"]):
            pred["boxes"][:, [0, 2]] *= args.img_size / float(w0)
            pred["boxes"][:, [1, 3]] *= args.img_size / float(h0)
        predictions.append({k: torch.as_tensor(v) for k, v in pred.items()})
        targets.append({"boxes": target["boxes"], "labels": target["labels"]})
    return evaluate_map50(predictions, targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["custom", "yolo"], required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--thresholds", default="0.05,0.25,0.35,0.45,0.50")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-detections", type=int, default=120)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--fpn-channels", type=int, default=160)
    p.add_argument("--head-convs", type=int, default=3)
    p.add_argument("--use-p2", action="store_true")
    p.add_argument("--reg-activation", choices=["relu", "softplus"], default="relu")
    p.add_argument("--regress-normalized", action="store_true")
    p.add_argument("--nms-mode", choices=["classwise", "agnostic", "classwise_then_agnostic"], default="classwise")
    p.add_argument("--agnostic-nms-iou", type=float, default=0.7)
    args = p.parse_args()

    args.device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    thresholds = parse_thresholds(args.thresholds)
    ds = VisDroneDETDataset(args.data_root, "val", args.img_size, args.max_samples)
    rows = []
    if args.model_type == "custom":
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate)
        model = FCOSLiteDetector(
            width=args.width,
            fpn_channels=args.fpn_channels,
            head_convs=args.head_convs,
            use_p2=args.use_p2,
            reg_activation=args.reg_activation,
            regress_normalized=args.regress_normalized,
        ).to(args.device)
        load_model_weights(model, args.weights, args.device)
        label_mode = "custom_visdrone"
        for conf in thresholds:
            metrics = evaluate_detector(
                model,
                loader,
                device=args.device,
                conf=conf,
                iou=args.iou,
                max_detections=args.max_detections,
                nms_mode=args.nms_mode,
                agnostic_nms_thresh=args.agnostic_nms_iou,
            )
            rows.append(row_from_metrics(args.weights, label_mode, conf, metrics))
    else:
        from ultralytics import YOLO

        yolo = YOLO(args.weights)
        label_mode = yolo_label_mode(yolo)
        for conf in thresholds:
            metrics = eval_yolo(yolo, ds, args, conf)
            rows.append(row_from_metrics(args.weights, label_mode, conf, metrics))

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "weights",
                "label_mode",
                "conf",
                "precision",
                "recall",
                "mAP50_approx",
                "avg_predictions_per_image",
                "num_predictions",
                "num_targets",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def row_from_metrics(weights, label_mode, conf, metrics):
    return {
        "weights": weights,
        "label_mode": label_mode,
        "conf": conf,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "mAP50_approx": metrics.get("mAP50_approx"),
        "avg_predictions_per_image": metrics.get("avg_predictions_per_image"),
        "num_predictions": metrics.get("num_predictions"),
        "num_targets": metrics.get("num_targets"),
    }


if __name__ == "__main__":
    main()
