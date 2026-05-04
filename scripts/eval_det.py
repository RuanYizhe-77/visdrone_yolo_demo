import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.evaluate import evaluate_detector
from src.modeling.detector import FCOSLiteDetector
from src.common.checkpoint import load_model_weights
from src.visdrone.det_dataset import VisDroneDETDataset, detection_collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--weights", required=True)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-detections", type=int, default=200)
    p.add_argument("--nms-mode", choices=["classwise", "agnostic", "classwise_then_agnostic"], default="classwise")
    p.add_argument("--agnostic-nms-iou", type=float, default=0.7)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-json", default="outputs/runs/det_fcos_lite_50e/metrics/val_metrics.json")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--fpn-channels", type=int, default=160)
    p.add_argument("--head-convs", type=int, default=3)
    p.add_argument("--use-p2", action="store_true")
    p.add_argument("--reg-activation", choices=["relu", "softplus"], default="relu")
    p.add_argument("--regress-normalized", action="store_true")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = FCOSLiteDetector(
        width=args.width,
        fpn_channels=args.fpn_channels,
        head_convs=args.head_convs,
        use_p2=args.use_p2,
        reg_activation=args.reg_activation,
        regress_normalized=args.regress_normalized,
    ).to(device)
    load_model_weights(model, args.weights, device)
    ds = VisDroneDETDataset(args.data_root, "val", args.img_size, args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=detection_collate)
    print(
        evaluate_detector(
            model,
            loader,
            device=device,
            conf=args.conf,
            iou=args.iou,
            output_json=args.output_json,
            max_detections=args.max_detections,
            nms_mode=args.nms_mode,
            agnostic_nms_thresh=args.agnostic_nms_iou,
        )
    )


if __name__ == "__main__":
    main()
