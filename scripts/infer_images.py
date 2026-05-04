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
    p.add_argument("--max-detections", type=int, default=100)
    p.add_argument("--nms-mode", choices=["classwise", "agnostic", "classwise_then_agnostic"], default="classwise")
    p.add_argument("--agnostic-nms-iou", type=float, default=0.7)
    p.add_argument("--output-dir", default="assets/predictions")
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
    for image_path in iter_images(args.source):
        out_path, pred = save_prediction_image(
            model,
            image_path,
            args.output_dir,
            args.img_size,
            device,
            args.conf,
            args.iou,
            args.max_detections,
            args.nms_mode,
            args.agnostic_nms_iou,
        )
        print(f"saved {out_path} ({len(pred['boxes'])} detections, {pred['infer_ms']:.1f} ms)")


if __name__ == "__main__":
    main()
