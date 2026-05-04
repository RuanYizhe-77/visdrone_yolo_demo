import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling.detector import FCOSLiteDetector
from src.common.checkpoint import load_model_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--output", default="outputs/runs/det_fcos_lite_50e/export/model.onnx")
    p.add_argument("--device", default="cpu")
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
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy,
            args.output,
            input_names=["images"],
            output_names=[
                "p2_logits",
                "p3_logits",
                "p4_logits",
                "p5_logits",
                "p2_bbox",
                "p3_bbox",
                "p4_bbox",
                "p5_bbox",
                "p2_centerness",
                "p3_centerness",
                "p4_centerness",
                "p5_centerness",
            ]
            if args.use_p2
            else [
                "p3_logits",
                "p4_logits",
                "p5_logits",
                "p3_bbox",
                "p4_bbox",
                "p5_bbox",
                "p3_centerness",
                "p4_centerness",
                "p5_centerness",
            ],
            opset_version=12,
        )
        print(f"saved {args.output}")
    except Exception as exc:
        print("ONNX export failed. Check whether onnx is installed and compatible with this PyTorch build.")
        print(f"error: {exc}")


if __name__ == "__main__":
    main()
