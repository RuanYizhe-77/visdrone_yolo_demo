import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.benchmark import benchmark_model
from src.modeling.detector import FCOSLiteDetector
from src.common.checkpoint import load_model_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--output-json", default="outputs/runs/det_fcos_lite_50e/metrics/benchmark.json")
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
    if args.weights:
        load_model_weights(model, args.weights, device)
    metrics = benchmark_model(model, args.img_size, args.batch_size, device, iters=args.iters)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
