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
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = FCOSLiteDetector().to(device)
    if args.weights:
        load_model_weights(model, args.weights, device)
    metrics = benchmark_model(model, args.img_size, args.batch_size, device, iters=args.iters)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
