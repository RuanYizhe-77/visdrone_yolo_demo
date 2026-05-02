import argparse
from pathlib import Path

import torch

from src.model import TinyCenterNet
from src.utils import ensure_dir, get_device, load_checkpoint


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["heatmap"], out["wh"], out["offset"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="checkpoints/best.pt")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="outputs/model.onnx")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    model = TinyCenterNet(num_classes=10).to(device)
    load_checkpoint(model, args.weights, device)
    model.eval()
    export_model = OnnxWrapper(model)
    ensure_dir(Path(args.output).parent)
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    try:
        torch.onnx.export(
            export_model,
            dummy,
            args.output,
            input_names=["images"],
            output_names=["heatmap", "wh", "offset"],
            opset_version=12,
            dynamic_axes={"images": {0: "batch"}, "heatmap": {0: "batch"}, "wh": {0: "batch"}, "offset": {0: "batch"}},
        )
        print(f"saved {args.output}")
    except Exception as exc:
        print("ONNX export failed. Check that the onnx Python package is available and that the weights path is valid.")
        print(f"error: {exc}")


if __name__ == "__main__":
    main()
