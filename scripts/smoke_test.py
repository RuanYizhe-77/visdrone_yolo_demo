import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.infer import save_prediction_image
from src.modeling.detector import FCOSLiteDetector
from src.modeling.losses import FCOSLoss
from src.common.checkpoint import save_checkpoint
from src.visdrone.det_dataset import VisDroneDETDataset, detection_collate
from src.visdrone.discovery import summarize_visdrone
from src.visualization.draw import draw_boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--device", default="cuda")
    p.add_argument("--img-size", type=int, default=320)
    p.add_argument("--output-dir", default="outputs/runs/smoke")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(summarize_visdrone(args.data_root))
    ds = VisDroneDETDataset(args.data_root, "train", args.img_size, max_samples=8, augment=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=detection_collate)
    images, targets = next(iter(loader))
    model = FCOSLiteDetector(width=32, fpn_channels=96).to(device)
    outputs = model(images.to(device))
    losses = FCOSLoss(model.num_classes)(outputs, targets, model.strides)
    losses["loss"].backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    ckpt = out_dir / "checkpoints" / "smoke.pt"
    save_checkpoint(ckpt, model, opt, 1, {"smoke_loss": float(losses["loss"].detach().cpu())})
    image_path = targets[0]["path"]
    save_prediction_image(model, image_path, out_dir / "predictions", args.img_size, device, conf=0.01)
    print(f"smoke_loss={float(losses['loss'].detach().cpu()):.4f}")
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
