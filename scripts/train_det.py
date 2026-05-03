import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.evaluate import evaluate_detector
from src.engine.train import train_detector
from src.modeling.detector import FCOSLiteDetector
from src.common.seed import seed_everything
from src.visdrone.det_dataset import VisDroneDETDataset, detection_collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--run-dir", default="outputs/runs/det_fcos_lite_50e")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--val-max-samples", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--width", type=int, default=64)
    args = p.parse_args()

    seed_everything(42)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    train_ds = VisDroneDETDataset(args.data_root, "train", args.img_size, args.max_samples, augment=True)
    val_ds = VisDroneDETDataset(args.data_root, "val", args.img_size, args.val_max_samples, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate)
    model = FCOSLiteDetector(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def eval_fn():
        epoch = len(list((Path(args.run_dir) / "checkpoints").glob("last.pt")))  # placeholder, train loop handles cadence externally
        return evaluate_detector(model, val_loader, device=device, conf=0.05, output_json=Path(args.run_dir) / "metrics" / "latest_val.json")

    # Full validation every epoch is expensive; disable callback unless explicitly frequent for short smoke runs.
    callback = eval_fn if args.eval_every == 1 else None
    train_detector(model, train_loader, val_loader, optimizer, args.epochs, device, args.run_dir, eval_fn=callback)
    if args.eval_every != 1:
        metrics = evaluate_detector(model, val_loader, device=device, conf=0.05, output_json=Path(args.run_dir) / "metrics" / "val_metrics.json")
        print(metrics)


if __name__ == "__main__":
    main()
