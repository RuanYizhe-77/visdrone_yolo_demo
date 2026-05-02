import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VisDroneDetectionDataset, collate_fn
from src.losses import CenterNetLoss
from src.model import TinyCenterNet
from src.utils import ensure_dir, get_device, save_checkpoint, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything()
    device = get_device(args.device)
    dataset = VisDroneDetectionDataset(args.data_root, "train", args.img_size, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    model = TinyCenterNet(num_classes=10).to(device)
    criterion = CenterNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ensure_dir("checkpoints")
    ensure_dir(args.output_dir)
    log_path = Path(args.output_dir) / "train_log.csv"
    best = float("inf")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "heatmap_loss", "wh_loss", "offset_loss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            sums = {"loss": 0.0, "heatmap_loss": 0.0, "wh_loss": 0.0, "offset_loss": 0.0}
            for imgs, targets, _, _ in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
                imgs = imgs.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                optimizer.zero_grad(set_to_none=True)
                losses = criterion(model(imgs), targets)
                losses["loss"].backward()
                optimizer.step()
                for k in sums:
                    sums[k] += float(losses[k])
            avg = {k: v / max(len(loader), 1) for k, v in sums.items()}
            row = {"epoch": epoch, **avg}
            writer.writerow(row)
            f.flush()
            save_checkpoint("checkpoints/last.pt", model, optimizer, epoch, avg)
            if avg["loss"] < best:
                best = avg["loss"]
                save_checkpoint("checkpoints/best.pt", model, optimizer, epoch, avg)
            print(row)


if __name__ == "__main__":
    main()
