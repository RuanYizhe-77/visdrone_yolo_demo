import csv
from pathlib import Path

import torch
from tqdm import tqdm

from src.modeling.losses import FCOSLoss
from src.common.checkpoint import save_checkpoint
from src.visualization.plots import plot_training_log


def train_detector(model, train_loader, val_loader, optimizer, epochs, device, run_dir, eval_fn=None):
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    log_path = run_dir / "logs" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    loss_fn = FCOSLoss(model.num_classes)
    best_score = -1.0
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "cls_loss", "reg_loss", "center_loss", "num_pos", "val_map50"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            totals = {"loss": 0.0, "cls_loss": 0.0, "reg_loss": 0.0, "center_loss": 0.0, "num_pos": 0.0}
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
            for images, targets in pbar:
                images = images.to(device)
                outputs = model(images)
                losses = loss_fn(outputs, targets, model.strides)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                for key in totals:
                    totals[key] += float(losses[key].detach().cpu())
                pbar.set_postfix(loss=float(losses["loss"].detach().cpu()))
            n = max(len(train_loader), 1)
            row = {k: totals[k] / n for k in totals}
            row["epoch"] = epoch
            row["val_map50"] = ""
            metrics = {}
            if eval_fn is not None:
                metrics = eval_fn()
                row["val_map50"] = metrics.get("mAP50_approx", 0.0)
            writer.writerow(row)
            f.flush()
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, metrics)
            score = float(metrics.get("mAP50_approx", -row["loss"])) if metrics else -row["loss"]
            if score > best_score:
                best_score = score
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, metrics)
            plot_training_log(log_path, run_dir / "curves" / "loss_curve.png")
    return log_path
