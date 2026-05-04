import csv
from pathlib import Path

import torch
from tqdm import tqdm

from src.modeling.losses import FCOSLoss
from src.common.checkpoint import save_checkpoint
from src.visualization.plots import plot_training_log


def _score_from_row(row):
    val_map50 = row.get("val_map50", "")
    if val_map50 not in ("", None):
        try:
            return float(val_map50)
        except ValueError:
            pass
    try:
        return -float(row["loss"])
    except (KeyError, TypeError, ValueError):
        return -float("inf")


def train_detector(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    device,
    run_dir,
    eval_fn=None,
    start_epoch=1,
    resume_log=False,
    eval_every=0,
    patience=0,
    min_epochs=0,
    monitor="mAP50_approx",
    scheduler=None,
    center_radius=1.5,
    min_delta=0.0,
    target_score=None,
    progress=True,
    amp=False,
    accum_steps=1,
):
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    log_path = run_dir / "logs" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    loss_fn = FCOSLoss(
        model.num_classes,
        center_radius=center_radius,
        regress_normalized=getattr(model, "regress_normalized", False),
    )
    best_score = -1.0
    if resume_log and log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            best_score = max(_score_from_row(row) for row in rows)
    stale_evals = 0
    amp_enabled = bool(amp) and str(device).startswith("cuda")
    accum_steps = max(int(accum_steps), 1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    mode = "a" if resume_log and log_path.exists() else "w"
    with open(log_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss",
                "cls_loss",
                "reg_loss",
                "center_loss",
                "num_pos",
                "val_precision",
                "val_recall",
                "val_map50",
                "val_avg_predictions",
                "lr",
            ],
        )
        if mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            totals = {"loss": 0.0, "cls_loss": 0.0, "reg_loss": 0.0, "center_loss": 0.0, "num_pos": 0.0}
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", disable=not progress)
            optimizer.zero_grad(set_to_none=True)
            for step_idx, (images, targets) in enumerate(pbar, start=1):
                images = images.to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(images)
                    losses = loss_fn(outputs, targets, model.strides)
                    scaled_loss = losses["loss"] / accum_steps
                if amp_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = step_idx % accum_steps == 0 or step_idx == len(train_loader)
                if should_step and amp_enabled:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                elif should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for key in totals:
                    totals[key] += float(losses[key].detach().cpu())
                pbar.set_postfix(loss=float(losses["loss"].detach().cpu()))
            n = max(len(train_loader), 1)
            row = {k: totals[k] / n for k in totals}
            row["epoch"] = epoch
            row["val_precision"] = ""
            row["val_recall"] = ""
            row["val_map50"] = ""
            row["val_avg_predictions"] = ""
            row["lr"] = optimizer.param_groups[0]["lr"]
            metrics = {}
            should_eval = eval_fn is not None and eval_every and (epoch % eval_every == 0 or epoch == epochs)
            if should_eval:
                metrics = eval_fn()
                row["val_precision"] = metrics.get("precision", 0.0)
                row["val_recall"] = metrics.get("recall", 0.0)
                row["val_map50"] = metrics.get("mAP50_approx", 0.0)
                row["val_avg_predictions"] = metrics.get("avg_predictions_per_image", 0.0)
            writer.writerow(row)
            f.flush()
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, metrics)
            score = float(metrics.get(monitor, metrics.get("mAP50_approx", -row["loss"]))) if metrics else -row["loss"]
            if score > best_score + min_delta:
                best_score = score
                stale_evals = 0
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, metrics)
            elif should_eval:
                stale_evals += 1
            plot_training_log(log_path, run_dir / "curves" / "loss_curve.png")
            can_stop_for_patience = target_score is None or best_score >= target_score
            if patience and should_eval and epoch >= min_epochs and can_stop_for_patience and stale_evals >= patience:
                print(f"early stopping at epoch {epoch}: no {monitor} improvement for {stale_evals} evaluations")
                break
            if scheduler is not None:
                scheduler.step()
    return log_path
