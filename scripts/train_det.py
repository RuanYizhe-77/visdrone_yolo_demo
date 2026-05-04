import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.evaluate import evaluate_detector
from src.engine.train import train_detector
from src.modeling.detector import FCOSLiteDetector
from src.common.checkpoint import load_model_weights
from src.common.seed import seed_everything
from src.visdrone.det_dataset import VisDroneDETDataset, detection_collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--run-dir", default="outputs/runs/det_fcos_lite_50e")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--val-max-samples", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-conf", type=float, default=0.25)
    p.add_argument("--eval-iou", type=float, default=0.5)
    p.add_argument("--eval-max-detections", type=int, default=120)
    p.add_argument("--eval-nms-mode", choices=["classwise", "agnostic", "classwise_then_agnostic"], default="classwise")
    p.add_argument("--eval-agnostic-nms-iou", type=float, default=0.7)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-epochs", type=int, default=0)
    p.add_argument("--monitor", choices=["mAP50_approx", "precision", "recall"], default="mAP50_approx")
    p.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation improvement needed to reset patience.")
    p.add_argument("--target-score", type=float, default=None, help="Do not early-stop before this monitored score is reached.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars for background training logs.")
    p.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision to reduce memory use.")
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps for larger effective batches.")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--fpn-channels", type=int, default=160)
    p.add_argument("--head-convs", type=int, default=3)
    p.add_argument("--use-p2", action="store_true", help="Add a stride-4 FPN level for small VisDrone objects.")
    p.add_argument("--reg-activation", choices=["relu", "softplus"], default="relu")
    p.add_argument("--regress-normalized", action="store_true", help="Predict box distances in stride-normalized units.")
    p.add_argument("--center-radius", type=float, default=1.5)
    p.add_argument("--resume", default=None, help="Resume model and optimizer state from a checkpoint.")
    p.add_argument("--resume-model-only", action="store_true", help="Load model weights but start a fresh optimizer/scheduler.")
    args = p.parse_args()

    seed_everything(42)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    train_ds = VisDroneDETDataset(args.data_root, "train", args.img_size, args.max_samples, augment=True)
    val_ds = VisDroneDETDataset(args.data_root, "val", args.img_size, args.val_max_samples, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate)
    model = FCOSLiteDetector(
        width=args.width,
        fpn_channels=args.fpn_channels,
        head_convs=args.head_convs,
        use_p2=args.use_p2,
        reg_activation=args.reg_activation,
        regress_normalized=args.regress_normalized,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    start_epoch = 1
    if args.resume:
        ckpt = load_model_weights(model, args.resume, device)
        if not args.resume_model_only and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        resume_mode = "model only" if args.resume_model_only else "model and optimizer"
        print(f"Resuming {resume_mode} from {args.resume} at epoch {start_epoch}/{args.epochs}")
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - start_epoch + 1, 1), eta_min=args.min_lr
        )

    def eval_fn():
        return evaluate_detector(
            model,
            val_loader,
            device=device,
            conf=args.eval_conf,
            iou=args.eval_iou,
            output_json=Path(args.run_dir) / "metrics" / "latest_val.json",
            max_detections=args.eval_max_detections,
            nms_mode=args.eval_nms_mode,
            agnostic_nms_thresh=args.eval_agnostic_nms_iou,
            progress=not args.no_progress,
        )

    callback = eval_fn if args.eval_every > 0 else None
    train_detector(
        model,
        train_loader,
        val_loader,
        optimizer,
        args.epochs,
        device,
        args.run_dir,
        eval_fn=callback,
        start_epoch=start_epoch,
        resume_log=bool(args.resume),
        eval_every=args.eval_every,
        patience=args.patience,
        min_epochs=args.min_epochs,
        monitor=args.monitor,
        scheduler=scheduler,
        center_radius=args.center_radius,
        min_delta=args.min_delta,
        target_score=args.target_score,
        progress=not args.no_progress,
        amp=args.amp,
        accum_steps=args.accum_steps,
    )
    if args.eval_every <= 0:
        metrics = evaluate_detector(
            model,
            val_loader,
            device=device,
            conf=args.eval_conf,
            iou=args.eval_iou,
            output_json=Path(args.run_dir) / "metrics" / "val_metrics.json",
            max_detections=args.eval_max_detections,
            nms_mode=args.eval_nms_mode,
            agnostic_nms_thresh=args.eval_agnostic_nms_iou,
            progress=not args.no_progress,
        )
        print(metrics)


if __name__ == "__main__":
    main()
