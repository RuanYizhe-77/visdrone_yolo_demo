import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VisDroneDetectionDataset, collate_fn
from src.decode import decode_predictions
from src.metrics import evaluate_detections
from src.model import TinyCenterNet
from src.utils import get_device, load_checkpoint, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--weights", default="checkpoints/best.pt")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    dataset = VisDroneDetectionDataset(args.data_root, "val", args.img_size, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model = TinyCenterNet(num_classes=10).to(device)
    load_checkpoint(model, args.weights, device)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for imgs, _, raw_targets, _ in tqdm(loader, desc="validating"):
            out = model(imgs.to(device))
            det = decode_predictions(out, args.conf, args.iou, model.stride, topk=300)[0]
            preds.append({k: v.cpu() for k, v in det.items()})
            targets.append(raw_targets[0])
    metrics = evaluate_detections(preds, targets, num_classes=10, iou_thresh=0.5)
    save_json(metrics, "outputs/val_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
