import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visdrone.det_dataset import VisDroneDETDataset
from src.visualization.draw import draw_boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--split", default="train")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--output-dir", default="assets/annotations")
    args = p.parse_args()

    ds = VisDroneDETDataset(args.data_root, args.split, args.img_size, max_samples=args.num_samples)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(args.num_samples, len(ds))):
        _, target = ds[i]
        image = cv2.imread(target["path"])
        h0, w0 = image.shape[:2]
        sx, sy = w0 / args.img_size, h0 / args.img_size
        boxes = target["boxes"].clone()
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        vis = draw_boxes(image, boxes.numpy(), target["labels"].numpy())
        out_path = out_dir / Path(target["path"]).name
        cv2.imwrite(str(out_path), vis)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
