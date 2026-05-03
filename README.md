# VisDrone Detection, Tracking, and Deployment Demo

This is a GitHub-ready, learning-oriented computer vision project for VisDrone. It is not a production detector and it does not claim state-of-the-art results.

The project demonstrates:

- object detection training on VisDrone-DET with a custom PyTorch detector
- video detection and tracking on VisDrone-VID
- annotation and prediction visualizations
- optional side-by-side comparison with a pretrained YOLO baseline
- ONNX export and inference speed benchmarking
- beginner-friendly notes on detection, tracking, metrics, and deployment

## What Is Implemented From Scratch

The main detector is an educational FCOS-lite style anchor-free model:

- ResNet-like CNN backbone
- FPN feature pyramid for small objects
- classification, box regression, and centerness heads
- FCOS-style point assignment
- focal loss, GIoU box loss, and centerness loss
- NMS decoding
- simplified AP50 / precision / recall evaluation

High-level training frameworks such as Ultralytics YOLO, MMDetection, and Detectron2 are not used for the custom training pipeline.

## Dataset Layout

The code discovers the official VisDrone folders under `--data-root`:

```text
datasets/VisDrone/
  VisDrone-DET/
    VisDrone2019-DET-train/
      images/
      annotations/
    VisDrone2019-DET-val/
      images/
      annotations/
  VisDrone-VID/
    VisDrone2019-VID-train/
      sequences/
      annotations/
    VisDrone2019-VID-val/
      sequences/
      annotations/
```

VisDrone-DET annotation format:

```text
bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
```

VisDrone-VID annotation format:

```text
frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
```

Categories `1-10` are mapped to class IDs `0-9`. Categories outside that range are ignored.

## Setup

Use your existing environment. On this server, GPU training should use:

```bash
source activate cat-sam
```

Core dependencies:

```bash
pip install -r requirements.txt
```

YOLO comparison is optional. If `ultralytics` is not installed, `scripts/yolo_compare.py` exits with a clear message.

## Smoke Test

Run this before real training:

```bash
python scripts/smoke_test.py --data-root . --device cuda --img-size 320 --output-dir outputs/runs/smoke_rebuild
```

It checks dataset discovery, one batch load, one forward pass, loss/backward, checkpoint saving, and one prediction visualization.

## Dataset Visualization

```bash
python scripts/visualize_det.py --data-root . --split train --num-samples 12 --output-dir assets/annotations
```

## Training

Full 50-epoch training:

```bash
python scripts/train_det.py \
  --data-root . \
  --epochs 50 \
  --batch-size 8 \
  --img-size 640 \
  --device cuda \
  --num-workers 4 \
  --run-dir outputs/runs/det_fcos_lite_50e
```

Outputs:

```text
outputs/runs/det_fcos_lite_50e/
  checkpoints/last.pt
  checkpoints/best.pt
  logs/train_log.csv
  curves/loss_curve.png
  metrics/val_metrics.json
```

## Evaluation

```bash
python scripts/eval_det.py \
  --data-root . \
  --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt \
  --device cuda \
  --output-json outputs/runs/det_fcos_lite_50e/metrics/val_metrics.json
```

The evaluator reports precision, recall, and simplified AP50/mAP50. This is not the official VisDrone evaluator.

## Image Inference

```bash
python scripts/infer_images.py \
  --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt \
  --source datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/images \
  --device cuda \
  --output-dir assets/predictions
```

## Video Tracking

```bash
python scripts/track_vid.py \
  --data-root . \
  --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt \
  --split val \
  --device cuda \
  --max-frames 300 \
  --output-dir assets/tracking
```

Tracking uses a simple class-aware IoU tracker. It is useful for learning tracking-by-detection, but it is not full ByteTrack.

## YOLO Baseline Comparison

This script is optional and isolated from the custom training pipeline:

```bash
python scripts/yolo_compare.py \
  --ours-weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt \
  --yolo-weights yolov8n.pt \
  --source datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/images \
  --device cuda \
  --output-dir assets/comparisons
```

## Benchmark

```bash
python scripts/benchmark.py \
  --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt \
  --img-size 640 \
  --batch-size 1 \
  --device cuda \
  --output-json outputs/runs/det_fcos_lite_50e/metrics/benchmark.json
```

## ONNX Export

```bash
python scripts/export_onnx.py \
  --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt \
  --img-size 640 \
  --output outputs/runs/det_fcos_lite_50e/export/model.onnx
```

## Documentation

- [Beginner Guide](docs/beginner_guide.md): object detection, VisDrone formats, mAP, video tracking, IDs, and deployment basics.
- [Deployment Notes](docs/deployment_notes.md): ONNX, TensorRT, FP16/INT8, FPS/latency, and production pipeline structure.

## Limitations

- This is a portfolio and learning demo.
- The detector is custom and compact; it should not be compared to production-grade YOLO/RT-DETR systems as an equal model.
- The mAP implementation is simplified and intended for educational feedback.
- The IoU tracker can switch IDs during occlusions and crowded motion.
- YOLO is used only as an off-the-shelf comparison baseline, not as the custom training framework.
