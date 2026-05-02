# VisDrone Tiny Center Detector

This is an interview-oriented local demo, not a production detection system.

The project implements a compact anchor-free object detector for VisDrone in plain PyTorch. It is similar in spirit to CenterNet: an input image passes through a small CNN backbone, then three heads predict class center heatmaps, bounding-box width/height, and center offsets. Predictions are decoded into boxes, filtered with NMS, evaluated with simplified metrics, visualized on images/videos, tracked with a simple IoU tracker, and exportable to ONNX.

No Ultralytics, YOLO framework, MMDetection, Detectron2, or pretrained detection library is used.

## Why This Demo Exists

The goal is not state-of-the-art accuracy. The goal is to show the main mechanics of object detection training and deployment-oriented computer vision engineering in readable code:

- heatmap target generation
- bounding-box regression
- train and validation loops
- NMS-based decoding
- image and video inference
- basic online tracking
- FPS and latency reporting
- ONNX export awareness

VisDrone is useful because it contains dense small objects from drone viewpoints, which makes it a practical dataset for detection and video CV demonstrations.

## Setup

Create or use your existing Python environment, then install the small dependency set:

```bash
pip install -r requirements.txt
```

Do not install Ultralytics for this project.

## Dataset Structure

The dataset discovery utility searches under `--data-root` for official VisDrone detection folders:

```text
VisDrone2019-DET-train/
  images/
  annotations/
VisDrone2019-DET-val/
  images/
  annotations/
```

It also supports the converted local layout found in this repository:

```text
datasets/VisDrone/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Official VisDrone annotations are expected as:

```text
bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
```

Categories `1` through `10` are mapped to class indices `0` through `9`. Categories `0` and `11` are ignored.

## Training

```bash
python train.py --data-root . --epochs 5 --batch-size 8 --img-size 512 --device cuda
```

Outputs:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `outputs/train_log.csv`

Use `--max-samples` for a quick smoke run:

```bash
python train.py --data-root . --epochs 1 --batch-size 2 --img-size 256 --device cpu --max-samples 8
```

## Validation

```bash
python val.py --data-root . --weights checkpoints/best.pt --device cuda
```

Outputs:

- `outputs/val_metrics.json`

Metrics include precision, recall, and an approximate mAP50. This is not a COCO-style evaluator.

## Image Inference

```bash
python infer_image.py --weights checkpoints/best.pt --source path/to/image.jpg --device cuda
```

Folder input is also supported. Visualizations are saved to `outputs/images`.

## Video Inference

```bash
python infer_video.py --weights checkpoints/best.pt --source path/to/video.mp4 --device cuda
```

The script saves `outputs/video_detected.mp4` and prints frame count, average inference time, and FPS.

## Tracking

```bash
python track_video.py --weights checkpoints/best.pt --source path/to/video.mp4 --device cuda
```

Tracking uses `SimpleIoUTracker`, a small online IoU-based tracker. It is ByteTrack-lite style in spirit but is not the full ByteTrack algorithm.

## ONNX Export

```bash
python export_onnx.py --weights checkpoints/best.pt
```

The exported model is saved to `outputs/model.onnx`. If export fails because the environment lacks ONNX support, the script prints a helpful message.

## Implemented From Scratch

- compact CNN detector
- heatmap, width/height, and offset heads
- VisDrone target generation
- detection loss
- decoding and NMS
- simplified validation metrics
- image/video inference loops
- IoU tracker
- ONNX export wrapper

## Not Implemented

- pretrained detection backbones
- anchor assignment
- YOLO-specific training or decoding
- COCO mAP evaluator
- multi-scale training
- advanced augmentation
- TensorRT/OpenVINO deployment
- production async video pipeline

## Limitations

This model is intentionally small and simple. It uses a stride-4 feature map and center-cell targets, so accuracy on dense small objects will be limited. The AP50 calculation is a simplified confidence-sorted approximation. The tracker matches boxes by IoU and class only, so identity switches are expected in crowded scenes.
