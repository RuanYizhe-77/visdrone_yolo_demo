# VisDrone Detection and Tracking Demo

Low-level PyTorch object detection and video tracking demo on VisDrone. The project trains an educational FCOS-style anchor-free detector from scratch, compares it with an older custom baseline and a VisDrone-pretrained YOLO reference, then visualizes detection and tracking behavior on images and videos.

This is a learning and presentation repo, not a production detector and not an official VisDrone benchmark.

## Demo Summary

Final validation comparison, using the simplified repo evaluator on VisDrone-DET val:

| model | role | tracker in demo | precision | recall | mAP50 approx | boxes/image |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pretrained YOLO | external VisDrone reference | ByteTrack-style | 0.6971 | 0.4957 | 0.2960 | 50.30 |
| newerbest P2 custom | improved low-level model | ByteTrack-style | 0.4041 | 0.4796 | 0.2211 | 83.94 |
| old 50-epoch custom | baseline | simple IoU | 0.2823 | 0.2504 | 0.0593 | 62.74 |

The intended visual ranking is:

```text
pretrained VisDrone YOLO > newer P2 custom > old 50-epoch custom
```

## Presentation Assets

Generated comparison assets are local outputs and may be too large for normal Git commits.

Main image and score comparison:

- Report: [assets/current_yolo_newerbest_oldest/comparison_report.md](assets/current_yolo_newerbest_oldest/comparison_report.md)
- Scores: [assets/current_yolo_newerbest_oldest/scores/metrics_summary.csv](assets/current_yolo_newerbest_oldest/scores/metrics_summary.csv)
- Full validation image grids: [assets/current_yolo_newerbest_oldest/images](assets/current_yolo_newerbest_oldest/images)

Clear tracking-strategy videos:

- Easier sequence, lowest density: [uav0000268_05773_v comparison](assets/presentation_tracking_comparison/tracking/uav0000268_05773_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4)
- Easier sequence, later segment: [uav0000268_05773_v later comparison](assets/presentation_tracking_comparison_uav0000268_late/tracking/uav0000268_05773_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4)
- Second easier sequence: [uav0000305_00000_v comparison](assets/presentation_tracking_comparison_uav0000305/tracking/uav0000305_00000_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4)
- Recommended stricter ByteTrack version: [uav0000305_00000_v high-threshold comparison](assets/presentation_tracking_comparison_uav0000305_highthr/tracking/uav0000305_00000_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4)

These videos use high-contrast panel labels:

```text
pretrained_yolo_bytetrack | newerbest_p2_small_bytetrack | oldest_50epoch_iou
```

For `uav0000305_00000_v`, the stricter render raises the detector/tracker thresholds and reduces the newer model's created IDs from 1590 to 309 over the same 184-frame clip.

## Model Difference

| part | old 50-epoch custom | newerbest P2 custom | pretrained YOLO |
| --- | --- | --- | --- |
| training code | low-level PyTorch | low-level PyTorch | external pretrained checkpoint |
| detector family | FCOS-style anchor-free | FCOS-style anchor-free | YOLO11-style |
| small-object feature | starts at stride 8 | adds P2 stride-4 feature | stronger YOLO pyramid |
| box regression | raw distances + ReLU | stride-normalized distances + Softplus | YOLO native regression |
| postprocess | classwise + agnostic NMS in comparison | classwise + agnostic NMS | YOLO postprocess |
| video tracker | simple IoU | ByteTrack-style | ByteTrack-style |

The newer custom model is better mainly because P2 gives a finer feature grid for tiny drone-view objects. The old detector misses more objects and produces less stable boxes, so its IoU tracking has frequent ID changes.

## What Is Implemented

- VisDrone-DET and VisDrone-VID dataset loaders.
- Educational FCOS-style detector:
  - small ResNet-like backbone
  - FPN with optional P2 stride-4 level
  - classification, box regression, and centerness heads
  - focal loss, aligned GIoU loss, and centerness loss
  - FCOS point assignment
  - classwise, agnostic, and classwise-then-agnostic NMS
- Training, evaluation, inference, ONNX export, and speed benchmark scripts.
- Trackers:
  - simple class-aware IoU tracker
  - ByteTrack-style two-stage association
  - stable ByteTrack-style mode for cleaner demos
- Full image, score, speed, and video comparison scripts.

High-level training frameworks such as Ultralytics, MMDetection, and Detectron2 are not used for custom training. Ultralytics is used only to load the pretrained YOLO reference for inference and comparison.

## Repository Map

```text
configs/
  current_yolo_newerbest_oldest.json
  current_yolo_newerbest_oldest_stable_tracking.json
docs/
  beginner_guide.md
  deployment_notes.md
  CODEX_HANDOFF.md
scripts/
  train_det.py
  eval_det.py
  infer_images.py
  track_vid.py
  run_final_comparison_suite.py
  sweep_eval.py
src/
  modeling/
  engine/
  tracking/
  visdrone/
  visualization/
models/pretrained/
  yolov11s_visdrone_risef_best.pt
outputs/runs/
  det_fcos_lite_50e/
  fcos_p2_small_softplus_norm_180e/
assets/
  current_yolo_newerbest_oldest/
  presentation_tracking_comparison/
  presentation_tracking_comparison_uav0000268_late/
  presentation_tracking_comparison_uav0000305/
```

Datasets, checkpoints, and generated videos are intentionally ignored by Git by default.

## Dataset Layout

Place VisDrone under `datasets/VisDrone/`:

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
    VisDrone2019-VID-val/
      sequences/
      annotations/
```

## Setup

On the Nessie server used for this run:

```bash
source activate cat-sam
pip install -r requirements.txt
```

For GPU commands in this repo, use the environment Python if needed:

```bash
/home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python
```

## Reproduce Key Outputs

Generate the current image, score, and tracking comparison:

```bash
python scripts/run_final_comparison_suite.py \
  --data-root . \
  --custom-specs configs/current_yolo_newerbest_oldest.json \
  --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt \
  --output-dir assets/current_yolo_newerbest_oldest \
  --device cuda:0 \
  --img-size 640 \
  --conf 0.35 \
  --eval-conf 0.25 \
  --batch-size 32 \
  --max-images 0 \
  --max-frames 300 \
  --max-detections 120 \
  --agnostic-nms-iou 0.6 \
  --track-low-conf 0.08 \
  --new-track-conf 0.45 \
  --yolo-first \
  --skip-benchmarks
```

Generate the clearest tracking-strategy videos:

```bash
python scripts/run_final_comparison_suite.py \
  --data-root . \
  --custom-specs configs/current_yolo_newerbest_oldest.json \
  --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt \
  --output-dir assets/presentation_tracking_comparison \
  --device cuda:0 \
  --video-split val \
  --sequence uav0000268_05773_v \
  --conf 0.35 \
  --track-low-conf 0.12 \
  --new-track-conf 0.45 \
  --track-iou 0.25 \
  --max-frames 300 \
  --max-detections 80 \
  --video-max-width 3840 \
  --yolo-first \
  --skip-images \
  --skip-scores \
  --skip-benchmarks
```

```bash
python scripts/run_final_comparison_suite.py \
  --data-root . \
  --custom-specs configs/current_yolo_newerbest_oldest.json \
  --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt \
  --output-dir assets/presentation_tracking_comparison_uav0000268_late \
  --device cuda:0 \
  --video-split val \
  --sequence uav0000268_05773_v \
  --start-frame 450 \
  --conf 0.35 \
  --track-low-conf 0.12 \
  --new-track-conf 0.45 \
  --track-iou 0.25 \
  --max-frames 300 \
  --max-detections 80 \
  --video-max-width 3840 \
  --yolo-first \
  --skip-images \
  --skip-scores \
  --skip-benchmarks
```

```bash
python scripts/run_final_comparison_suite.py \
  --data-root . \
  --custom-specs configs/current_yolo_newerbest_oldest.json \
  --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt \
  --output-dir assets/presentation_tracking_comparison_uav0000305 \
  --device cuda:0 \
  --video-split val \
  --sequence uav0000305_00000_v \
  --conf 0.35 \
  --track-low-conf 0.12 \
  --new-track-conf 0.45 \
  --track-iou 0.25 \
  --max-frames 184 \
  --max-detections 80 \
  --video-max-width 3840 \
  --yolo-first \
  --skip-images \
  --skip-scores \
  --skip-benchmarks
```

Generate the stricter ByteTrack version of the second easier sequence:

```bash
python scripts/run_final_comparison_suite.py \
  --data-root . \
  --custom-specs configs/current_yolo_newerbest_oldest.json \
  --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt \
  --output-dir assets/presentation_tracking_comparison_uav0000305_highthr \
  --device cuda:0 \
  --video-split val \
  --sequence uav0000305_00000_v \
  --conf 0.50 \
  --track-low-conf 0.25 \
  --new-track-conf 0.70 \
  --track-iou 0.25 \
  --max-frames 184 \
  --max-detections 60 \
  --agnostic-nms-iou 0.55 \
  --video-max-width 3840 \
  --yolo-first \
  --skip-images \
  --skip-scores \
  --skip-benchmarks
```

Evaluate a custom checkpoint:

```bash
python scripts/eval_det.py \
  --data-root . \
  --weights outputs/runs/fcos_p2_small_softplus_norm_180e/checkpoints/best.pt \
  --device cuda:0 \
  --width 64 \
  --fpn-channels 160 \
  --head-convs 3 \
  --use-p2 \
  --reg-activation softplus \
  --regress-normalized \
  --nms-mode classwise_then_agnostic \
  --agnostic-nms-iou 0.6
```

## Training

Old baseline:

```bash
python scripts/train_det.py \
  --data-root . \
  --epochs 50 \
  --batch-size 8 \
  --img-size 640 \
  --device cuda:0 \
  --run-dir outputs/runs/det_fcos_lite_50e
```

Newer P2 custom model:

```bash
python scripts/train_det.py \
  --data-root . \
  --epochs 180 \
  --batch-size 32 \
  --img-size 640 \
  --device cuda:0 \
  --run-dir outputs/runs/fcos_p2_small_softplus_norm_180e \
  --eval-every 5 \
  --eval-conf 0.25 \
  --eval-nms-mode classwise_then_agnostic \
  --eval-agnostic-nms-iou 0.6 \
  --scheduler cosine \
  --lr 0.0006 \
  --width 64 \
  --fpn-channels 160 \
  --head-convs 3 \
  --use-p2 \
  --reg-activation softplus \
  --regress-normalized
```

## Notes

- `mAP50_approx` is a simplified AP50-style metric implemented in this repo. It is useful for internal comparison, but it is not the official VisDrone or COCO evaluator.
- ByteTrack-style tracking improves ID continuity only when detections are reasonably stable. It cannot fully fix missed objects, wrong boxes, or duplicate detections from a weak detector.
- The old 50-epoch baseline is intentionally kept for contrast.
- The final training search was stopped after generating the presentation assets.
