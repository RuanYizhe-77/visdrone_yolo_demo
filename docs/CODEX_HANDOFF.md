# Codex Handoff

## Goals And Constraints

- Build a GitHub-ready VisDrone detection/tracking demo.
- Keep custom training low-level PyTorch only; do not use Ultralytics/MMDetection/Detectron2 for custom training.
- Use Ultralytics only to load/run the pretrained VisDrone YOLO reference.
- Preserve the old 50-epoch baseline for comparison.
- Final presentation should compare: pretrained VisDrone YOLO, newer custom P2 model, old 50-epoch model.
- Operate only inside this repository.
- Do not delete datasets, checkpoints, or generated comparison assets.

## Current Repo Structure

```text
configs/
  current_yolo_newerbest_oldest.json
  current_yolo_newerbest_oldest_stable_tracking.json
docs/
  CODEX_HANDOFF.md
  beginner_guide.md
  deployment_notes.md
scripts/
  train_det.py
  eval_det.py
  infer_images.py
  track_vid.py
  run_final_comparison_suite.py
  sweep_eval.py
src/
  common/
  engine/
  modeling/
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
  presentation_tracking_comparison_uav0000305_highthr/
```

Generated assets, datasets, checkpoints, and videos are ignored by Git.

## Files Created Or Modified

Important created files:

- `configs/current_yolo_newerbest_oldest.json`
- `configs/current_yolo_newerbest_oldest_stable_tracking.json`
- `scripts/run_final_comparison_suite.py`
- `scripts/sweep_eval.py`
- `src/tracking/bytetrack_lite.py`
- `src/engine/yolo_infer.py`
- `assets/current_yolo_newerbest_oldest/`
- `assets/presentation_tracking_comparison/`
- `assets/presentation_tracking_comparison_uav0000305/`
- `assets/presentation_tracking_comparison_uav0000305_highthr/`

Important modified files:

- `README.md`
- `report.md`
- `.gitignore`
- `scripts/train_det.py`
- `scripts/eval_det.py`
- `scripts/infer_images.py`
- `scripts/track_vid.py`
- `scripts/benchmark.py`
- `scripts/export_onnx.py`
- `scripts/run_comparison_suite.py`
- `scripts/run_final_comparison_suite.py`
- `src/engine/train.py`
- `src/engine/evaluate.py`
- `src/modeling/*`
- `src/visualization/draw.py`

## Implemented

- FCOS-style detector with optional P2 stride-4 feature level.
- Softplus regression head and stride-normalized box distances.
- Aligned GIoU loss to avoid large pairwise IoU memory use.
- Classwise, agnostic, and classwise-then-agnostic NMS.
- ByteTrack-style tracker and stable demo tracker.
- Final comparison suite:
  - image grids
  - tracking videos
  - score CSV/JSON
  - benchmark support
  - Markdown report
- High-contrast comparison video labels.
- Stricter ByteTrack presentation render for `uav0000305_00000_v`.
- Presentation README and concise final report.

## Tested

- Syntax compile:
  - `scripts/run_final_comparison_suite.py`
  - `scripts/run_comparison_suite.py`
  - `scripts/train_det.py`
  - `src/engine/train.py`
  - `src/engine/evaluate.py`
  - `src/visualization/draw.py`
- Generated full image/score comparison:
  - `assets/current_yolo_newerbest_oldest/`
  - 548 image grids
- Generated tracking videos:
  - `assets/presentation_tracking_comparison/tracking/uav0000268_05773_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4`
  - `assets/presentation_tracking_comparison_uav0000268_late/tracking/uav0000268_05773_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4`
  - `assets/presentation_tracking_comparison_uav0000305/tracking/uav0000305_00000_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4`
  - `assets/presentation_tracking_comparison_uav0000305_highthr/tracking/uav0000305_00000_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4`
- Verified generated MP4s open with OpenCV:
  - `uav0000268_05773_v`: 300 frames, 20 FPS, 3840x720
  - `uav0000268_05773_v` later segment: 300 frames, 20 FPS, 3840x720
  - `uav0000305_00000_v`: 184 frames, 20 FPS, 3840x720
  - `uav0000305_00000_v` high threshold: 184 frames, 20 FPS, 3840x720
- Compared `uav0000305_00000_v` track counts:
  - lower-threshold newer model: 11915 rows, 1590 unique IDs
  - high-threshold newer model: 4350 rows, 309 unique IDs
- Stopped the parallel training search PIDs:
  - `3069730`
  - `3074315`
  - `3076632`
  - `3076798`
  - `3076868`
  - `3080276`

## Final Scores

From `assets/current_yolo_newerbest_oldest/scores/metrics_summary.csv`:

| model | precision | recall | mAP50 approx | boxes/image |
| --- | ---: | ---: | ---: | ---: |
| pretrained_yolo | 0.6971 | 0.4957 | 0.2960 | 50.30 |
| newerbest_p2_small | 0.4041 | 0.4796 | 0.2211 | 83.94 |
| oldest_50epoch | 0.2823 | 0.2504 | 0.0593 | 62.74 |

## Important Commands Run

Final full comparison:

```bash
/home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/run_final_comparison_suite.py --data-root . --custom-specs configs/current_yolo_newerbest_oldest.json --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt --output-dir assets/current_yolo_newerbest_oldest --device cuda:0 --img-size 640 --conf 0.35 --eval-conf 0.25 --batch-size 32 --num-workers 4 --max-images 0 --max-frames 300 --max-detections 120 --agnostic-nms-iou 0.6 --track-low-conf 0.08 --new-track-conf 0.45 --yolo-first --skip-benchmarks
```

Presentation tracking videos:

```bash
/home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/run_final_comparison_suite.py --data-root . --custom-specs configs/current_yolo_newerbest_oldest.json --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt --output-dir assets/presentation_tracking_comparison --device cuda:0 --img-size 640 --conf 0.35 --eval-conf 0.25 --batch-size 32 --num-workers 4 --video-split val --sequence uav0000268_05773_v --max-frames 300 --max-detections 80 --agnostic-nms-iou 0.6 --track-low-conf 0.12 --new-track-conf 0.45 --track-iou 0.25 --video-max-width 3840 --yolo-first --skip-images --skip-scores --skip-benchmarks --no-progress
```

```bash
/home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/run_final_comparison_suite.py --data-root . --custom-specs configs/current_yolo_newerbest_oldest.json --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt --output-dir assets/presentation_tracking_comparison_uav0000305 --device cuda:0 --img-size 640 --conf 0.35 --eval-conf 0.25 --batch-size 32 --num-workers 4 --video-split val --sequence uav0000305_00000_v --max-frames 184 --max-detections 80 --agnostic-nms-iou 0.6 --track-low-conf 0.12 --new-track-conf 0.45 --track-iou 0.25 --video-max-width 3840 --yolo-first --skip-images --skip-scores --skip-benchmarks --no-progress
```

```bash
/home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/run_final_comparison_suite.py --data-root . --custom-specs configs/current_yolo_newerbest_oldest.json --yolo-weights models/pretrained/yolov11s_visdrone_risef_best.pt --output-dir assets/presentation_tracking_comparison_uav0000305_highthr --device cuda:0 --img-size 640 --conf 0.50 --eval-conf 0.25 --batch-size 32 --num-workers 4 --video-split val --sequence uav0000305_00000_v --max-frames 184 --max-detections 60 --agnostic-nms-iou 0.55 --track-low-conf 0.25 --new-track-conf 0.70 --track-iou 0.25 --video-max-width 3840 --yolo-first --skip-images --skip-scores --skip-benchmarks --no-progress
```

Stopped search:

```bash
kill -INT 3069730 3074315 3076632 3076798 3076868 3080276
```

## Status

- Training search is stopped.
- Final presentation assets are generated, including the stricter high-threshold ByteTrack video.
- README and report are updated for GitHub-style presentation.
- Remaining GPU jobs shown by `nvidia-smi` belong to other users/processes and were not touched.

## Remaining TODOs

- Optional: add small compressed GIF/PNG thumbnails for GitHub if assets should be visible directly on GitHub.
- Optional: run official VisDrone or COCO-style evaluation if official metrics are needed.
- Optional: commit only code/docs/configs; keep datasets/checkpoints/videos out of Git unless explicitly desired.
