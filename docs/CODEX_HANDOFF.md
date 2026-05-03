# Codex Handoff

## 1. Original Goals And Constraints

Goals:

- Rebuild the VisDrone demo into a GitHub-ready project.
- Cover VisDrone-DET image object detection training/evaluation/inference.
- Cover VisDrone-VID video detection and tracking.
- Generate visual GitHub assets: annotation samples, prediction images, tracking videos, curves, and comparison outputs.
- Compare the custom trained detector with an optional off-the-shelf pretrained YOLO baseline.
- Add deployment-oriented pieces: ONNX export, speed benchmark, TensorRT/GPU/FP16 discussion.
- Add beginner-friendly technical docs for detection, metrics, tracking, video CV, and deployment.

Constraints:

- Do not use Ultralytics YOLO, MMDetection, Detectron2, or other high-level detection training frameworks for the custom model.
- PyTorch, torchvision, OpenCV, NumPy, tqdm, matplotlib are allowed.
- YOLO is allowed only as an optional pretrained comparison baseline.
- Keep code educational and readable.
- Work only inside this repository.
- Do not delete datasets, checkpoints, or important existing outputs.
- Use `cat-sam` for GPU training.

## 2. Current Repo Structure

Important current structure:

```text
configs/
  det_fcos_lite.yaml
  smoke.yaml

docs/
  beginner_guide.md
  deployment_notes.md
  CODEX_HANDOFF.md

scripts/
  benchmark.py
  eval_det.py
  export_onnx.py
  infer_images.py
  smoke_test.py
  track_vid.py
  train_det.py
  visualize_det.py
  yolo_compare.py

src/
  common/
  engine/
  modeling/
  tracking/
  visdrone/
  visualization/

datasets/VisDrone/
  VisDrone-DET/
  VisDrone-VID/

outputs/runs/
  smoke_rebuild/
  det_fcos_lite_50e/
```

Legacy files from the previous CenterNet-style demo still exist at repo root and under `src/`.

## 3. Files Created Or Modified

Created:

- `configs/det_fcos_lite.yaml`
- `configs/smoke.yaml`
- `docs/beginner_guide.md`
- `docs/deployment_notes.md`
- `docs/CODEX_HANDOFF.md`
- `scripts/benchmark.py`
- `scripts/eval_det.py`
- `scripts/export_onnx.py`
- `scripts/infer_images.py`
- `scripts/smoke_test.py`
- `scripts/track_vid.py`
- `scripts/train_det.py`
- `scripts/visualize_det.py`
- `scripts/yolo_compare.py`
- `src/common/boxes.py`
- `src/common/checkpoint.py`
- `src/common/metrics.py`
- `src/common/seed.py`
- `src/engine/benchmark.py`
- `src/engine/evaluate.py`
- `src/engine/infer.py`
- `src/engine/train.py`
- `src/modeling/assigner.py`
- `src/modeling/backbone.py`
- `src/modeling/decode.py`
- `src/modeling/detector.py`
- `src/modeling/fpn.py`
- `src/modeling/losses.py`
- `src/tracking/iou_tracker.py`
- `src/visdrone/classes.py`
- `src/visdrone/det_dataset.py`
- `src/visdrone/discovery.py`
- `src/visdrone/vid_dataset.py`
- `src/visualization/draw.py`
- `src/visualization/plots.py`

Modified:

- `README.md`
- `requirements.txt`
- `report.md` was already modified from the previous run.

Generated during tests/training:

- `outputs/runs/smoke_rebuild/checkpoints/smoke.pt`
- `outputs/runs/smoke_rebuild/predictions/...`
- `outputs/runs/det_fcos_lite_50e/checkpoints/last.pt`
- `outputs/runs/det_fcos_lite_50e/checkpoints/best.pt`
- `outputs/runs/det_fcos_lite_50e/logs/train_log.csv`
- `outputs/runs/det_fcos_lite_50e/curves/loss_curve.png`
- Python `__pycache__/` files in new module folders.

## 4. What Has Been Implemented

- Official VisDrone-DET discovery and dataset loading.
- Official VisDrone-VID sequence discovery and annotation parsing.
- Custom FCOS-lite anchor-free detector:
  - small ResNet-like backbone
  - FPN
  - classification, bbox regression, centerness heads
  - point-based FCOS-style assignment
  - focal loss, GIoU loss, centerness loss
  - NMS decoding
- Training engine with checkpoints, CSV logs, and loss curve.
- Evaluation engine with simplified AP50/precision/recall.
- Image inference and visualization.
- DET annotation visualization script.
- VID tracking script with simple class-aware IoU tracking.
- Inference speed benchmark script.
- ONNX export script.
- Optional YOLO comparison script that gracefully exits if `ultralytics` is unavailable.
- Beginner guide and deployment notes.
- GitHub-ready README draft with reproduction commands.

## 5. What Has Been Tested

Tested successfully in `cat-sam`:

- Python syntax check for new scripts/modules.
- Dataset discovery for DET and VID.
- DET dataset load.
- One FCOS-lite forward pass.
- Loss computation and backward pass.
- Optimizer step.
- Smoke checkpoint saving.
- One prediction visualization from smoke model.
- 50-epoch training has started and is actively running.

Not yet tested after full training:

- Full validation on `outputs/runs/det_fcos_lite_50e/checkpoints/last.pt`.
- Final image prediction gallery.
- VID tracking video output.
- Benchmark output.
- ONNX export for the new FCOS-lite model.
- YOLO comparison output.

## 6. Exact Commands Already Run

Inspection:

```bash
pwd
git status --short
find . -maxdepth 3 -type d | sort
find . -maxdepth 2 -type f | sort
find datasets/VisDrone -maxdepth 5 -type d | sort
find datasets/VisDrone/VisDrone-VID -maxdepth 5 -type d | sort | sed -n '1,200p'
find datasets/VisDrone/VisDrone-DET -maxdepth 3 -type d | sort
find datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-train/images -type f | wc -l
find datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/images -type f | wc -l
find datasets/VisDrone/VisDrone-VID/VisDrone2019-VID-train/sequences -mindepth 1 -maxdepth 1 -type d | wc -l
find datasets/VisDrone/VisDrone-VID/VisDrone2019-VID-val/sequences -mindepth 1 -maxdepth 1 -type d | wc -l
sed -n '1,5p' datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-train/annotations/0000047_02500_d_0000093.txt
sed -n '1,8p' datasets/VisDrone/VisDrone-VID/VisDrone2019-VID-val/annotations/uav0000086_00000_v.txt
```

Directory creation:

```bash
mkdir -p configs src/visdrone src/modeling src/engine src/tracking src/visualization src/utils scripts docs assets/annotations assets/predictions assets/comparisons assets/tracking assets/curves outputs/runs checkpoints/det_fcos_lite checkpoints/legacy
```

Syntax check:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python -m py_compile scripts/smoke_test.py scripts/train_det.py scripts/eval_det.py scripts/infer_images.py scripts/track_vid.py scripts/benchmark.py scripts/export_onnx.py scripts/yolo_compare.py src/visdrone/classes.py src/visdrone/discovery.py src/visdrone/det_dataset.py src/visdrone/vid_dataset.py src/modeling/backbone.py src/modeling/fpn.py src/modeling/detector.py src/modeling/assigner.py src/modeling/losses.py src/modeling/decode.py src/engine/train.py src/engine/evaluate.py src/engine/infer.py src/engine/benchmark.py src/tracking/iou_tracker.py src/visualization/draw.py src/visualization/plots.py src/utils/boxes.py src/utils/checkpoint.py src/utils/metrics.py src/utils/seed.py
```

Smoke test attempts:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/smoke_test.py --data-root . --device cuda --img-size 320 --output-dir outputs/runs/smoke_rebuild
```

This was run three times:

- First failed: script could not import `src` from `scripts/`.
- Second failed: legacy `src/utils.py` conflicted with new `src/utils/` package.
- Third passed after adding script path bootstraps and moving new helpers to `src/common/`.

Training command currently running:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/train_det.py --data-root . --epochs 50 --batch-size 8 --img-size 640 --device cuda --num-workers 4 --run-dir outputs/runs/det_fcos_lite_50e --eval-every 0
```

Log/status commands run during training:

```bash
tail -n 5 outputs/runs/det_fcos_lite_50e/logs/train_log.csv
tail -n 8 outputs/runs/det_fcos_lite_50e/logs/train_log.csv
tail -n 12 outputs/runs/det_fcos_lite_50e/logs/train_log.csv
find outputs/runs/det_fcos_lite_50e -maxdepth 3 -type f | sort | sed -n '1,80p'
git status --short
find configs docs scripts src/visdrone src/modeling src/engine src/tracking src/visualization src/common -maxdepth 2 -type f | sort
```

## 7. Current Training/Evaluation Status

Current long training process:

- Session ID in Codex tool: `98590`.
- Command: 50 epochs, batch size 8, image size 640, CUDA, `cat-sam`.
- Current observed live status: epoch `20/50`, around `25%` complete.
- Latest completed CSV epoch: epoch `19`.
- Latest completed epoch loss: `0.9923857073259295`.
- Loss trend:

```text
epoch 1:  1.6359
epoch 5:  1.2290
epoch 10: 1.1159
epoch 15: 1.0434
epoch 19: 0.9924
```

Evaluation status:

- Full validation for the new FCOS-lite model has not been run yet.
- Training script is using `--eval-every 0`, but currently any value other than `1` means final validation runs after training completes.
- No final AP50/precision/recall numbers exist yet for the new model.

## 8. Remaining TODOs By Priority

Priority 1:

- Let 50-epoch training finish.
- Verify final `outputs/runs/det_fcos_lite_50e/checkpoints/last.pt` and `best.pt`.
- Run full validation:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/eval_det.py --data-root . --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt --device cuda --output-json outputs/runs/det_fcos_lite_50e/metrics/val_metrics.json
```

Priority 2:

- Generate annotation samples:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/visualize_det.py --data-root . --split train --num-samples 12 --output-dir assets/annotations
```

- Generate prediction images:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/infer_images.py --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt --source datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/images --device cuda --output-dir assets/predictions
```

Priority 3:

- Run VID tracking:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/track_vid.py --data-root . --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt --split val --device cuda --max-frames 300 --output-dir assets/tracking
```

- Run benchmark:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/benchmark.py --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt --img-size 640 --batch-size 1 --device cuda --output-json outputs/runs/det_fcos_lite_50e/metrics/benchmark.json
```

- Export ONNX:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/export_onnx.py --weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt --img-size 640 --output outputs/runs/det_fcos_lite_50e/export/model.onnx
```

Priority 4:

- Run optional YOLO comparison if `ultralytics` is installed:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python scripts/yolo_compare.py --ours-weights outputs/runs/det_fcos_lite_50e/checkpoints/last.pt --yolo-weights yolov8n.pt --source datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/images --device cuda --output-dir assets/comparisons
```

- Update `report.md` with final metrics, qualitative notes, and output paths.
- Consider adding a small gallery section to README after visual assets are generated.

## 9. Important Warnings, Assumptions, And Dataset Paths

Warnings:

- Do not delete datasets, checkpoints, or previous outputs.
- The active training job is still running. Do not start a second full training job until it finishes or is intentionally stopped.
- `src/utils.py` from the legacy demo conflicts with a `src/utils/` package name. New shared helpers were moved to `src/common/` for this reason.
- `scripts/train_det.py --eval-every 0` currently means no per-epoch validation callback; final validation is run only after training completes.
- `requirements.txt` no longer includes `pandas`; new plotting code uses Python `csv`.
- YOLO comparison is optional and depends on `ultralytics`; it is not a core dependency.
- The new evaluator is simplified AP50/mAP50, not official VisDrone or COCO mAP.
- The tracker is simple IoU tracking, not full SORT or ByteTrack.

Assumptions:

- `cat-sam` is the correct CUDA environment.
- The full training command is allowed to continue.
- Existing `yolov8n.pt` and `yolo26n.pt` are local baseline weights; no download is required.
- The project should preserve the old CenterNet-style demo while adding the new GitHub-ready pipeline.

Dataset paths:

```text
DET train images:
datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-train/images

DET train annotations:
datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-train/annotations

DET val images:
datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/images

DET val annotations:
datasets/VisDrone/VisDrone-DET/VisDrone2019-DET-val/annotations

VID train sequences:
datasets/VisDrone/VisDrone-VID/VisDrone2019-VID-train/sequences

VID train annotations:
datasets/VisDrone/VisDrone-VID/VisDrone2019-VID-train/annotations

VID val sequences:
datasets/VisDrone/VisDrone-VID/VisDrone2019-VID-val/sequences

VID val annotations:
datasets/VisDrone/VisDrone-VID/VisDrone2019-VID-val/annotations
```

