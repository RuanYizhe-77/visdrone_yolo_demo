# Interview Report

## Project Summary

I built a compact anchor-free object detection demo for VisDrone using PyTorch from scratch. The system trains a small CNN detector with center heatmaps, width/height regression, and offset regression, then supports validation, image inference, video inference, simple IoU tracking, and ONNX export.

This is an interview-oriented local demo, not a production detection system.

## Pipeline

```text
VisDrone images
  -> target generation
  -> anchor-free detector
  -> training
  -> validation
  -> image/video inference
  -> IoU tracking
  -> ONNX export
```

## Skills Demonstrated

- object detection pipeline design
- heatmap target generation
- bounding-box width/height regression
- center offset regression
- loss construction for dense prediction
- prediction decoding and NMS
- validation metrics
- video inference
- simple online tracking
- FPS and latency measurement
- deployment awareness through ONNX export

## Limitations

- The detector is intentionally simple and not SOTA.
- The heatmap target uses center cells rather than a full optimized Gaussian formulation.
- The mAP50 calculation is simplified and is not a COCO evaluator.
- The IoU tracker is basic and can lose IDs in occlusions and crowded scenes.
- The project does not include full production deployment.

## Deployment Considerations

For deployment, I would export the model to ONNX and then evaluate TensorRT or OpenVINO depending on the target hardware. I would benchmark FP32, FP16, and INT8 where supported, tune input resolution against small-object recall, and measure FPS, latency, and memory under realistic video conditions. For a production video system, I would also consider batching, asynchronous frame capture and inference, backpressure, tracking state management, and a model size appropriate for the target device.

## 60-Second Interview Explanation

I am a PhD candidate in computer vision and medical image analysis. My main research is segmentation and foundation model adaptation. I built this demo to strengthen object detection, video computer vision, and deployment-oriented engineering.

In this project, I implemented a small anchor-free detector for VisDrone in PyTorch without using Ultralytics, YOLO frameworks, MMDetection, Detectron2, or pretrained detection libraries. The model predicts object center heatmaps, width and height, and center offsets. I generate training targets from VisDrone annotations, train the model with heatmap and regression losses, decode predictions into boxes, apply NMS, and evaluate precision, recall, and approximate AP50.

I also added image inference, video inference with FPS reporting, a simple IoU-based tracker for assigning IDs across frames, and ONNX export. The model is intentionally not SOTA, but it demonstrates that I understand the full object detection workflow from dataset parsing and target generation through inference, tracking, and deployment preparation.

## Demo Pipeline Run Log

Date: 2026-05-03

### Commands Actually Run

```bash
python -c "from src.dataset import discover_visdrone; print('train:', discover_visdrone('.', 'train')); print('val:', discover_visdrone('.', 'val'))"
python train.py --data-root . --epochs 1 --batch-size 4 --img-size 512 --device cuda --max-samples 200
python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"
nvidia-smi
source activate cat-sam && python -c "from src.dataset import discover_visdrone; print('train:', discover_visdrone('.', 'train')); print('val:', discover_visdrone('.', 'val'))"
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python -c "import sys, torch; print('python', sys.version); print('executable', sys.executable); print('torch', torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"
```

### Step Results

- Dataset discovery: passed.
- Training smoke test: stopped before completion because CUDA was unavailable and the run fell back to CPU.
- Validation: not run because no checkpoint was produced.
- Image inference: not run because no checkpoint was produced.
- Video tracking: not run because no checkpoint was produced and no video path was provided.
- ONNX export: not run because no checkpoint was produced.
- Full-dataset training/validation/testing: not run because CUDA is unavailable in the current runtime.

### Environment Diagnosis

Base environment:

```text
Python: 3.7.3
PyTorch: 1.7.0
torch.version.cuda: 11.0
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

`cat-sam` environment:

```text
Python: 3.9.21
Executable: /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python
PyTorch: 1.13.1+cu117
torch.version.cuda: 11.7
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
Warning: Can't initialize NVML
```

System GPU check:

```text
nvidia-smi failed because it could not communicate with the NVIDIA driver.
```

### Key Metrics

No training or validation metrics were produced because the CUDA smoke training did not complete.

### Output Paths

- `outputs/train_log.csv`: created but empty because training did not complete.
- `checkpoints/last.pt`: not created.
- `checkpoints/best.pt`: not created.
- `outputs/val_metrics.json`: not created.
- `outputs/images/`: not created by inference.
- `outputs/model.onnx`: not created.

### Known Issues And Limitations From This Run

- The selected Python environment has a CUDA-enabled PyTorch build, but the runtime cannot access an NVIDIA driver/device.
- `nvidia-smi` fails independently of Python, so this is a driver/GPU visibility issue rather than a project code issue.
- The shell activation sets `CONDA_DEFAULT_ENV=cat-sam`, but the direct environment Python path was used for the final diagnosis to avoid pyenv shim ambiguity.
- Training should be retried only after `nvidia-smi` works and `torch.cuda.is_available()` is true inside `cat-sam`.

## Successful Host-GPU Demo Run

Date: 2026-05-03

The default Codex sandbox could not see `/dev/nvidia*`, but running commands outside the sandbox with the explicit `cat-sam` Python executable exposed the GPUs correctly:

```text
Executable: /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python
PyTorch: 1.13.1+cu117
CUDA available: true
Visible GPUs: 8 Tesla V100-SXM2-32GB devices
```

### Smoke Pipeline

Commands run:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python train.py --data-root . --epochs 1 --batch-size 4 --img-size 512 --device cuda --max-samples 200
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python val.py --data-root . --weights checkpoints/last.pt --device cuda --max-samples 200
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_02999_d_0000005.jpg --device cuda
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_03499_d_0000006.jpg --device cuda
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_03999_d_0000007.jpg --device cuda
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python export_onnx.py --weights checkpoints/last.pt
```

Smoke results:

- Training: passed, 1 epoch on 200 samples.
- Training loss: `1.2541`.
- Validation: passed on 200 validation samples.
- Smoke precision/recall/mAP50 approximation: `0.0 / 0.0 / 0.0`.
- Image inference: passed on 3 validation images.
- ONNX export: passed.

Smoke output paths:

- `checkpoints/last.pt`
- `checkpoints/best.pt`
- `outputs/train_log.csv`
- `outputs/val_metrics.json`
- `outputs/images/0000001_02999_d_0000005.jpg`
- `outputs/images/0000001_03499_d_0000006.jpg`
- `outputs/images/0000001_03999_d_0000007.jpg`
- `outputs/model.onnx`

### Full-Dataset Pipeline

Dataset counts:

- Train images: `6471`
- Validation images: `548`
- Test images were available under `datasets/VisDrone/images/test`.
- No local video files were found, so video tracking was skipped until a video path is provided.

Commands run:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python train.py --data-root . --epochs 5 --batch-size 8 --img-size 512 --device cuda
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python val.py --data-root . --weights checkpoints/last.pt --device cuda
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_02999_d_0000005.jpg --device cuda --output-dir outputs/full_val_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_03499_d_0000006.jpg --device cuda --output-dir outputs/full_val_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_03999_d_0000007.jpg --device cuda --output-dir outputs/full_val_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/test/0000006_00159_d_0000001.jpg --device cuda --output-dir outputs/test_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/test/0000006_00611_d_0000002.jpg --device cuda --output-dir outputs/test_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/test/0000006_01111_d_0000003.jpg --device cuda --output-dir outputs/test_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python export_onnx.py --weights checkpoints/last.pt
```

Full training results:

```text
epoch 1: loss 0.8196, heatmap 0.0055, wh 3.5807, offset 0.4560
epoch 2: loss 0.6925, heatmap 0.0020, wh 2.7378, offset 0.4166
epoch 3: loss 0.6523, heatmap 0.0019, wh 2.4842, offset 0.4020
epoch 4: loss 0.6268, heatmap 0.0018, wh 2.3253, offset 0.3924
epoch 5: loss 0.6079, heatmap 0.0018, wh 2.2152, offset 0.3846
```

Full validation results:

```text
precision: 0.8228464419475655
recall: 0.05668360896823964
mAP50_approx: 0.014072179363029461
```

Per-class AP50 summary:

```text
pedestrian: 0.0062
people: 0.0000
bicycle: 0.0000
car: 0.1345
van: 0.0000
truck: 0.0000
tricycle: 0.0000
awning-tricycle: 0.0000
bus: 0.0000
motor: 0.0000
```

Full output paths:

- `checkpoints/last.pt` (`8.4M`)
- `checkpoints/best.pt` (`8.4M`)
- `outputs/train_log.csv`
- `outputs/val_metrics.json`
- `outputs/full_val_images/0000001_02999_d_0000005.jpg`
- `outputs/full_val_images/0000001_03499_d_0000006.jpg`
- `outputs/full_val_images/0000001_03999_d_0000007.jpg`
- `outputs/test_images/0000006_00159_d_0000001.jpg`
- `outputs/test_images/0000006_00611_d_0000002.jpg`
- `outputs/test_images/0000006_01111_d_0000003.jpg`
- `outputs/model.onnx` (`2.8M`)

Known limitations from the full run:

- The model is intentionally tiny and trained from scratch for only 5 epochs.
- The detector mostly learned high-confidence car-like detections; recall remains low.
- `mAP50_approx` is a simplified evaluator, not official COCO/VisDrone mAP.
- Test image inference is qualitative because the local test split does not have labels in this demo flow.
- Video tracking was not run because no local video file was found. Run `track_video.py` with a video path to exercise that part.

### 50-Epoch Full-Dataset Pipeline

This run repeated the same full-dataset workflow with a longer 50-epoch training schedule in the `cat-sam` CUDA environment. It overwrote `checkpoints/last.pt`, `checkpoints/best.pt`, `outputs/train_log.csv`, `outputs/val_metrics.json`, and `outputs/model.onnx`.

Commands run:

```bash
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python train.py --data-root . --epochs 50 --batch-size 8 --img-size 512 --device cuda
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python val.py --data-root . --weights checkpoints/last.pt --device cuda
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_02999_d_0000005.jpg --device cuda --output-dir outputs/50epoch_val_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_03499_d_0000006.jpg --device cuda --output-dir outputs/50epoch_val_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/val/0000001_03999_d_0000007.jpg --device cuda --output-dir outputs/50epoch_val_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/test/0000006_00159_d_0000001.jpg --device cuda --output-dir outputs/50epoch_test_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/test/0000006_00611_d_0000002.jpg --device cuda --output-dir outputs/50epoch_test_images
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python infer_image.py --weights checkpoints/last.pt --source datasets/VisDrone/images/test/0000006_01111_d_0000003.jpg --device cuda --output-dir outputs/50epoch_test_images
find . -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.mov' -o -name '*.mkv' \)
source activate cat-sam && /home/mil/ruan/.pyenv/versions/anaconda3-2019.07/envs/cat-sam/bin/python export_onnx.py --weights checkpoints/last.pt
```

Step status:

- Training: passed.
- Validation: passed.
- Validation image inference: passed for 3 images.
- Test image inference: passed for 3 images.
- Video tracking: skipped because no local video file was found under the repository.
- ONNX export: passed.

Selected training loss trend:

```text
epoch 1:  loss 0.8196, heatmap 0.0055, wh 3.5807, offset 0.4560
epoch 10: loss 0.5541, heatmap 0.0017, wh 1.9218, offset 0.3603
epoch 20: loss 0.4883, heatmap 0.0016, wh 1.6387, offset 0.3229
epoch 30: loss 0.4475, heatmap 0.0016, wh 1.5079, offset 0.2951
epoch 40: loss 0.4197, heatmap 0.0015, wh 1.4219, offset 0.2760
epoch 50: loss 0.3989, heatmap 0.0015, wh 1.3598, offset 0.2614
```

Validation results:

```text
precision: 0.6998268897864974
recall: 0.15645398488093087
mAP50_approx: 0.042730564599670354
```

Per-class AP50 summary:

```text
pedestrian: 0.0509
people: 0.0343
bicycle: 0.0000
car: 0.2944
van: 0.0071
truck: 0.0000
tricycle: 0.0000
awning-tricycle: 0.0000
bus: 0.0000
motor: 0.0406
```

Output paths:

- `checkpoints/last.pt` (`8.4M`)
- `checkpoints/best.pt` (`8.4M`)
- `outputs/train_log.csv`
- `outputs/val_metrics.json`
- `outputs/50epoch_val_images/0000001_02999_d_0000005.jpg`
- `outputs/50epoch_val_images/0000001_03499_d_0000006.jpg`
- `outputs/50epoch_val_images/0000001_03999_d_0000007.jpg`
- `outputs/50epoch_test_images/0000006_00159_d_0000001.jpg`
- `outputs/50epoch_test_images/0000006_00611_d_0000002.jpg`
- `outputs/50epoch_test_images/0000006_01111_d_0000003.jpg`
- `outputs/model.onnx` (`2.8M`)

Known limitations from the 50-epoch run:

- The detector is still a compact from-scratch demo model, not a production VisDrone detector.
- Recall improved over the 5-epoch run, but overall mAP remains low.
- Cars dominate the learned detections; smaller and rarer classes remain weak.
- `mAP50_approx` is simplified and should not be reported as official COCO or VisDrone mAP.
- Test image inference is qualitative only in this local flow.
- Video tracking was not executed because no video file was present locally.
