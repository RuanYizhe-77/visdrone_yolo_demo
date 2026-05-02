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
