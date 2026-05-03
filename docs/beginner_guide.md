# Beginner Guide: Detection, Tracking, and Deployment

## Image Object Detection

Object detection predicts both **what** is in an image and **where** it is. The output is usually a set of bounding boxes, class labels, and confidence scores.

## Detector Architectures

Modern one-stage detectors predict dense candidates over feature maps. The backbone extracts visual features, the neck such as an FPN combines low-level and high-level features, and heads predict classes and boxes.

This project uses an FCOS-style anchor-free detector. Instead of predefined anchors, each feature-map point can represent an object if it falls near the object center.

## VisDrone Format

VisDrone-DET annotation lines are:

```text
bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
```

Valid categories are 1 through 10. Category IDs are mapped to class indices 0 through 9.

VisDrone-VID annotation lines include frame and identity:

```text
frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
```

## IoU, Precision, Recall, AP, mAP

IoU measures box overlap. A prediction is usually considered correct if it has the right class and IoU is above a threshold such as 0.5.

Precision asks: among predicted boxes, how many were correct?

Recall asks: among ground-truth objects, how many did the detector find?

AP summarizes a precision-recall curve for one class. mAP averages AP across classes.

## Video Detection vs Image Detection

Video detection can run an image detector frame by frame, but video adds temporal consistency, motion blur, occlusion, and identity tracking.

## Tracking-by-Detection

Tracking-by-detection first detects objects in every frame, then matches detections across frames to maintain track IDs.

Simple IoU tracking:

```text
new frame detections
-> compare each active track box to new detection boxes
-> match high-IoU boxes with the same class
-> keep matched IDs
-> create new IDs for unmatched detections
-> remove old unmatched tracks after max_age frames
```

SORT adds a Kalman filter for motion prediction. ByteTrack also uses low-confidence detections to recover tracks, which reduces identity loss in crowded scenes.

## GPU Inference and Deployment

GPU inference is fast because convolution and matrix operations are massively parallel. Batching improves throughput but can increase latency. ONNX is a portable graph format. TensorRT can optimize ONNX models for NVIDIA GPUs. FP16 and INT8 reduce memory bandwidth and often improve FPS.

For deployment, always report:

- GPU model
- input resolution
- batch size
- precision
- preprocessing time
- inference time
- postprocessing/NMS time
- end-to-end FPS

