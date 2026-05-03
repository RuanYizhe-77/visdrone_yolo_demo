# Deployment Notes

This project is a learning-oriented demo, not a production detector. The deployment path is still useful to understand:

- PyTorch inference is easiest for research iteration.
- ONNX export makes the model portable to ONNX Runtime, TensorRT, OpenVINO, or other runtimes.
- TensorRT can fuse layers and run FP16/INT8 kernels for lower latency on NVIDIA GPUs.
- FP16 usually improves throughput with small accuracy impact. INT8 requires calibration data and more careful validation.
- Real video systems normally use asynchronous decode, batched inference, tracking, rendering, and streaming queues.
- Report latency with batch size, image size, hardware, precision, preprocessing time, model time, NMS time, and end-to-end FPS.

Typical production pipeline:

```text
camera/video stream
-> frame decode
-> resize/normalize
-> GPU inference
-> NMS/postprocess
-> tracker/update IDs
-> event logic/storage
-> visualization/API/stream output
```

