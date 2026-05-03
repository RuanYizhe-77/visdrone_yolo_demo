import time

import torch


@torch.no_grad()
def benchmark_model(model, img_size=640, batch_size=1, device="cuda", warmup=10, iters=50):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / iters
    fps = batch_size * 1000.0 / latency_ms
    return {"batch_size": batch_size, "img_size": img_size, "latency_ms": latency_ms, "fps": fps}

