"""Forward-pass latency shared by every model."""
import platform
import time

import numpy as np
import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def benchmark_latency(model, dataset, device, count=200, warmup=10, operation=None):
    """Batch-1 inference, excluding decode, transforms and host/device transfer.

    Cycles over short datasets so warmup never consumes all measurements. operation may
    include scoring; callers must record that scope separately. Synchronizes CUDA AND MPS.
    """
    if count < 1 or warmup < 0 or len(dataset) == 0:
        raise ValueError("Need a nonempty dataset, count >= 1 and warmup >= 0")
    device = torch.device(device)
    model.eval().to(device)
    operation = operation or model
    times = []
    for index in range(count + warmup):
        image = dataset[index % len(dataset)][0].unsqueeze(0).to(device)
        synchronize(device)
        start = time.perf_counter()
        operation(image)
        synchronize(device)
        if index >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    return dict(mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times, .95)), count=count, warmup=warmup, batch_size=1,
                unique_images=min(len(dataset), count + warmup), samples_ms=times,
                device=str(device), hardware=hardware, platform=platform.platform(),
                torch=str(torch.__version__), threads=torch.get_num_threads(),
                dtype=str(next(model.parameters(), torch.empty(0)).dtype),
                input_shape=list(image.shape), includes="inference only; excludes I/O, preprocessing and transfer")


def measure_latency(model, dataset, device, count, warmup=10):
    """Backward-compatible median used by both training pipelines."""
    if count == 0 or len(dataset) == 0:
        return None  # Existing training smoke runs explicitly disable latency with count=0.
    return benchmark_latency(model, dataset, device, count, warmup)["median_ms"]
