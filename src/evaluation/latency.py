"""Forward-pass latency shared by every model."""
import time

import numpy as np
import torch


@torch.no_grad()
def measure_latency(model, dataset, device, count, warmup=10):
    """Median forward time in ms for batch size 1 on already preprocessed images."""
    model.eval().to(device)
    times = []
    for index in range(min(count + warmup, len(dataset))):
        image = dataset[index][0].unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if index >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)) if times else None
