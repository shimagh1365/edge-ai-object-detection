from ultralytics import YOLO
import torch
import time
import numpy as np

def benchmark(model, img_size=640, runs=100):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Running on: {device}")
    
    # Normalized input 0-1 (fixes the warning)
    dummy = torch.rand(1, 3, img_size, img_size)
    
    # Warmup
    for _ in range(10):
        model(dummy, verbose=False)
    
    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model(dummy, verbose=False)
        times.append((time.perf_counter() - start) * 1000)
    
    params = sum(p.numel() for p in model.model.parameters())
    size_mb = sum(p.numel() * p.element_size()
                  for p in model.model.parameters()) / 1e6
    
    return {
        'latency_mean_ms': np.mean(times),
        'latency_p95_ms': np.percentile(times, 95),
        'params_M': params / 1e6,
        'size_mb': size_mb,
        'energy_mj': 100 * (np.mean(times) / 1000) * 1000
    }

print("=" * 50)
print("YOLOv8-Nano Baseline")
print("=" * 50)
model = YOLO('yolov8n.pt')
r = benchmark(model)
for k, v in r.items():
    print(f"  {k}: {v:.3f}")
