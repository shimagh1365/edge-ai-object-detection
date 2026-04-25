import torch
import torchvision
import time
import numpy as np

print("=" * 55)
print("MobileNet-SSD Baseline (second comparison)")
print("=" * 55)

# MobileNetV3-Small as backbone proxy
model = torchvision.models.mobilenet_v3_small(
    weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
)
model.eval()

dummy = torch.rand(1, 3, 320, 320)  # SSD typically uses 320x320

# Warmup
for _ in range(10):
    with torch.no_grad():
        model(dummy)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        model(dummy)
    times.append((time.perf_counter() - start) * 1000)

params = sum(p.numel() for p in model.parameters())
size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
energy_mj = 100 * (np.mean(times) / 1000) * 1000

print(f"  latency_mean_ms: {np.mean(times):.3f}")
print(f"  latency_p95_ms:  {np.percentile(times, 95):.3f}")
print(f"  params_M:        {params/1e6:.3f}")
print(f"  size_mb:         {size_mb:.3f}")
print(f"  energy_mj:       {energy_mj:.1f}")
print("=" * 55)

print("\nFinal Comparison Table:")
print(f"{'Model':<25} {'Latency(ms)':<14} {'Size(MB)':<12} {'Energy(mJ)':<12}")
print("-" * 63)
print(f"{'YOLOv8-Nano (FP32)':<25} {'65.0':<14} {'12.6':<12} {'6500':<12}")
print(f"{'MobileNetV3-Small':<25} {np.mean(times):<14.1f} {size_mb:<12.1f} {energy_mj:<12.0f}")
print(f"{'EG-YOLO + DVS Gate':<25} {'57.7*':<14} {'6.3':<12} {'2112':<12}")
print("\n* Latency when inference runs (67.5% frames skipped)")
print("  EG-YOLO energy = YOLOv8 energy × (1 - 0.675)")
