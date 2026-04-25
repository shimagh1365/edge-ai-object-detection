# EG-YOLO: Event-Guided Efficient Object Detection for Edge AI

## imec Gen-100 Edge AI Chip — Algorithm Design (Part B)

### Problem
Deploy real-time human detection on a humanoid robot using RGB + DVS sensors within 100mW power budget on imec Gen-100 edge hardware.

### Key Innovation: DVS Temporal Gating
Standard YOLO processes every frame at full cost. EG-YOLO uses the DVS event camera as a temporal gate — only triggering RGB inference when meaningful motion is detected.

### Results
| Model | Latency | Size | Energy | Saving |
|---|---|---|---|---|
| YOLOv8-Nano (baseline) | 65.0ms | 12.6MB | 6,500mJ | — |
| MobileNetV3-Small | 49.5ms | 10.2MB | 4,951mJ | -24% |
| EG-YOLO + DVS Gate | 57.7ms* | 6.3MB | 2,112mJ | -67.5% |

*Latency when inference runs. 67.5% of frames skipped by DVS gate.

### Files
- `baseline.py` — YOLOv8-Nano benchmark
- `dvs_gate.py` — DVS temporal gating simulation
- `quantize.py` — FP16 quantization
- `mobilenet_baseline.py` — MobileNetV3 comparison

### Install
pip install ultralytics torch torchvision opencv-python numpy matplotlib

### Run
python baseline.py
python dvs_gate.py
python mobilenet_baseline.py
