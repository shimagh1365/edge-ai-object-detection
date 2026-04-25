from ultralytics import YOLO

print("=" * 55)
print("COCO Persons Evaluation — Val set only")
print("=" * 55)

model = YOLO('yolov8n.pt')

# Validate on val2017 only — much smaller download
results = model.val(
    data='coco8.yaml',   # coco8 = tiny 8-image subset, instant
    imgsz=640,
    classes=[0],         # persons only
    verbose=True,
    plots=False,
)

print()
print("=" * 55)
print(f"  mAP@0.5:       {results.box.map50:.4f}")
print(f"  mAP@0.5:0.95:  {results.box.map:.4f}")
print(f"  Precision:     {results.box.mp:.4f}")
print(f"  Recall:        {results.box.mr:.4f}")
print("=" * 55)
