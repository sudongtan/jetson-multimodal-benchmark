"""Export a YOLO model to a TensorRT engine at a given precision.

Run inside the ultralytics/ultralytics:latest-jetson-jetpack6 container:
    python3 export.py --model yolov8n.pt --precision fp32
    python3 export.py --model yolov8n.pt --precision fp16
    python3 export.py --model yolov8n.pt --precision int8 --data coco8.yaml
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export(model_path: str, precision: str, data: str, imgsz: int, workspace: float):
    model = YOLO(model_path)
    common = dict(format="engine", imgsz=imgsz, workspace=workspace, device=0)

    if precision == "fp32":
        engine_path = model.export(**common)
    elif precision == "fp16":
        try:
            engine_path = model.export(quantize=16, **common)
        except TypeError:
            engine_path = model.export(half=True, **common)
    elif precision == "int8":
        try:
            engine_path = model.export(quantize=8, data=data, **common)
        except TypeError:
            engine_path = model.export(int8=True, data=data, **common)
    else:
        raise ValueError(f"unknown precision: {precision}")

    stem = Path(model_path).stem
    dest = Path(model_path).with_name(f"{stem}_{precision}.engine")
    shutil.move(engine_path, dest)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    ap.add_argument("--data", default="coco128.yaml", help="calibration/val dataset for int8")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workspace", type=float, default=4)
    args = ap.parse_args()
    export(args.model, args.precision, args.data, args.imgsz, args.workspace)
