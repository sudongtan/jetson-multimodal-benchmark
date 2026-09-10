"""Capture one frame from a webcam, run YOLO (TensorRT engine) on it, crop the
highest-confidence detection, and hand off crop + frame + detection metadata for
the VLM stage (describe_crop.py) to pick up.

Run inside the ultralytics/ultralytics:latest-jetson-jetpack6 container, with the
webcam device passed through (see run_pipeline.sh for the full docker run):
    python3 capture_and_detect.py --engine yolov8n_fp16.engine
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Padding around the detected box before cropping, as a fraction of the box's own
# width/height -- gives the VLM a bit of surrounding context rather than a tight,
# possibly-ambiguous crop.
CROP_PADDING = 0.15


def capture_frame(video_device: int, warmup_frames: int):
    cap = cv2.VideoCapture(video_device)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video device {video_device} -- is the webcam connected and passed through to the container (--device)?")
    try:
        # Most USB webcams' first few frames are garbage/misexposed while auto
        # exposure/white-balance settle -- discard a handful before the real capture.
        for _ in range(warmup_frames):
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read a frame from video device {video_device}")
        return frame
    finally:
        cap.release()


def crop_with_padding(frame, box_xyxy, padding: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(w, int(x2 + bw * padding))
    y2 = min(h, int(y2 + bh * padding))
    return frame[y1:y2, x1:x2]


def run(engine_path: str, video_device: int, warmup_frames: int, warmup_inferences: int, imgsz: int, out_dir: Path):
    t_capture_start = time.perf_counter()
    frame = capture_frame(video_device, warmup_frames)
    capture_ms = (time.perf_counter() - t_capture_start) * 1000

    t_load_start = time.perf_counter()
    model = YOLO(engine_path, task="detect")
    load_ms = (time.perf_counter() - t_load_start) * 1000

    # The first predict() call after loading an engine pays TensorRT/CUDA's
    # one-time kernel-compilation and memory-allocation cost -- without this,
    # inference_ms measures "cold start," not steady-state per-frame latency
    # (confirmed: an un-warmed-up run here measured 6.1s vs. vision-benchmarks/
    # benchmark.py's own warmed-up ~8ms for the same engine). Same pattern that
    # script already uses, just applied to the real captured frame instead of a
    # dummy image -- what matters for warmup is exercising the kernels once, not
    # the pixel content.
    cold_inference_ms = None
    if warmup_inferences > 0:
        t_cold_start = time.perf_counter()
        model.predict(frame, imgsz=imgsz, verbose=False, device=0)
        cold_inference_ms = (time.perf_counter() - t_cold_start) * 1000
        for _ in range(warmup_inferences - 1):
            model.predict(frame, imgsz=imgsz, verbose=False, device=0)

    t_infer_start = time.perf_counter()
    results = model.predict(frame, imgsz=imgsz, verbose=False, device=0)[0]
    inference_ms = (time.perf_counter() - t_infer_start) * 1000

    detection = None
    crop = frame
    if len(results.boxes) > 0:
        # Highest-confidence detection -- boxes aren't guaranteed sorted, so take
        # the argmax explicitly rather than assuming index 0 is the best one.
        best_idx = int(np.argmax(results.boxes.conf.cpu().numpy()))
        box = results.boxes[best_idx]
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        class_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        detection = {"class_name": class_name, "confidence": confidence, "box_xyxy": xyxy}
        crop = crop_with_padding(frame, xyxy, CROP_PADDING)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "frame.jpg"), frame)
    cv2.imwrite(str(out_dir / "crop.jpg"), crop)

    # frame.jpg/crop.jpg above are the stable paths describe_crop.py reads -- they
    # get overwritten every run, so nothing survives across runs unless archived
    # separately. Kept under out_dir (.cache/) so it inherits the existing
    # gitignore rule -- webcam images are as privacy-sensitive archived as they are
    # fresh, so this should stay out of git the same way.
    ts = time.strftime("%Y%m%dT%H%M%S")
    archive_dir = out_dir / "frames"
    archive_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(archive_dir / f"{ts}_frame.jpg"), frame)
    cv2.imwrite(str(archive_dir / f"{ts}_crop.jpg"), crop)

    metadata = {
        "detection": detection,
        "capture_ms": capture_ms,
        "vision_load_ms": load_ms,
        "vision_cold_inference_ms": cold_inference_ms,  # first call, pre-warmup -- None if warmup_inferences=0
        "vision_inference_ms": inference_ms,  # steady-state, post-warmup
        "engine": engine_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "detection.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"wrote {out_dir}/frame.jpg, {out_dir}/crop.jpg, {out_dir}/detection.json")
    print(f"archived {archive_dir}/{ts}_frame.jpg, {archive_dir}/{ts}_crop.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to a YOLO .engine file, e.g. yolov8n_fp16.engine")
    ap.add_argument("--video-device", type=int, default=0, help="cv2.VideoCapture index, matches /dev/video<N>")
    ap.add_argument("--warmup-frames", type=int, default=5, help="frames discarded before the captured one, for exposure/white-balance to settle")
    ap.add_argument("--warmup-inferences", type=int, default=3, help="throwaway predict() calls before the timed one, so vision_inference_ms reflects steady-state latency, not TensorRT/CUDA's one-time kernel-compilation cost")
    ap.add_argument("--imgsz", type=int, default=640, help="must match what the engine was exported/built with")
    ap.add_argument("--out-dir", default="/workspace/combined-pipeline/.cache")
    args = ap.parse_args()

    run(args.engine, args.video_device, args.warmup_frames, args.warmup_inferences, args.imgsz, Path(args.out_dir))
