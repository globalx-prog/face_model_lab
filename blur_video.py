from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO


def oval_blur(frame, box, padding_factor: float = 0.5):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    face_w = x2 - x1
    face_h = y2 - y1
    if face_w <= 0 or face_h <= 0:
        return frame

    pad_x = int(face_w * padding_factor)
    pad_y = int(face_h * padding_factor)
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(width, x2 + pad_x)
    ey2 = min(height, y2 + pad_y)
    roi = frame[ey1:ey2, ex1:ex2]
    if roi.size == 0:
        return frame

    blurred_roi = cv2.GaussianBlur(roi, (99, 99), 0)
    mask = np.zeros_like(roi, dtype=np.uint8)
    center = (x1 + face_w // 2 - ex1, y1 + face_h // 2 - ey1)
    axes = (max(1, int(face_w * 0.9)), max(1, int(face_h * 0.9)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    mask = cv2.GaussianBlur(mask, (71, 71), 0).astype(float) / 255.0
    blended = blurred_roi.astype(float) * mask + roi.astype(float) * (1.0 - mask)
    frame[ey1:ey2, ex1:ex2] = blended.astype(np.uint8)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Blur faces in a video using a YOLO face model.")
    parser.add_argument("--model", required=True, help="YOLO .pt model path")
    parser.add_argument("--input", required=True, help="Input video")
    parser.add_argument("--output", default=None, help="Output video")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_blurred.mp4")
    model = YOLO(args.model)
    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = min(args.max_frames, total) if args.max_frames else total

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cached_boxes = []
    start = time.perf_counter()

    for frame_idx in tqdm(range(frame_limit), desc="face blur video"):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.frame_skip == 0:
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
            )[0]
            cached_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        for box in cached_boxes:
            frame = oval_blur(frame, box)
        writer.write(frame)

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    print(f"saved {output_path}")
    print(f"speed={frame_limit / elapsed:.2f} fps")


if __name__ == "__main__":
    main()
