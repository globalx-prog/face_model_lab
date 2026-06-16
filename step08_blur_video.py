from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "matplotlib_cache"))

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO


def odd_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def check_if_interlaced(video_path: Path, num_frames_to_test: int = 30, skip_seconds: int = 0) -> bool:
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-ss",
        str(skip_seconds),
        "-filter_complex",
        "idet",
        "-frames:v",
        str(num_frames_to_test),
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    match = re.search(
        r"Multi frame detection:\s+TFF:\s*(\d+)\s+BFF:\s*(\d+)\s+Progressive:\s*(\d+)\s+Undetermined:\s*(\d+)",
        result.stderr,
    )
    if not match:
        print("FFmpeg idet output could not be parsed; using deinterlace as safe fallback.")
        return True
    tff, bff, progressive, undetermined = map(int, match.groups())
    total_interlaced = tff + bff
    total_scanned = total_interlaced + progressive + undetermined
    print(f"idet: interlaced={total_interlaced}, progressive={progressive}, undetermined={undetermined}")
    if total_scanned == 0 or undetermined > total_scanned * 0.8:
        return True
    return total_interlaced > progressive * 0.1


def run_ffmpeg_deinterlace(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "yadif=1",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def prepare_video(input_path: Path, mode: str) -> Path:
    if mode == "never":
        return input_path
    deinterlaced = input_path.with_name(input_path.stem + "_progressiv" + input_path.suffix)
    if mode == "smoke":
        return deinterlaced if deinterlaced.exists() else input_path
    if mode == "always":
        return deinterlaced if deinterlaced.exists() else run_ffmpeg_deinterlace(input_path, deinterlaced)
    if mode == "auto":
        if check_if_interlaced(input_path):
            return deinterlaced if deinterlaced.exists() else run_ffmpeg_deinterlace(input_path, deinterlaced)
        return input_path
    raise ValueError("--deinterlace must be auto, always, never, or smoke")


def oval_blur(frame, box, padding_factor: float = 0.5, blur_kernel: int = 99, mask_kernel: int = 71):
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

    blur_kernel = odd_kernel(blur_kernel)
    mask_kernel = odd_kernel(mask_kernel)
    blurred_roi = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)
    mask = np.zeros_like(roi, dtype=np.uint8)
    center = (x1 + face_w // 2 - ex1, y1 + face_h // 2 - ey1)
    axes = (max(1, int(face_w * 0.9)), max(1, int(face_h * 0.9)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    mask = cv2.GaussianBlur(mask, (mask_kernel, mask_kernel), 0).astype(float) / 255.0
    blended = blurred_roi.astype(float) * mask + roi.astype(float) * (1.0 - mask)
    frame[ey1:ey2, ex1:ex2] = blended.astype(np.uint8)
    return frame


def expanded_roi(frame, box, padding_factor: float):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    face_w = x2 - x1
    face_h = y2 - y1
    if face_w <= 0 or face_h <= 0:
        return None
    pad_x = int(face_w * padding_factor)
    pad_y = int(face_h * padding_factor)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def box_blur(frame, box, padding_factor: float = 0.35, blur_kernel: int = 99):
    roi_coords = expanded_roi(frame, box, padding_factor)
    if roi_coords is None:
        return frame
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    blur_kernel = odd_kernel(blur_kernel)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)
    return frame


def pixelate(frame, box, padding_factor: float = 0.35, pixel_size: int = 18):
    roi_coords = expanded_roi(frame, box, padding_factor)
    if roi_coords is None:
        return frame
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    h, w = roi.shape[:2]
    scale_w = max(1, w // max(2, pixel_size))
    scale_h = max(1, h // max(2, pixel_size))
    small = cv2.resize(roi, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
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
    parser.add_argument("--deinterlace", choices=["auto", "always", "never", "smoke"], default="smoke")
    parser.add_argument("--background-threshold-percent", type=float, default=0.0, help="Only blur faces whose box area is >= this fraction of the frame. 0 disables filtering.")
    parser.add_argument("--padding-factor", type=float, default=0.5)
    parser.add_argument("--blur-kernel", type=int, default=99)
    parser.add_argument("--mask-kernel", type=int, default=71)
    parser.add_argument("--blur-mode", choices=["oval", "box", "pixelate"], default="oval")
    parser.add_argument("--pixel-size", type=int, default=18)
    args = parser.parse_args()

    input_path = Path(args.input)
    input_path = prepare_video(input_path, args.deinterlace)
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
    frame_area = max(1, width * height)
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
            x1, y1, x2, y2 = map(float, box)
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if args.background_threshold_percent and box_area / frame_area < args.background_threshold_percent:
                continue
            if args.blur_mode == "oval":
                frame = oval_blur(frame, box, args.padding_factor, args.blur_kernel, args.mask_kernel)
            elif args.blur_mode == "box":
                frame = box_blur(frame, box, args.padding_factor, args.blur_kernel)
            elif args.blur_mode == "pixelate":
                frame = pixelate(frame, box, args.padding_factor, args.pixel_size)
        writer.write(frame)

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start
    print(f"saved {output_path}")
    print(f"speed={frame_limit / elapsed:.2f} fps")


if __name__ == "__main__":
    main()
