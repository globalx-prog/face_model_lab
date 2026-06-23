# Blurrt Gesichter in Videos mit einem YOLO/RT-DETR-.pt oder Torchvision-.pth Modell.
# Wichtige Parameter: --model, --model-kind auto|fasterrcnn|retinanet|fcos,
# --input, --output, --conf, --imgsz, --frame-skip, --max-frames,
# --deinterlace, --blur-mode oval|box|pixelate.
# Fuer gezieltes Blurring: erst --mode preview ausfuehren, dann mit
# --target-track-ids oder --target-ranks konkrete erkannte Gesichter blurren.
# Beispiel:
#   python face_model_lab/step08_blur_video.py --model trained_models/yolo_yolov8m_widerface_rocm_bs2_red1_ep15.pt --input Videos/Feuerwehr_progressiv.mp4 --output Videos/lab_outputs/Feuerwehr_blur.mp4 --conf 0.25 --max-frames 300

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
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from tqdm.auto import tqdm
from ultralytics import YOLO


def parse_csv_values(value: str | None, cast=str) -> set:
    if not value:
        return set()
    return {cast(part.strip()) for part in value.split(",") if part.strip()}


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


def build_detection(box, score: float | None = None, track_id: int | None = None, rank: int | None = None) -> dict:
    return {
        "box": np.asarray(box, dtype=float),
        "score": None if score is None else float(score),
        "track_id": None if track_id is None else int(track_id),
        "rank": rank,
    }


def assign_left_to_right_ranks(detections: list[dict]) -> list[dict]:
    ranked = sorted(detections, key=lambda det: (float(det["box"][0]), float(det["box"][1])))
    for rank, det in enumerate(ranked, start=1):
        det["rank"] = rank
    return detections


def box_region(box, width: int, height: int) -> set[str]:
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    regions = set()
    if cx < width / 3:
        regions.add("left")
    elif cx > width * 2 / 3:
        regions.add("right")
    else:
        regions.add("center")
    if cy < height / 3:
        regions.add("top")
    elif cy > height * 2 / 3:
        regions.add("bottom")
    else:
        regions.add("middle")
    return regions


def select_detections(detections: list[dict], args, frame_shape) -> list[dict]:
    target_track_ids = parse_csv_values(args.target_track_ids, int)
    target_ranks = parse_csv_values(args.target_ranks, int)
    target_regions = parse_csv_values(args.target_regions, str)
    has_filter = bool(target_track_ids or target_ranks or target_regions)
    if not has_filter:
        return detections

    height, width = frame_shape[:2]
    selected = []
    for det in detections:
        if target_track_ids and det.get("track_id") in target_track_ids:
            selected.append(det)
            continue
        if target_ranks and det.get("rank") in target_ranks:
            selected.append(det)
            continue
        if target_regions and box_region(det["box"], width, height) & target_regions:
            selected.append(det)
    return selected


def draw_detection_labels(frame, detections: list[dict], selected: list[dict]) -> np.ndarray:
    selected_keys = {(det.get("track_id"), det.get("rank")) for det in selected}
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        is_selected = (det.get("track_id"), det.get("rank")) in selected_keys
        color = (0, 220, 0) if is_selected else (0, 165, 255)
        label_parts = [f"rank {det.get('rank')}"]
        if det.get("track_id") is not None:
            label_parts.append(f"id {det['track_id']}")
        if det.get("score") is not None:
            label_parts.append(f"{det['score']:.2f}")
        label = " | ".join(label_parts)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        y_text = max(16, y1 - 6)
        cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame


def infer_model_kind(model_path: Path) -> str:
    name = model_path.stem.lower()
    if "retinanet" in name:
        return "retinanet"
    if "fcos" in name:
        return "fcos"
    if "fasterrcnn" in name or "faster_rcnn" in name:
        return "fasterrcnn"
    return "fasterrcnn"


def build_fasterrcnn(num_classes: int = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_retinanet(num_classes: int = 2):
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, weights_backbone=None)
    old_head = model.head.classification_head
    model.head.classification_head = RetinaNetClassificationHead(
        old_head.cls_logits.in_channels,
        old_head.num_anchors,
        num_classes,
    )
    return model


def build_fcos(num_classes: int = 2):
    model = torchvision.models.detection.fcos_resnet50_fpn(weights=None, weights_backbone=None)
    old_head = model.head.classification_head
    model.head.classification_head = FCOSClassificationHead(
        old_head.cls_logits.in_channels,
        old_head.num_anchors,
        num_classes,
    )
    return model


def build_torchvision_model(kind: str):
    if kind == "fasterrcnn":
        return build_fasterrcnn()
    if kind == "retinanet":
        return build_retinanet()
    if kind == "fcos":
        return build_fcos()
    raise ValueError(f"Unsupported torchvision model kind: {kind}")


def frame_to_tensor(frame, device: torch.device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def load_detector(model_path: Path, model_kind: str, conf: float, imgsz: int):
    if model_path.suffix == ".pt":
        model = YOLO(str(model_path))
        device = 0 if torch.cuda.is_available() else "cpu"

        def detect(frame):
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=conf,
                imgsz=imgsz,
                device=device,
            )[0]
            if result.boxes is None:
                return []
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [None] * len(boxes)
            track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else [None] * len(boxes)
            detections = [
                build_detection(box, score=score, track_id=track_id)
                for box, score, track_id in zip(boxes, scores, track_ids)
            ]
            return assign_left_to_right_ranks(detections)

        return detect

    if model_path.suffix == ".pth":
        kind = infer_model_kind(model_path) if model_kind == "auto" else model_kind
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_torchvision_model(kind).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"loaded Torchvision {kind} model on {device}")

        def detect(frame):
            with torch.inference_mode():
                prediction = model([frame_to_tensor(frame, device)])[0]
            scores = prediction["scores"].detach().cpu().numpy()
            boxes = prediction["boxes"].detach().cpu().numpy()
            detections = [
                build_detection(box, score=score)
                for box, score in zip(boxes, scores)
                if score >= conf
            ]
            return assign_left_to_right_ranks(detections)

        return detect

    raise ValueError("--model must be a .pt Ultralytics model or a .pth Torchvision model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blur faces in a video using a YOLO .pt or Torchvision .pth face model.")
    parser.add_argument("--model", required=True, help="YOLO .pt or Torchvision .pth model path")
    parser.add_argument("--model-kind", choices=["auto", "fasterrcnn", "retinanet", "fcos"], default="auto", help="Torchvision architecture for .pth models; inferred from filename by default.")
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
    parser.add_argument("--mode", choices=["blur", "preview"], default="blur", help="blur anonymizes selected faces; preview draws boxes/ranks/track IDs without blurring.")
    parser.add_argument("--target-track-ids", default="", help="Comma-separated YOLO/ByteTrack IDs to blur, e.g. 1,4. Empty means all faces unless another target filter is set.")
    parser.add_argument("--target-ranks", default="", help="Comma-separated left-to-right detection ranks per frame, e.g. 1,3. Useful when track IDs are unavailable.")
    parser.add_argument("--target-regions", default="", help="Comma-separated regions to blur: left,center,right,top,middle,bottom.")
    args = parser.parse_args()

    input_path = Path(args.input)
    input_path = prepare_video(input_path, args.deinterlace)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_blurred.mp4")
    detector = load_detector(Path(args.model), args.model_kind, args.conf, args.imgsz)

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
    cached_detections = []
    start = time.perf_counter()
    processed_faces = 0
    selected_faces = 0

    for frame_idx in tqdm(range(frame_limit), desc="face blur video"):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.frame_skip == 0:
            cached_detections = detector(frame)
        selected_detections = select_detections(cached_detections, args, frame.shape)
        processed_faces += len(cached_detections)
        selected_faces += len(selected_detections)
        if args.mode == "preview":
            frame = draw_detection_labels(frame, cached_detections, selected_detections)
            writer.write(frame)
            continue
        for det in selected_detections:
            box = det["box"]
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
    print(f"detections_seen={processed_faces}")
    print(f"detections_selected={selected_faces}")
    if args.mode == "preview":
        print("preview mode: boxes show rank and YOLO track id. Use --target-track-ids or --target-ranks for the second blur run.")


if __name__ == "__main__":
    main()
