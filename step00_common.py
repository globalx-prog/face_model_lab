from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "matplotlib_cache"))

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "wider_face"
ANNOTATIONS_DIR = ROOT / "annotations"
MODEL_DIR = ROOT / "trained_models"
RESULTS_DIR = ROOT / "model_results"
YOLO_DATASET_DIR = ROOT / "datasets" / "wider_face_yolo"


def ensure_dirs() -> None:
    for directory in [ANNOTATIONS_DIR, MODEL_DIR, RESULTS_DIR, YOLO_DATASET_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def rocm_device(require_gpu: bool = True) -> torch.device:
    if torch.cuda.is_available():
        print(f"Using ROCm device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError("ROCm GPU is not visible to PyTorch. Run outside sandbox and check /dev/kfd.")
    print("ROCm GPU not visible; falling back to CPU.")
    return torch.device("cpu")


def vram_status(device: torch.device | str | None = None) -> dict[str, str]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        return {"vram": "cpu"}

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    try:
        free, total = torch.cuda.mem_get_info(device)
        return {
            "alloc": f"{allocated:.1f}G",
            "res": f"{reserved:.1f}G",
            "free": f"{free / 1024**3:.1f}G",
            "total": f"{total / 1024**3:.0f}G",
        }
    except Exception:
        return {"alloc": f"{allocated:.1f}G", "res": f"{reserved:.1f}G"}


def model_name(model_type: str, batch_size: int, epochs: int, suffix: str) -> str:
    clean_type = model_type.replace("/", "_").replace("-", "").lower()
    return f"{clean_type}_bs{batch_size}_ep{epochs}.{suffix}"


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_wider_face_gt(gt_file: Path) -> dict[str, list[dict[str, list[float]]]]:
    lines = gt_file.read_text(encoding="utf-8", errors="replace").splitlines()
    annotations: dict[str, list[dict[str, list[float]]]] = {}
    cursor = 0

    while cursor < len(lines):
        rel_path = lines[cursor].strip()
        cursor += 1
        if not rel_path or cursor >= len(lines):
            continue

        try:
            face_count = int(lines[cursor].strip())
        except ValueError:
            break
        cursor += 1

        faces = []
        for _ in range(face_count):
            if cursor >= len(lines):
                break
            parts = lines[cursor].strip().split()
            cursor += 1
            if len(parts) < 4:
                continue
            x, y, w, h = map(float, parts[:4])
            if w > 0 and h > 0:
                faces.append({"bbox": [x, y, w, h]})
        annotations[rel_path] = faces

    return annotations


def wider_paths(split: str) -> tuple[Path, Path]:
    if split == "train":
        image_dir = DATASET_DIR / "WIDER_train" / "WIDER_train" / "images"
        gt_file = DATASET_DIR / "wider_face_split" / "wider_face_split" / "wider_face_train_bbx_gt.txt"
    elif split == "val":
        image_dir = DATASET_DIR / "WIDER_val" / "WIDER_val" / "images"
        gt_file = DATASET_DIR / "wider_face_split" / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    else:
        raise ValueError(f"Unknown split: {split}")
    return image_dir, gt_file


def xywh_to_xyxy(face: dict[str, list[float]]) -> list[float]:
    x, y, w, h = face["bbox"]
    return [x, y, x + w, y + h]


def compute_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def count_matches(gt_faces: list[dict[str, list[float]]], pred_boxes: Iterable[Iterable[float]], iou_thresh: float) -> int:
    predictions = [list(box) for box in pred_boxes]
    matched = 0
    used: set[int] = set()
    for face in gt_faces:
        gt_box = xywh_to_xyxy(face)
        best_idx = None
        best_iou = 0.0
        for idx, pred_box in enumerate(predictions):
            if idx in used:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_idx = idx
                best_iou = iou
        if best_idx is not None and best_iou >= iou_thresh:
            used.add(best_idx)
            matched += 1
    return matched


@dataclass
class EvalResult:
    model: str
    images: int
    true_faces: int
    detected_faces: int
    recall: float
    ms_per_image: float


def write_results_csv(path: Path, rows: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvalResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_results_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
