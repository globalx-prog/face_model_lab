from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "matplotlib_cache"))

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from tqdm.auto import tqdm
from ultralytics import YOLO

from common import (
    DATASET_DIR,
    RESULTS_DIR,
    EvalResult,
    count_matches,
    ensure_dirs,
    parse_wider_face_gt,
    timestamp,
    wider_paths,
    write_results_csv,
    write_results_json,
)


MODEL_NOTES = {
    "yolo": {
        "family": "YOLO / Ultralytics",
        "strengths": "Sehr stark für Video: schnelle One-Stage-Detection, einfache ROCm-Nutzung, Tracking direkt verfügbar.",
        "watch": "Kann kleine/weit entfernte Gesichter verpassen, wenn imgsz/conf zu konservativ gewählt sind.",
    },
    "rtdetr": {
        "family": "RT-DETR / Ultralytics",
        "strengths": "End-to-end Transformer-Detector ohne klassisches NMS; gute Accuracy-Speed-Balance als YOLO-Alternative.",
        "watch": "Training und Inferenz können schwerer sein als YOLO; für kleine Gesichter muss imgsz oft höher sein.",
    },
    "fasterrcnn": {
        "family": "Faster R-CNN",
        "strengths": "Solide Two-Stage-Qualitätsbaseline; gute Lokalisierung und interpretierbares Training.",
        "watch": "Für Video eher langsam und VRAM/CPU-overhead-intensiv.",
    },
    "retinanet": {
        "family": "RetinaNet",
        "strengths": "One-Stage-Detector mit Focal Loss; interessant bei vielen einfachen Negativen und kleinen Objekten.",
        "watch": "Meist langsamer/umständlicher als YOLO in der Video-Pipeline.",
    },
    "fcos": {
        "family": "FCOS",
        "strengths": "Anchor-free One-Stage-Detector; weniger Anchor-Tuning, gute Forschungsbaseline.",
        "watch": "Kann bei sehr kleinen Gesichtern empfindlich auf Auflösung und Trainingsdauer reagieren.",
    },
}


def infer_model_kind(model_path: Path) -> str:
    name = model_path.stem.lower()
    if "rtdetr" in name:
        return "rtdetr"
    if "retinanet" in name:
        return "retinanet"
    if "fcos" in name:
        return "fcos"
    if "fasterrcnn" in name or "faster_rcnn" in name:
        return "fasterrcnn"
    if model_path.suffix == ".pt":
        return "yolo"
    return "fasterrcnn"


def build_fasterrcnn(num_classes: int = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_retinanet(num_classes: int = 2):
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
    old_head = model.head.classification_head
    model.head.classification_head = RetinaNetClassificationHead(
        old_head.cls_logits.in_channels,
        old_head.num_anchors,
        num_classes,
    )
    return model


def build_fcos(num_classes: int = 2):
    model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
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


def evaluate_yolo(model_path: Path, samples, image_dir: Path, iou: float, conf: float, imgsz: int) -> EvalResult:
    model = YOLO(str(model_path))
    detected = 0
    true_faces = 0
    t0 = time.perf_counter()
    for rel_path, faces in tqdm(samples, desc=f"eval {model_path.name}"):
        image = cv2.imread(str(image_dir / rel_path))
        if image is None:
            continue
        result = model(image, verbose=False, conf=conf, imgsz=imgsz, device=0 if torch.cuda.is_available() else "cpu")[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        detected += count_matches(faces, pred_boxes, iou)
        true_faces += len(faces)
    elapsed = time.perf_counter() - t0
    recall = detected / true_faces if true_faces else 0.0
    return EvalResult(model_path.name, len(samples), true_faces, detected, recall, elapsed / max(len(samples), 1) * 1000)


def evaluate_torchvision(model_path: Path, samples, image_dir: Path, iou: float, conf: float, kind: str) -> EvalResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torchvision_model(kind).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    transform = T.ToTensor()

    detected = 0
    true_faces = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for rel_path, faces in tqdm(samples, desc=f"eval {model_path.name}"):
            image = cv2.imread(str(image_dir / rel_path))
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred = model([transform(rgb).to(device)])[0]
            keep = pred["scores"].detach().cpu().numpy() >= conf
            pred_boxes = pred["boxes"].detach().cpu().numpy()[keep]
            detected += count_matches(faces, pred_boxes, iou)
            true_faces += len(faces)
    elapsed = time.perf_counter() - t0
    recall = detected / true_faces if true_faces else 0.0
    return EvalResult(model_path.name, len(samples), true_faces, detected, recall, elapsed / max(len(samples), 1) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained face detectors on WIDER FACE validation samples.")
    parser.add_argument("--models", nargs="+", required=True, help="Model files: .pt Ultralytics or .pth Torchvision")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    image_dir, gt_file = wider_paths("val")
    annotations = [(path, faces) for path, faces in parse_wider_face_gt(gt_file).items() if faces]
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(annotations))[: args.limit]
    samples = [annotations[int(i)] for i in indices]

    rows = []
    notes = {}
    for raw_model in args.models:
        model_path = Path(raw_model)
        kind = infer_model_kind(model_path)
        notes[model_path.name] = MODEL_NOTES.get(kind, MODEL_NOTES["yolo"])
        if model_path.suffix == ".pt":
            rows.append(evaluate_yolo(model_path, samples, image_dir, args.iou, args.conf, args.imgsz))
        elif model_path.suffix == ".pth":
            rows.append(evaluate_torchvision(model_path, samples, image_dir, args.iou, args.conf, kind))
        else:
            print(f"Skipping unknown model type: {model_path}")

    stamp = timestamp()
    csv_path = RESULTS_DIR / f"evaluation_{stamp}.csv"
    json_path = RESULTS_DIR / f"evaluation_{stamp}.json"
    write_results_csv(csv_path, rows)
    write_results_json(json_path, {"args": vars(args), "model_notes": notes, "results": [row.__dict__ for row in rows]})

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    for row in rows:
        note = notes.get(row.model, {})
        print(f"{row.model}: recall={row.recall:.3f}, ms/image={row.ms_per_image:.1f}")
        if note:
            print(f"  {note['family']}: {note['strengths']}")
            print(f"  Achtung: {note['watch']}")


if __name__ == "__main__":
    main()
