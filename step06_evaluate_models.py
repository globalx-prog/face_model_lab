# Evaluiert trainierte Face-Detektoren auf WIDER FACE Val-Samples.
# Wichtige Parameter: --models <.pt/.pth ...>, --limit, --iou, --conf,
# --imgsz, --score-thresholds, --include-pretrained-coco.
# Beispiel:
#   python face_model_lab/step06_evaluate_models.py --models trained_models/fasterrcnn_resnet50_fpn_rocm_bs2_red2_ep10.pth trained_models/yolo_yolov8m_widerface_rocm_bs2_red1_ep15.pt --limit 100 --conf 0.25

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "matplotlib_cache"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from tqdm.auto import tqdm
from ultralytics import RTDETR, YOLO

from step00_common import (
    DATASET_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    ROOT,
    EvalResult,
    compare_train_val,
    count_matches,
    ensure_dirs,
    parse_wider_face_gt,
    timestamp,
    ultralytics_device,
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
    "pretrained_fasterrcnn": {
        "family": "Pretrained Faster R-CNN / COCO",
        "strengths": "Nur Fallback-Baseline: prüft Pipeline und Geschwindigkeit ohne lokal trainiertes Face-Modell.",
        "watch": "Nicht auf Gesichter trainiert. Ergebnisse nicht als Face-Detection-Qualität interpretieren.",
    },
    "pretrained_retinanet": {
        "family": "Pretrained RetinaNet / COCO",
        "strengths": "COCO-vortrainierte One-Stage-Baseline mit Focal Loss; nützlich als Negativ-/Plausibilitätsvergleich.",
        "watch": "Nicht face-spezifisch trainiert. Gute COCO-Objekterkennung bedeutet nicht gute Gesichtserkennung.",
    },
    "pretrained_fcos": {
        "family": "Pretrained FCOS / COCO",
        "strengths": "COCO-vortrainierte anchor-free Baseline; zeigt, ob reine COCO-Features zufällig Gesichter treffen.",
        "watch": "Nicht face-spezifisch trainiert. Ergebnisse nur als Baseline interpretieren.",
    },
    "pretrained_yolo": {
        "family": "Pretrained YOLO / COCO",
        "strengths": "Sehr schnelle COCO-Baseline; praktisch zum Gegencheck gegen ein face-finetuned YOLO.",
        "watch": "Nicht auf Face-Boxes trainiert. Person-/Objektboxen passen in der Regel schlecht auf Gesichtsbboxen.",
    },
    "pretrained_rtdetr": {
        "family": "Pretrained RT-DETR / COCO",
        "strengths": "Transformer-basierte COCO-Baseline als Alternative zu YOLO/Faster R-CNN.",
        "watch": "Nicht face-spezifisch trainiert und auf ROCm teils schwerer/langsamer.",
    },
    "insightface": {
        "family": "InsightFace",
        "strengths": "Starke klassische Face-Analysis-Baseline, oft gut für Gesichter ohne eigenes Training.",
        "watch": "Externe Pipeline, nicht dein eigenes trainiertes Modell; Provider/CPU kann langsam sein.",
    },
    "mtcnn": {
        "family": "MTCNN",
        "strengths": "Bewährte Face-Detection-Baseline, gut als Plausibilitätsvergleich.",
        "watch": "Oft langsamer und weniger robust bei kleinen/weit entfernten Gesichtern.",
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


def resize_kwargs(min_size: int | None = None, max_size: int | None = None) -> dict:
    kwargs = {}
    if min_size is not None:
        kwargs["min_size"] = min_size
    if max_size is not None:
        kwargs["max_size"] = max_size
    return kwargs


def build_fasterrcnn(num_classes: int = 2, min_size: int | None = None, max_size: int | None = None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", **resize_kwargs(min_size, max_size))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_pretrained_fasterrcnn_coco():
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")


def build_pretrained_retinanet_coco():
    return torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")


def build_pretrained_fcos_coco():
    return torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")


def build_retinanet(num_classes: int = 2, min_size: int | None = None, max_size: int | None = None):
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT", **resize_kwargs(min_size, max_size))
    old_head = model.head.classification_head
    model.head.classification_head = RetinaNetClassificationHead(
        old_head.cls_logits.in_channels,
        old_head.num_anchors,
        num_classes,
    )
    return model


def build_fcos(num_classes: int = 2, min_size: int | None = None, max_size: int | None = None):
    model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT", **resize_kwargs(min_size, max_size))
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


def build_pretrained_torchvision_model(kind: str):
    if kind == "fasterrcnn":
        return build_pretrained_fasterrcnn_coco()
    if kind == "retinanet":
        return build_pretrained_retinanet_coco()
    if kind == "fcos":
        return build_pretrained_fcos_coco()
    raise ValueError(f"Unsupported pretrained torchvision kind: {kind}")


def evaluate_yolo(model_path: Path, samples, image_dir: Path, iou: float, conf: float, imgsz: int) -> EvalResult:
    model = YOLO(str(model_path))
    return evaluate_ultralytics_model(model, model_path.name, samples, image_dir, iou, conf, imgsz)


def evaluate_ultralytics_model(model, label: str, samples, image_dir: Path, iou: float, conf: float, imgsz: int) -> EvalResult:
    detected = 0
    true_faces = 0
    device = ultralytics_device()
    t0 = time.perf_counter()
    for rel_path, faces in tqdm(samples, desc=f"eval {label}"):
        image = cv2.imread(str(image_dir / rel_path))
        if image is None:
            continue
        result = model(image, verbose=False, conf=conf, imgsz=imgsz, device=device)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        detected += count_matches(faces, pred_boxes, iou)
        true_faces += len(faces)
    elapsed = time.perf_counter() - t0
    recall = detected / true_faces if true_faces else 0.0
    return EvalResult(label, len(samples), true_faces, detected, recall, elapsed / max(len(samples), 1) * 1000)


def evaluate_insightface(samples, image_dir: Path, iou: float) -> EvalResult | None:
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        print(f"Skipping InsightFace baseline: {exc}")
        return None

    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as exc:
        print(f"Skipping InsightFace baseline: {exc}")
        return None

    detected = 0
    true_faces = 0
    t0 = time.perf_counter()
    for rel_path, faces in tqdm(samples, desc="eval InsightFace"):
        image = cv2.imread(str(image_dir / rel_path))
        if image is None:
            continue
        result = app.get(image)
        pred_boxes = [face.bbox.astype(float) for face in result]
        detected += count_matches(faces, pred_boxes, iou)
        true_faces += len(faces)
    elapsed = time.perf_counter() - t0
    recall = detected / true_faces if true_faces else 0.0
    return EvalResult("BASELINE_InsightFace_buffalo_l", len(samples), true_faces, detected, recall, elapsed / max(len(samples), 1) * 1000)


def evaluate_mtcnn(samples, image_dir: Path, iou: float) -> EvalResult | None:
    try:
        from facenet_pytorch import MTCNN
    except Exception as exc:
        print(f"Skipping MTCNN baseline: {exc}")
        return None

    detector = MTCNN(keep_all=True, device=torch.device("cpu"))
    detected = 0
    true_faces = 0
    t0 = time.perf_counter()
    for rel_path, faces in tqdm(samples, desc="eval MTCNN"):
        image = cv2.imread(str(image_dir / rel_path))
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_boxes, _ = detector.detect(rgb)
        detected += count_matches(faces, pred_boxes if pred_boxes is not None else [], iou)
        true_faces += len(faces)
    elapsed = time.perf_counter() - t0
    recall = detected / true_faces if true_faces else 0.0
    return EvalResult("BASELINE_MTCNN", len(samples), true_faces, detected, recall, elapsed / max(len(samples), 1) * 1000)


def evaluate_torchvision(model_path: Path, samples, image_dir: Path, iou: float, conf: float, kind: str) -> EvalResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torchvision_model(kind).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return evaluate_torchvision_model(model, model_path.name, samples, image_dir, iou, conf, device)


def evaluate_pretrained_fallback(samples, image_dir: Path, iou: float, conf: float) -> EvalResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pretrained_fasterrcnn_coco().to(device)
    return evaluate_torchvision_model(
        model,
        "PRETRAINED_COCO_fasterrcnn_resnet50_fpn",
        samples,
        image_dir,
        iou,
        conf,
        device,
    )


def evaluate_pretrained_torchvision(kind: str, samples, image_dir: Path, iou: float, conf: float) -> EvalResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pretrained_torchvision_model(kind).to(device)
    return evaluate_torchvision_model(
        model,
        f"BASELINE_PRETRAINED_COCO_{kind}",
        samples,
        image_dir,
        iou,
        conf,
        device,
    )


def evaluate_pretrained_ultralytics(base: str, family: str, samples, image_dir: Path, iou: float, conf: float, imgsz: int) -> EvalResult:
    model = RTDETR(base) if family == "rtdetr" else YOLO(base)
    return evaluate_ultralytics_model(
        model,
        f"BASELINE_PRETRAINED_COCO_{Path(base).stem}",
        samples,
        image_dir,
        iou,
        conf,
        imgsz,
    )


def evaluate_torchvision_model(model, label: str, samples, image_dir: Path, iou: float, conf: float, device: torch.device) -> EvalResult:
    model.eval()
    transform = T.ToTensor()

    detected = 0
    true_faces = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for rel_path, faces in tqdm(samples, desc=f"eval {label}"):
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
    return EvalResult(label, len(samples), true_faces, detected, recall, elapsed / max(len(samples), 1) * 1000)


def threshold_sweep_torchvision(model_path: Path, samples, image_dir: Path, iou: float, thresholds: list[float], kind: str) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torchvision_model(kind).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    transform = T.ToTensor()
    rows = []
    with torch.no_grad():
        cached = []
        for rel_path, faces in tqdm(samples, desc=f"threshold sweep {model_path.name}"):
            image = cv2.imread(str(image_dir / rel_path))
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred = model([transform(rgb).to(device)])[0]
            cached.append((faces, pred["boxes"].detach().cpu().numpy(), pred["scores"].detach().cpu().numpy()))
    for threshold in thresholds:
        true_faces = 0
        detected = 0
        predicted = 0
        for faces, boxes, scores in cached:
            keep = scores >= threshold
            pred_boxes = boxes[keep]
            predicted += len(pred_boxes)
            detected += count_matches(faces, pred_boxes, iou)
            true_faces += len(faces)
        recall = detected / true_faces if true_faces else 0.0
        precision = detected / predicted if predicted else 0.0
        rows.append({
            "model": model_path.name,
            "score_threshold": threshold,
            "true_faces": true_faces,
            "predicted_faces": predicted,
            "detected_faces": detected,
            "precision": precision,
            "recall": recall,
        })
    return rows


def discover_local_models() -> list[Path]:
    candidates: list[Path] = []
    candidates += sorted(MODEL_DIR.glob("*.pth"))
    candidates += sorted(MODEL_DIR.glob("*.pt"))
    candidates += [ROOT / "face_yolov8m.pt", ROOT / "rtdetr-l.pt"]

    seen = set()
    existing = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists() and candidate not in seen:
            existing.append(candidate)
            seen.add(candidate)
    return existing


def save_validation_plots(stamp: str, rows: list[EvalResult], dataset_comparison: list[dict], threshold_sweeps: list[dict]) -> list[str]:
    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    if dataset_comparison:
        labels = [row["split"] for row in dataset_comparison]
        images = [row["images"] for row in dataset_comparison]
        faces = [row["faces"] for row in dataset_comparison]
        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - width / 2, images, width, label="images")
        ax.bar(x + width / 2, faces, width, label="faces")
        ax.set_xticks(x, labels)
        ax.set_title("Train vs. validation data")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        path = plot_dir / f"dataset_train_vs_val_{stamp}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(str(path))

    if rows:
        names = [row.model for row in rows]
        recall = [row.recall for row in rows]
        latency = [row.ms_per_image for row in rows]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.8))
        ax.bar(names, recall, color="#1f77b4")
        ax.set_ylim(0, 1)
        ax.set_title("Validation detection accuracy / recall")
        ax.set_ylabel("matched faces / true faces")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = plot_dir / f"validation_recall_{stamp}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(str(path))

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.8))
        ax.bar(names, latency, color="#ff7f0e")
        ax.set_title("Validation inference time")
        ax.set_ylabel("ms / image")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = plot_dir / f"validation_latency_{stamp}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(str(path))

    if threshold_sweeps:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        grouped: dict[str, list[dict]] = {}
        for row in threshold_sweeps:
            grouped.setdefault(row["model"], []).append(row)
        for model, values in grouped.items():
            values = sorted(values, key=lambda item: item["score_threshold"])
            ax.plot(
                [item["score_threshold"] for item in values],
                [item["recall"] for item in values],
                marker="o",
                label=f"{model} recall",
            )
            ax.plot(
                [item["score_threshold"] for item in values],
                [item["precision"] for item in values],
                marker="x",
                linestyle="--",
                label=f"{model} precision",
            )
        ax.set_ylim(0, 1)
        ax.set_title("Score threshold sweep")
        ax.set_xlabel("score threshold")
        ax.set_ylabel("metric")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / f"threshold_sweep_{stamp}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(str(path))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained face detectors on WIDER FACE validation samples.")
    parser.add_argument("--models", nargs="+", help="Model files: .pt Ultralytics or .pth Torchvision. If omitted, local trained models are discovered.")
    parser.add_argument("--allow-pretrained-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-insightface", action="store_true", help="Also evaluate InsightFace as external baseline.")
    parser.add_argument("--include-mtcnn", action="store_true", help="Also evaluate MTCNN as external baseline.")
    parser.add_argument(
        "--include-pretrained-coco",
        nargs="*",
        choices=["fasterrcnn", "retinanet", "fcos", "yolov8m", "rtdetr-l"],
        default=[],
        help="Add COCO-pretrained baselines. These are not face-finetuned and are highlighted as baselines.",
    )
    parser.add_argument("--score-thresholds", nargs="*", type=float, default=[], help="Optional score thresholds for precision/recall sweep on Torchvision .pth models.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    dataset_comparison = compare_train_val()
    print("Train-vs-Val dataset comparison:")
    for split in dataset_comparison:
        print(
            f"  {split['split']}: images={split['images']}, faces={split['faces']}, "
            f"mean_faces/image={split['faces_per_image_mean']:.2f}, max_faces/image={split['faces_per_image_max']}"
        )

    image_dir, gt_file = wider_paths("val")
    annotations = [(path, faces) for path, faces in parse_wider_face_gt(gt_file).items() if faces]
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(annotations))[: args.limit]
    samples = [annotations[int(i)] for i in indices]

    model_paths = [Path(raw_model) for raw_model in args.models] if args.models else discover_local_models()
    missing = [path for path in model_paths if not path.exists()]
    for path in missing:
        print(f"Skipping missing model: {path}")
    model_paths = [path for path in model_paths if path.exists()]

    rows = []
    notes = {}
    threshold_sweeps = []
    used_pretrained_fallback = False

    if not model_paths:
        if not args.allow_pretrained_fallback:
            raise FileNotFoundError("No local trained model found and pretrained fallback is disabled.")
        used_pretrained_fallback = True
        fallback_name = "PRETRAINED_COCO_fasterrcnn_resnet50_fpn"
        notes[fallback_name] = MODEL_NOTES["pretrained_fasterrcnn"]
        print()
        print("!!! PRETRAINED FALLBACK !!!")
        print("No local trained model was found.")
        print("Evaluating a COCO-pretrained Faster R-CNN only as a pipeline/speed baseline.")
        print("This model is NOT face-finetuned; do not compare its recall as trained face-model quality.")
        print()
        rows.append(evaluate_pretrained_fallback(samples, image_dir, args.iou, args.conf))

    for model_path in model_paths:
        kind = infer_model_kind(model_path)
        notes[model_path.name] = MODEL_NOTES.get(kind, MODEL_NOTES["yolo"])
        if model_path.suffix == ".pt":
            rows.append(evaluate_yolo(model_path, samples, image_dir, args.iou, args.conf, args.imgsz))
        elif model_path.suffix == ".pth":
            rows.append(evaluate_torchvision(model_path, samples, image_dir, args.iou, args.conf, kind))
            if args.score_thresholds:
                threshold_sweeps.extend(threshold_sweep_torchvision(model_path, samples, image_dir, args.iou, args.score_thresholds, kind))
        else:
            print(f"Skipping unknown model type: {model_path}")

    for pretrained in args.include_pretrained_coco:
        print()
        print(f"!!! PRETRAINED COCO BASELINE: {pretrained} !!!")
        print("This baseline is NOT face-finetuned; use it only as a reference, not as trained face-model quality.")
        if pretrained in {"fasterrcnn", "retinanet", "fcos"}:
            label = f"BASELINE_PRETRAINED_COCO_{pretrained}"
            notes[label] = MODEL_NOTES.get(f"pretrained_{pretrained}", MODEL_NOTES["pretrained_fasterrcnn"])
            rows.append(evaluate_pretrained_torchvision(pretrained, samples, image_dir, args.iou, args.conf))
        elif pretrained == "yolov8m":
            label = "BASELINE_PRETRAINED_COCO_yolov8m"
            notes[label] = MODEL_NOTES["pretrained_yolo"]
            rows.append(evaluate_pretrained_ultralytics("yolov8m.pt", "yolo", samples, image_dir, args.iou, args.conf, args.imgsz))
        elif pretrained == "rtdetr-l":
            label = "BASELINE_PRETRAINED_COCO_rtdetr-l"
            notes[label] = MODEL_NOTES["pretrained_rtdetr"]
            rows.append(evaluate_pretrained_ultralytics("rtdetr-l.pt", "rtdetr", samples, image_dir, args.iou, args.conf, args.imgsz))

    if args.include_insightface:
        notes["BASELINE_InsightFace_buffalo_l"] = MODEL_NOTES["insightface"]
        result = evaluate_insightface(samples, image_dir, args.iou)
        if result is not None:
            rows.append(result)
    if args.include_mtcnn:
        notes["BASELINE_MTCNN"] = MODEL_NOTES["mtcnn"]
        result = evaluate_mtcnn(samples, image_dir, args.iou)
        if result is not None:
            rows.append(result)

    stamp = timestamp()
    csv_path = RESULTS_DIR / f"evaluation_{stamp}.csv"
    json_path = RESULTS_DIR / f"evaluation_{stamp}.json"
    write_results_csv(csv_path, rows)
    plot_paths = save_validation_plots(stamp, rows, dataset_comparison, threshold_sweeps)
    write_results_json(json_path, {
        "args": vars(args),
        "used_pretrained_fallback": used_pretrained_fallback,
        "dataset_comparison": dataset_comparison,
        "model_notes": notes,
        "threshold_sweeps": threshold_sweeps,
        "validation_plots": plot_paths,
        "results": [row.__dict__ for row in rows],
    })

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    for plot_path in plot_paths:
        print(f"wrote plot {plot_path}")
    for row in rows:
        note = notes.get(row.model, {})
        print(f"{row.model}: detection_accuracy_recall={row.recall:.3f}, ms/image={row.ms_per_image:.1f}")
        if note:
            print(f"  {note['family']}: {note['strengths']}")
            print(f"  Achtung: {note['watch']}")


if __name__ == "__main__":
    main()
