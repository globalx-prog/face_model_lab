from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "matplotlib_cache"))

import cv2
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO, RTDETR

from step00_common import MODEL_DIR, ROOT, YOLO_DATASET_DIR, ensure_dirs, model_name, parse_wider_face_gt, vram_status, wider_paths


KNOWN_ULTRALYTICS_STEMS = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "rtdetr-l.pt",
    "rtdetr-x.pt",
}


def resolve_base_model(base: str) -> str:
    candidate = Path(base).expanduser()
    search_paths = [
        candidate,
        ROOT / candidate,
        MODEL_DIR / candidate,
    ]
    for path in search_paths:
        if path.exists():
            return str(path)

    if candidate.name in KNOWN_ULTRALYTICS_STEMS:
        print(f"Base model {base} not found locally; Ultralytics may download it if network is available.")
        return base

    checked = "\n".join(f"  - {path}" for path in search_paths)
    raise FileNotFoundError(
        f"Base model '{base}' was not found.\n"
        f"Checked:\n{checked}\n\n"
        "Options:\n"
        "  1. Put your model at /home/clemi/projekte/MIM/<model>.pt\n"
        "  2. Pass an existing absolute path via --base /path/to/model.pt\n"
        "  3. Use an official Ultralytics base, e.g. --base yolov8n.pt, yolov8m.pt, or rtdetr-l.pt "
        "(requires network on first run and may not be face-pretrained)."
    )


def prepare_split(split: str, limit: int | None = None) -> None:
    image_root, gt_file = wider_paths(split)
    annotations = list(parse_wider_face_gt(gt_file).items())
    if limit:
        annotations = annotations[:limit]

    images_out = YOLO_DATASET_DIR / "images" / split
    labels_out = YOLO_DATASET_DIR / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for rel_path, faces in tqdm(annotations, desc=f"prepare {split}"):
        if not faces:
            continue
        src_img = image_root / rel_path
        image = cv2.imread(str(src_img))
        if image is None:
            continue
        h_img, w_img = image.shape[:2]
        dst_img = images_out / rel_path
        dst_label = labels_out / Path(rel_path).with_suffix(".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)

        lines = []
        for face in faces:
            x, y, w, h = face["bbox"]
            lines.append(
                f"0 {(x + w / 2) / w_img:.6f} {(y + h / 2) / h_img:.6f} {w / w_img:.6f} {h / h_img:.6f}"
            )
        dst_label.write_text("\n".join(lines), encoding="utf-8")


def write_yaml() -> Path:
    yaml_path = YOLO_DATASET_DIR / "wider_face_yolo.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {YOLO_DATASET_DIR}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: face",
        ])
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def load_model(base: str, family: str):
    if family == "rtdetr":
        return RTDETR(base)
    if family == "yolo":
        return YOLO(base)
    raise ValueError("--family must be yolo or rtdetr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO or RT-DETR on WIDER FACE.")
    parser.add_argument("--family", choices=["yolo", "rtdetr"], default="rtdetr")
    parser.add_argument("--base", default="rtdetr-l.pt", help="Examples: face_yolov8m.pt, yolov8m.pt, rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--train-limit", type=int, default=600)
    parser.add_argument("--val-limit", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mosaic", type=float, default=0.6)
    parser.add_argument("--mixup", type=float, default=0.05)
    parser.add_argument("--vram-log-seconds", type=float, default=30.0, help="Minimum seconds between VRAM live log lines.")
    args = parser.parse_args()

    ensure_dirs()
    prepare_split("train", args.train_limit)
    prepare_split("val", args.val_limit)
    yaml_path = write_yaml()

    base_model = resolve_base_model(args.base)
    run_type = f"{args.family}_{Path(args.base).stem}_widerface_rocm"
    run_name = model_name(run_type, args.batch, args.epochs, "run").removesuffix(".run")
    output = MODEL_DIR / model_name(run_type, args.batch, args.epochs, "pt")

    print("VRAM before training:", vram_status())

    last_vram_log = 0.0

    def vram_callback(trainer):
        nonlocal last_vram_log
        now = time.monotonic()
        if now - last_vram_log >= args.vram_log_seconds:
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            batch_i = getattr(trainer, "batch_i", "?")
            print(f"{args.family.upper()} VRAM epoch {epoch}/{args.epochs} batch {batch_i}: {vram_status()}")
            last_vram_log = now

    model = load_model(base_model, args.family)
    model.add_callback("on_train_batch_end", vram_callback)
    train_kwargs = {
        "data": str(yaml_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "project": str(MODEL_DIR / f"{args.family}_runs"),
        "name": run_name,
        "exist_ok": True,
        "workers": args.workers,
        "seed": args.seed,
        "patience": 20,
    }
    if args.family == "yolo":
        train_kwargs.update({"mosaic": args.mosaic, "mixup": args.mixup, "cos_lr": True})
    model.train(**train_kwargs)

    best = MODEL_DIR / f"{args.family}_runs" / run_name / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, output)
        print(f"saved {output}")
    else:
        print(f"training finished but best.pt was not found at {best}")
    print("VRAM after training:", vram_status())


if __name__ == "__main__":
    main()
