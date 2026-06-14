from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO, RTDETR

from common import MODEL_DIR, YOLO_DATASET_DIR, ensure_dirs, model_name, parse_wider_face_gt, vram_status, wider_paths


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
    args = parser.parse_args()

    ensure_dirs()
    prepare_split("train", args.train_limit)
    prepare_split("val", args.val_limit)
    yaml_path = write_yaml()

    run_type = f"{args.family}_{Path(args.base).stem}_widerface_rocm"
    run_name = model_name(run_type, args.batch, args.epochs, "run").removesuffix(".run")
    output = MODEL_DIR / model_name(run_type, args.batch, args.epochs, "pt")

    print("VRAM before training:", vram_status())

    def vram_callback(trainer):
        batch_i = getattr(trainer, "batch_i", 0)
        if batch_i % 10 == 0:
            print(f"{args.family.upper()} VRAM batch {batch_i}: {vram_status()}")

    model = load_model(args.base, args.family)
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
