from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from step00_common import MODEL_DIR, YOLO_DATASET_DIR, ensure_dirs, model_name, parse_wider_face_gt, vram_status, wider_paths


def prepare_split(split: str, limit: int | None = None) -> None:
    image_root, gt_file = wider_paths(split)
    annotations = list(parse_wider_face_gt(gt_file).items())
    if limit:
        annotations = annotations[:limit]

    images_out = YOLO_DATASET_DIR / "images" / split
    labels_out = YOLO_DATASET_DIR / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for rel_path, faces in tqdm(annotations, desc=f"prepare YOLO {split}"):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO face detector on WIDER FACE.")
    parser.add_argument("--base", default="face_yolov8m.pt", help="Base model, e.g. face_yolov8m.pt or yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--train-limit", type=int, default=600)
    parser.add_argument("--val-limit", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mosaic", type=float, default=0.6)
    parser.add_argument("--mixup", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    prepare_split("train", args.train_limit)
    prepare_split("val", args.val_limit)
    yaml_path = write_yaml()

    run_type = Path(args.base).stem.replace("-", "").replace("_", "") + "_widerface_rocm"
    run_name = model_name(run_type, args.batch, args.epochs, "run").removesuffix(".run")
    output = MODEL_DIR / model_name(run_type, args.batch, args.epochs, "pt")

    print("VRAM before training:", vram_status())

    def vram_callback(trainer):
        batch_i = getattr(trainer, "batch_i", 0)
        if batch_i % 10 == 0:
            print(f"YOLO VRAM batch {batch_i}: {vram_status()}")

    model = YOLO(args.base)
    model.add_callback("on_train_batch_end", vram_callback)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(MODEL_DIR / "yolo_runs"),
        name=run_name,
        exist_ok=True,
        workers=args.workers,
        seed=args.seed,
        mosaic=args.mosaic,
        mixup=args.mixup,
        cos_lr=True,
        patience=20,
    )

    best = MODEL_DIR / "yolo_runs" / run_name / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, output)
        print(f"saved {output}")
    else:
        print(f"training finished but best.pt was not found at {best}")
    print("VRAM after training:", vram_status())


if __name__ == "__main__":
    main()
