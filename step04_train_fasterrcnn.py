# Trainiert Faster R-CNN auf WIDER FACE.
#
# Was beim Ausfuehren passiert:
# 1. Pfade und Ausgabeordner werden vorbereitet.
# 2. Die beste verfuegbare Hardware wird gewaehlt: CUDA/ROCm, Apple MPS, sonst CPU.
# 3. WIDER-FACE-Annotationen werden bei Bedarf in COCO-JSON umgewandelt.
# 4. Der Trainingsdatensatz wird geladen und optional mit --reduction verkleinert.
# 5. Ein COCO-vortrainiertes Faster R-CNN wird auf die Klasse "face" umgebaut.
# 6. Das Modell wird trainiert, Checkpoints und Trainingshistorie werden gespeichert.
#
# Wichtige Parameter: --epochs, --batch, --reduction, --lr, --workers,
# --save-every. Speichert .pth-Checkpoints in trained_models/.
# Namensschema: fasterrcnn_resnet50_fpn_rocm_bs<batch>_red<reduction>_ep<epochs>.pth.
# Beispiel:
#   python face_model_lab/step04_train_fasterrcnn.py --epochs 10 --batch 2 --reduction 1 --lr 0.0001 --save-every 1

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
import torchvision
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm

from step00_common import ANNOTATIONS_DIR, MODEL_DIR, RESULTS_DIR, ensure_dirs, model_name, rocm_device, timestamp, vram_status, wider_paths


class CocoFaceDataset(Dataset):
    """PyTorch-Dataset, das COCO-Annotationen und WIDER-FACE-Bilder zusammenfuehrt.

    Faster R-CNN erwartet pro Bild einen Tensor und ein Target-Dictionary mit
    Bounding Boxes im Format [x1, y1, x2, y2]. Diese Klasse liest die Bilder mit
    OpenCV, normalisiert Pixel auf 0..1 und baut genau dieses Target-Format.
    """

    def __init__(self, images_dir: Path, ann_file: Path):
        """Laedt die COCO-JSON und merkt sich alle Bild-IDs."""
        self.images_dir = images_dir
        self.coco = COCO(str(ann_file))
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self) -> int:
        """Gibt die Anzahl der Bilder im Dataset zurueck."""
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        """Laedt ein Bild und die zugehoerigen Gesichtsboxen fuer einen Index."""
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        img_path = self.images_dir / img_info["file_name"]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            areas.append(w * h)
            iscrowd.append(0)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
        return image, target


def collate_fn(batch):
    """Baut variable Detection-Batches fuer Torchvision.

    Detection-Modelle haben pro Bild unterschiedlich viele Boxen. Deshalb kann
    der Standard-Collate von PyTorch die Targets nicht sinnvoll stapeln.
    """
    return tuple(zip(*batch))


def build_model(num_classes: int = 2):
    """Erstellt Faster R-CNN und ersetzt den Klassifikationskopf.

    `weights="DEFAULT"` nutzt ein COCO-vortrainiertes Modell. Der letzte Head
    wird danach auf zwei Klassen angepasst: Hintergrund und Gesicht.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def write_history(path: Path, rows: list[dict]) -> None:
    """Schreibt die Trainingshistorie als CSV, damit Laeufe vergleichbar bleiben."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "epoch", "mean_loss", "lr", "batch", "reduction", "images", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)


def convert_wider_to_coco(gt_file: Path, output_json: Path) -> None:
    """Konvertiert WIDER-FACE-Textannotation in ein minimales COCO-JSON.

    Torchvision selbst braucht kein COCO-JSON, aber `pycocotools.COCO` macht das
    spaetere Laden der Annotationen robust und einheitlich. WIDER FACE enthaelt
    bei Bildern ohne Gesichter eine Dummy-Zeile; diese wird hier uebersprungen.
    """
    lines = gt_file.read_text(encoding="utf-8", errors="replace").splitlines()
    data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "face"}]}
    cursor = 0
    image_id = 1
    ann_id = 1
    while cursor < len(lines):
        rel_path = lines[cursor].strip()
        cursor += 1
        if not rel_path or cursor >= len(lines):
            continue
        face_count = int(lines[cursor].strip())
        cursor += 1
        data["images"].append({"id": image_id, "file_name": rel_path})
        if face_count == 0 and cursor < len(lines):
            parts = lines[cursor].strip().split()
            if len(parts) >= 4:
                try:
                    [float(value) for value in parts[:4]]
                    cursor += 1
                except ValueError:
                    pass
        for _ in range(face_count):
            parts = lines[cursor].strip().split()
            cursor += 1
            if len(parts) < 4:
                continue
            x, y, w, h = map(float, parts[:4])
            if w <= 0 or h <= 0:
                continue
            data["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1
        image_id += 1
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Parst CLI-Argumente und fuehrt den kompletten Trainingslauf aus."""
    parser = argparse.ArgumentParser(description="Train Faster R-CNN face detector on WIDER FACE.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--reduction", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-gpu", action="store_true", help="Fail instead of falling back to CPU if no GPU is visible.")
    parser.add_argument("--save-every", type=int, default=1, help="Save an epoch checkpoint every N epochs. Use 0 to disable.")
    args = parser.parse_args()

    # Ausgabeordner anlegen und automatisch die passende Hardware auswaehlen.
    ensure_dirs()
    device = rocm_device(args.require_gpu)

    # WIDER FACE liegt als Textdatei vor. Fuer dieses Training wird daraus beim
    # ersten Lauf eine wiederverwendbare COCO-JSON-Datei erzeugt.
    train_images, train_gt = wider_paths("train")
    ann_file = ANNOTATIONS_DIR / "instances_train.json"
    if not ann_file.exists():
        convert_wider_to_coco(train_gt, ann_file)

    # --reduction ist der wichtigste Hebel fuer schnelle Testlaeufe:
    # reduction=10 nimmt jedes zehnte Bild, reduction=1 nimmt alle Bilder.
    dataset = CocoFaceDataset(train_images, ann_file)
    if args.reduction > 1:
        dataset = torch.utils.data.Subset(dataset, list(range(0, len(dataset), args.reduction)))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    image_count = len(dataset)

    # Modell, Optimizer und Lernratenplan vorbereiten.
    model = build_model().to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    # Trainingsschleife: Forward Pass liefert Losses, Backward Pass aktualisiert
    # die Modellgewichte. tqdm zeigt Fortschritt, Loss und Speichernutzung.
    model.train()
    history = []
    history_path = RESULTS_DIR / f"training_history_fasterrcnn_resnet50_fpn_rocm_bs{args.batch}_red{args.reduction}_ep{args.epochs}_{timestamp()}.csv"
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Faster R-CNN epoch {epoch + 1}/{args.epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            losses = sum(loss for loss in model(images, targets).values())
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}", **vram_status(device)})
        scheduler.step()
        mean_loss = total_loss / len(loader)
        checkpoint = ""
        print(f"epoch={epoch + 1} mean_loss={mean_loss:.4f}")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            # Zwischenspeichern pro Epoche schuetzt lange Laeufe vor Datenverlust.
            checkpoint_path = MODEL_DIR / model_name("fasterrcnn_resnet50_fpn_rocm", args.batch, epoch + 1, "pth", args.reduction)
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint = str(checkpoint_path)
            print(f"checkpoint saved {checkpoint_path}")
        history.append({
            "model": "fasterrcnn_resnet50_fpn_rocm",
            "epoch": epoch + 1,
            "mean_loss": mean_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "batch": args.batch,
            "reduction": args.reduction,
            "images": image_count,
            "checkpoint": checkpoint,
        })
        write_history(history_path, history)
        print(f"history updated {history_path}")

    # Finaler Checkpoint mit dem Namensschema des gesamten Laufs.
    output = MODEL_DIR / model_name("fasterrcnn_resnet50_fpn_rocm", args.batch, args.epochs, "pth", args.reduction)
    torch.save(model.state_dict(), output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
