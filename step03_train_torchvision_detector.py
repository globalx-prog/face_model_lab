# Trainiert Torchvision-Detektoren auf WIDER FACE.
# Wichtige Parameter: --kind fasterrcnn|retinanet|fcos, --epochs, --batch,
# --reduction, --lr, --workers, --save-every.
# Speichert als <modelltyp>_bs<batch>_red<reduction>_ep<epochs>.pth.
# Beispiel:
#   python face_model_lab/step03_train_torchvision_detector.py --kind retinanet --epochs 10 --batch 2 --reduction 1 --save-every 1

from __future__ import annotations

import argparse
import csv

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from step00_common import ANNOTATIONS_DIR, MODEL_DIR, RESULTS_DIR, ensure_dirs, model_name, rocm_device, timestamp, vram_status, wider_paths
from step06_evaluate_models import build_fcos, build_fasterrcnn, build_retinanet
from step04_train_fasterrcnn import CocoFaceDataset, collate_fn, convert_wider_to_coco


def build_model(kind: str):
    if kind == "fasterrcnn":
        return build_fasterrcnn()
    if kind == "retinanet":
        return build_retinanet()
    if kind == "fcos":
        return build_fcos()
    raise ValueError("--kind must be fasterrcnn, retinanet, or fcos")


def write_history(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "epoch", "mean_loss", "lr", "batch", "reduction", "images", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Torchvision detector families on WIDER FACE.")
    parser.add_argument("--kind", choices=["fasterrcnn", "retinanet", "fcos"], default="retinanet")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--reduction", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1, help="Save an epoch checkpoint every N epochs. Use 0 to disable.")
    args = parser.parse_args()

    ensure_dirs()
    device = rocm_device(require_gpu=True)
    train_images, train_gt = wider_paths("train")
    ann_file = ANNOTATIONS_DIR / "instances_train.json"
    if not ann_file.exists():
        convert_wider_to_coco(train_gt, ann_file)

    dataset = CocoFaceDataset(train_images, ann_file)
    if args.reduction > 1:
        dataset = torch.utils.data.Subset(dataset, list(range(0, len(dataset), args.reduction)))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    image_count = len(dataset)

    model = build_model(args.kind).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    model.train()
    history = []
    history_path = RESULTS_DIR / f"training_history_{args.kind}_resnet50_fpn_rocm_bs{args.batch}_red{args.reduction}_ep{args.epochs}_{timestamp()}.csv"
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"{args.kind} epoch {epoch + 1}/{args.epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
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
            checkpoint_path = MODEL_DIR / model_name(f"{args.kind}_resnet50_fpn_rocm", args.batch, epoch + 1, "pth", args.reduction)
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint = str(checkpoint_path)
            print(f"checkpoint saved {checkpoint_path}")
        history.append({
            "model": f"{args.kind}_resnet50_fpn_rocm",
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

    output = MODEL_DIR / model_name(f"{args.kind}_resnet50_fpn_rocm", args.batch, args.epochs, "pth", args.reduction)
    torch.save(model.state_dict(), output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
