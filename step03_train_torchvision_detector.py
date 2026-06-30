# Trainiert Torchvision-Detektoren auf WIDER FACE.
# Wichtige Parameter: --kind fasterrcnn|fasterrcnn_mobile|retinanet|fcos, --epochs, --batch,
# --reduction, --lr, --workers, --save-every, --amp, --min-size, --max-size,
# --resume-from, --start-epoch.
# Speichert als <modelltyp>_bs<batch>_red<reduction>_ep<epochs>.pth.
# Beispiel:
#   python face_model_lab/step03_train_torchvision_detector.py --kind retinanet --epochs 10 --batch 2 --reduction 1 --save-every 1

from __future__ import annotations

import argparse
import csv
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from step00_common import ANNOTATIONS_DIR, MODEL_DIR, RESULTS_DIR, ensure_dirs, model_name, rocm_device, timestamp, vram_status, wider_paths
from step06_evaluate_models import build_fasterrcnn, build_fasterrcnn_mobile, build_fcos, build_retinanet
from step04_train_fasterrcnn import CocoFaceDataset, collate_fn, convert_wider_to_coco


def build_model(kind: str, min_size: int | None = None, max_size: int | None = None):
    if kind == "fasterrcnn":
        return build_fasterrcnn(min_size=min_size, max_size=max_size)
    if kind == "fasterrcnn_mobile":
        return build_fasterrcnn_mobile(min_size=min_size, max_size=max_size)
    if kind == "retinanet":
        return build_retinanet(min_size=min_size, max_size=max_size)
    if kind == "fcos":
        return build_fcos(min_size=min_size, max_size=max_size)
    raise ValueError("--kind must be fasterrcnn, fasterrcnn_mobile, retinanet, or fcos")


def write_history(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "epoch", "mean_loss", "lr", "batch", "reduction", "images", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Torchvision detector families on WIDER FACE.")
    parser.add_argument("--kind", choices=["fasterrcnn", "fasterrcnn_mobile", "retinanet", "fcos"], default="retinanet")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--reduction", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1, help="Save an epoch checkpoint every N epochs. Use 0 to disable.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA/ROCm to reduce memory and often improve throughput.")
    parser.add_argument("--min-size", type=int, default=None, help="Torchvision detector resize min_size. Default keeps torchvision's model default.")
    parser.add_argument("--max-size", type=int, default=None, help="Torchvision detector resize max_size. Default keeps torchvision's model default.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0.")
    parser.add_argument("--resume-from", type=str, default=None, help="Load model weights from this .pth checkpoint before training.")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch number already completed by --resume-from. New checkpoints continue from this number.")
    args = parser.parse_args()

    ensure_dirs()
    device = rocm_device(require_gpu=False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    train_images, train_gt = wider_paths("train")
    ann_file = ANNOTATIONS_DIR / "instances_train.json"
    if not ann_file.exists():
        convert_wider_to_coco(train_gt, ann_file)

    dataset = CocoFaceDataset(train_images, ann_file)
    if args.reduction > 1:
        dataset = torch.utils.data.Subset(dataset, list(range(0, len(dataset), args.reduction)))
    loader_kwargs = {
        "batch_size": args.batch,
        "shuffle": True,
        "num_workers": args.workers,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    image_count = len(dataset)

    model = build_model(args.kind, min_size=args.min_size, max_size=args.max_size).to(device)
    if args.resume_from:
        resume_path = str(args.resume_from)
        print(f"Loading checkpoint: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
        print(f"Resuming from epoch {args.start_epoch}; training {args.epochs} additional epochs.")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(
        "Training config:",
        f"kind={args.kind}",
        f"images={image_count}",
        f"batches/epoch={len(loader)}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"amp={amp_enabled}",
        f"min_size={args.min_size or 'default'}",
        f"max_size={args.max_size or 'default'}",
        f"resume_from={args.resume_from or 'none'}",
        f"start_epoch={args.start_epoch}",
    )

    model.train()
    history = []
    model_label = f"{args.kind}_resnet50_fpn_rocm" if args.kind != "fasterrcnn_mobile" else "fasterrcnn_mobile_fpn_rocm"
    final_epoch = args.start_epoch + args.epochs
    history_path = RESULTS_DIR / f"training_history_{model_label}_bs{args.batch}_red{args.reduction}_ep{final_epoch}_{timestamp()}.csv"
    for epoch_offset in range(args.epochs):
        global_epoch = args.start_epoch + epoch_offset + 1
        total_loss = 0.0
        epoch_t0 = time.perf_counter()
        pbar = tqdm(loader, desc=f"{args.kind} epoch {global_epoch}/{final_epoch}")
        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}", **vram_status(device)})
        scheduler.step()
        mean_loss = total_loss / len(loader)
        epoch_seconds = time.perf_counter() - epoch_t0
        checkpoint = ""
        print(f"epoch={global_epoch} mean_loss={mean_loss:.4f} seconds={epoch_seconds:.1f} sec_per_batch={epoch_seconds / len(loader):.3f}")
        if args.save_every and global_epoch % args.save_every == 0:
            checkpoint_path = MODEL_DIR / model_name(model_label, args.batch, global_epoch, "pth", args.reduction)
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint = str(checkpoint_path)
            print(f"checkpoint saved {checkpoint_path}")
        history.append({
            "model": model_label,
            "epoch": global_epoch,
            "mean_loss": mean_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "batch": args.batch,
            "reduction": args.reduction,
            "images": image_count,
            "checkpoint": checkpoint,
        })
        write_history(history_path, history)
        print(f"history updated {history_path}")

    output = MODEL_DIR / model_name(model_label, args.batch, final_epoch, "pth", args.reduction)
    torch.save(model.state_dict(), output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
