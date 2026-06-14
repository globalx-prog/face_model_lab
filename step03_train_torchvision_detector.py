from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from step00_common import ANNOTATIONS_DIR, MODEL_DIR, ensure_dirs, model_name, rocm_device, vram_status, wider_paths
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

    model = build_model(args.kind).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    model.train()
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
        print(f"epoch={epoch + 1} mean_loss={total_loss / len(loader):.4f}")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            checkpoint = MODEL_DIR / model_name(f"{args.kind}_resnet50_fpn_rocm", args.batch, epoch + 1, "pth")
            torch.save(model.state_dict(), checkpoint)
            print(f"checkpoint saved {checkpoint}")

    output = MODEL_DIR / model_name(f"{args.kind}_resnet50_fpn_rocm", args.batch, args.epochs, "pth")
    torch.save(model.state_dict(), output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
