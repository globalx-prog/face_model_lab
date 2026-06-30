"""Train Faster R-CNN on the full WIDER FACE train split after the red6 run.

Default flow:
1. Wait until the red6 ep25 checkpoint exists.
2. Resume the same Faster R-CNN ResNet50-FPN model.
3. Train 5 additional epochs with reduction=1, batch=2, GPU required.

This script is intentionally a thin launcher around
step03_train_torchvision_detector.py so the actual training behavior stays in
one place.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "face_model_lab" / "step03_train_torchvision_detector.py"
DEFAULT_RESUME = ROOT / "trained_models" / "fasterrcnn_resnet50_fpn_rocm_bs2_red6_ep25.pth"
DEFAULT_LOG = ROOT / "model_results" / "full_fasterrcnn_red1_ep25_to_ep30.log"


def wait_for_stable_file(path: Path, poll_seconds: int, stable_seconds: int) -> None:
    """Wait until a checkpoint exists and its size is stable."""
    print(f"Waiting for checkpoint: {path}", flush=True)
    while not path.exists():
        time.sleep(poll_seconds)

    previous_size = -1
    while True:
        current_size = path.stat().st_size
        if current_size > 0 and current_size == previous_size:
            print(f"Checkpoint ready: {path} ({current_size} bytes)", flush=True)
            return
        previous_size = current_size
        time.sleep(stable_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue Faster R-CNN from red6 ep25 on the full dataset.")
    parser.add_argument("--resume-from", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--start-epoch", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=5, help="Additional full-dataset epochs.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--stable-seconds", type=int, default=30)
    parser.add_argument("--no-wait", action="store_true", help="Fail immediately if --resume-from does not exist.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    resume_path = args.resume_from if args.resume_from.is_absolute() else ROOT / args.resume_from
    log_path = args.log if args.log.is_absolute() else ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.no_wait:
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
    else:
        wait_for_stable_file(resume_path, args.poll_seconds, args.stable_seconds)

    final_epoch = args.start_epoch + args.epochs
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--kind",
        "fasterrcnn",
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--reduction",
        "1",
        "--lr",
        str(args.lr),
        "--workers",
        str(args.workers),
        "--save-every",
        str(args.save_every),
        "--resume-from",
        str(resume_path),
        "--start-epoch",
        str(args.start_epoch),
        "--min-size",
        str(args.min_size),
        "--max-size",
        str(args.max_size),
        "--require-gpu",
    ]

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n=== Faster R-CNN full dataset continuation ===\n")
        log.write(f"resume_from={resume_path}\n")
        log.write(f"full_dataset_epochs={args.epochs}\n")
        log.write(f"target_checkpoint=trained_models/fasterrcnn_resnet50_fpn_rocm_bs{args.batch}_red1_ep{final_epoch}.pth\n")
        log.write("command=" + " ".join(command) + "\n\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
