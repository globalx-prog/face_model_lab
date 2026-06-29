from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def parse_rocm_smi_csv(text: str) -> dict[str, float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    reader = csv.DictReader(lines)
    row = next(reader, None)
    if not row:
        return {}

    def value(name: str, default: float = 0.0) -> float:
        raw = (row.get(name) or "").strip()
        try:
            return float(raw)
        except ValueError:
            return default

    return {
        "edge_c": value("Temperature (Sensor edge) (C)"),
        "junction_c": value("Temperature (Sensor junction) (C)"),
        "memory_c": value("Temperature (Sensor memory) (C)"),
        "power_w": value("Average Graphics Package Power (W)"),
        "gpu_use_pct": value("GPU use (%)"),
        "vram_pct": value("GPU Memory Allocated (VRAM%)"),
    }


def read_rocm_metrics() -> dict[str, float]:
    result = subprocess.run(
        ["rocm-smi", "--showtemp", "--showpower", "--showuse", "--showmemuse", "--csv"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return {}
    return parse_rocm_smi_csv(result.stdout)


def write_row(path: Path, row: dict[str, float | str]) -> None:
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "elapsed_s", "edge_c", "junction_c", "memory_c", "power_w", "gpu_use_pct", "vram_pct"],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command while monitoring ROCm temperatures.")
    parser.add_argument("--max-junction", type=float, default=95.0, help="Terminate the process at or above this junction temperature.")
    parser.add_argument("--max-edge", type=float, default=90.0, help="Terminate the process at or above this edge temperature.")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between ROCm readings.")
    parser.add_argument("--log", type=Path, required=True, help="CSV file for temperature and utilization readings.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    process = subprocess.Popen(command)
    start = time.time()
    max_seen = {"edge_c": 0.0, "junction_c": 0.0, "memory_c": 0.0}
    exit_code = 0

    try:
        while process.poll() is None:
            metrics = read_rocm_metrics()
            elapsed = time.time() - start
            if metrics:
                for key in max_seen:
                    max_seen[key] = max(max_seen[key], metrics.get(key, 0.0))
                row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_s": f"{elapsed:.1f}",
                    **{key: f"{metrics.get(key, 0.0):.1f}" for key in ["edge_c", "junction_c", "memory_c", "power_w", "gpu_use_pct", "vram_pct"]},
                }
                write_row(args.log, row)
                print(
                    "rocm_guard",
                    f"elapsed={elapsed / 60:.1f}min",
                    f"edge={metrics.get('edge_c', 0.0):.1f}C",
                    f"junction={metrics.get('junction_c', 0.0):.1f}C",
                    f"memory={metrics.get('memory_c', 0.0):.1f}C",
                    f"power={metrics.get('power_w', 0.0):.1f}W",
                    flush=True,
                )
                if metrics.get("junction_c", 0.0) >= args.max_junction or metrics.get("edge_c", 0.0) >= args.max_edge:
                    print("rocm_guard temperature limit reached; terminating training.", flush=True)
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return 75
            time.sleep(args.interval)
        exit_code = process.returncode or 0
    except KeyboardInterrupt:
        process.terminate()
        raise
    finally:
        print(
            "rocm_guard summary",
            f"max_edge={max_seen['edge_c']:.1f}C",
            f"max_junction={max_seen['junction_c']:.1f}C",
            f"max_memory={max_seen['memory_c']:.1f}C",
            f"log={args.log}",
            flush=True,
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
