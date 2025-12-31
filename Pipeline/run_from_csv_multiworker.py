#!/usr/bin/env python3
"""Run SceneWeaver scene generation with multiple parallel workers.

Automatically distributes incomplete scenes across workers.

Usage:
    # Run with 4 workers (default)
    python run_from_csv_multiworker.py

    # Run with 8 workers
    python run_from_csv_multiworker.py --num_workers 8

    # Only process specific ID range
    python run_from_csv_multiworker.py --start_id 0 --end_id 50
"""

import argparse
import csv
import os
import subprocess
import sys
from multiprocessing import Pool, current_process
from pathlib import Path

# Set SCENEWEAVER_DIR before any subprocess calls
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
os.environ["SCENEWEAVER_DIR"] = str(PROJECT_DIR)

CSV_FILE = str(Path.home() / "SceneEval/input/annotations.csv")
RESULTS_DIR = "~/efs/nicholas/scene-agent-eval-scenes/SceneWeaver/"


def is_scene_complete(results_dir: Path, scene_id: int) -> bool:
    """Check if a scene has finished processing (has render_14.jpg)."""
    render_file = results_dir / f"scene_{scene_id:03d}" / "record_scene" / "render_14.jpg"
    return render_file.exists()


def process_scene(args: tuple) -> tuple:
    """Process a single scene. Returns (scene_id, success)."""
    scene_id, description, results_dir = args
    worker_name = current_process().name

    save_dir = results_dir / f"scene_{scene_id:03d}"
    venv_python = PROJECT_DIR / ".venv" / "bin" / "python"

    print(f"[{worker_name}] Starting scene {scene_id}: {description[:50]}...")

    cmd = [
        str(venv_python),
        "main.py",
        "--prompt",
        description,
        "--save_dir",
        str(save_dir),
        "--cnt",
        "1",
    ]

    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[{worker_name}] Scene {scene_id} FAILED (code {result.returncode})")
        return (scene_id, False)

    print(f"[{worker_name}] Scene {scene_id} completed successfully")
    return (scene_id, True)


def main():
    parser = argparse.ArgumentParser(
        description="Run SceneWeaver with multiple parallel workers"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=CSV_FILE,
        help=f"Path to CSV file with prompts (default: {CSV_FILE})",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Directory to save results (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=None,
        help="Start from this ID (inclusive)",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="End at this ID (inclusive)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Process all scenes, even if already complete",
    )
    args = parser.parse_args()

    # Setup paths
    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = SCRIPT_DIR / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        prompts = list(reader)

    # Filter by ID range
    scenes = []
    for row in prompts:
        scene_id = int(row["ID"])
        if args.start_id is not None and scene_id < args.start_id:
            continue
        if args.end_id is not None and scene_id > args.end_id:
            continue
        scenes.append((scene_id, row["Description"]))

    # Filter out complete scenes
    if not args.no_skip_existing:
        incomplete = []
        for scene_id, desc in scenes:
            if not is_scene_complete(results_dir, scene_id):
                incomplete.append((scene_id, desc))

        skipped = len(scenes) - len(incomplete)
        scenes = incomplete
        print(f"Skipping {skipped} already-complete scenes")

    if not scenes:
        print("No scenes to process!")
        return

    print("=" * 60)
    print("SceneWeaver Multi-Worker Batch Generator")
    print("=" * 60)
    print(f"Workers: {args.num_workers}")
    print(f"Scenes to process: {len(scenes)}")
    print(f"Output: {results_dir}")
    print("=" * 60)

    # Prepare work items
    work_items = [(scene_id, desc, results_dir) for scene_id, desc in scenes]

    # Process in parallel
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_scene, work_items)

    # Summary
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    print("\n" + "=" * 60)
    print("Batch complete!")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
