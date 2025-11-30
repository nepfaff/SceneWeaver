#!/usr/bin/env python3
"""Run SceneWeaver scene generation from CSV prompts.

Usage:
    # Run all prompts
    python run_from_csv.py

    # Run prompts 0-2 (inclusive)
    python run_from_csv.py --start_id 0 --end_id 2

    # Run specific prompts by ID
    python run_from_csv.py --indices "0,2,4"

    # Custom output directory
    python run_from_csv.py --results_dir ./my_output --indices "1,3"
"""

import argparse
import csv
import subprocess
import os
from pathlib import Path

CSV_FILE = "/home/ubuntu/SceneWeaver/prompts.csv"
RESULTS_DIR = "./output"


def main():
    parser = argparse.ArgumentParser(
        description="Run SceneWeaver scene generation from CSV prompts"
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
        "--indices",
        type=str,
        default=None,
        help='Comma-separated list of IDs to process (e.g., "0,2,5")',
    )
    args = parser.parse_args()

    # Parse indices if provided
    selected_ids = None
    if args.indices:
        selected_ids = set(int(x.strip()) for x in args.indices.split(","))

    # Read CSV
    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        prompts = list(reader)

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = script_dir / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Count scenes to process
    scenes_to_process = []
    for row in prompts:
        prompt_id = int(row["ID"])
        if args.start_id is not None and prompt_id < args.start_id:
            continue
        if args.end_id is not None and prompt_id > args.end_id:
            continue
        if selected_ids is not None and prompt_id not in selected_ids:
            continue
        scenes_to_process.append((prompt_id, row["Description"]))

    total = len(scenes_to_process)
    print("=" * 60)
    print("SceneWeaver CSV Batch Generator")
    print(f"Processing {total} scenes...")
    print("=" * 60)

    for i, (prompt_id, description) in enumerate(scenes_to_process):
        save_dir = results_dir / f"scene_{prompt_id:03d}"

        print(f"\n[{i+1}/{total}] Scene {prompt_id}: {description[:50]}...")
        print("-" * 60)

        # Use venv Python to ensure correct environment
        venv_python = project_dir / ".venv" / "bin" / "python"

        # Run main.py with explicit save_dir
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

        result = subprocess.run(cmd, cwd=script_dir)
        if result.returncode != 0:
            print(f"Warning: Scene {prompt_id} failed with code {result.returncode}")

    print("\n" + "=" * 60)
    print(f"Batch complete! {total} scenes processed.")
    print(f"Output saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
