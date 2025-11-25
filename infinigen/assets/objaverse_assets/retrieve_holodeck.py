#!/usr/bin/env python
"""
Holodeck-based Objaverse asset retrieval.

This script retrieves 3D assets from a pre-downloaded objathor/layoutvlm dataset
by matching category names using text similarity.

Prerequisites:
- Download layoutvlm-objathor data from SceneEval or use prepare_objathor.py
- Set OBJATHOR_ASSETS_DIR environment variable to the data directory

Usage:
    python retrieve_holodeck.py <save_dir>
"""

import gzip
import json
import os
import re
import sys
from pathlib import Path

# Try different possible data locations
POSSIBLE_DATA_DIRS = [
    os.environ.get("OBJATHOR_ASSETS_DIR", ""),
    os.path.expanduser("~/SceneEval/_data/layoutvlm-objathor"),
    os.path.expanduser("~/.objathor-assets/objathor-assets"),
    os.path.expanduser("~/workspace/Holodeck/data/2023_09_23/assets"),
    "/home/ubuntu/SceneEval/_data/layoutvlm-objathor",
]


def find_data_dir():
    """Find the first existing data directory."""
    for path in POSSIBLE_DATA_DIRS:
        if path and os.path.isdir(path):
            return path
    return None


def load_annotations(data_dir):
    """Load category annotations from all assets."""
    annotations = {}
    data_path = Path(data_dir)

    for asset_dir in data_path.iterdir():
        if not asset_dir.is_dir():
            continue

        asset_id = asset_dir.name

        # Check for GLB file
        glb_file = asset_dir / f"{asset_id}.glb"
        if not glb_file.exists():
            continue

        # Try to load data.json (layoutvlm format)
        data_json = asset_dir / "data.json"
        if data_json.exists():
            try:
                with open(data_json, "r") as f:
                    data = json.load(f)
                    if "annotations" in data and "category" in data["annotations"]:
                        annotations[asset_id] = {
                            "category": data["annotations"]["category"].lower(),
                            "description": data["annotations"].get("description", ""),
                            "glb_path": str(glb_file),
                        }
            except Exception as e:
                print(f"Warning: Failed to load {data_json}: {e}")
                continue

        # Try to load annotations.json.gz (objathor format)
        elif (asset_dir / "annotations.json.gz").exists():
            try:
                with gzip.open(asset_dir / "annotations.json.gz", "rt") as f:
                    data = json.load(f)
                    if "category" in data:
                        annotations[asset_id] = {
                            "category": data["category"].lower(),
                            "description": data.get("description", ""),
                            "glb_path": str(glb_file),
                        }
            except Exception as e:
                print(f"Warning: Failed to load annotations from {asset_dir}: {e}")
                continue

    return annotations


def normalize_category(name):
    """Normalize a category name for matching."""
    # Remove numbers and underscores
    name = re.sub(r"\d+", "", name)
    name = name.replace("_", " ")
    # Remove extra whitespace
    name = " ".join(name.split())
    return name.lower().strip()


def find_matching_assets(query_category, annotations, top_k=5):
    """Find assets matching the query category using simple text matching."""
    query = normalize_category(query_category)
    query_words = set(query.split())

    matches = []
    for asset_id, info in annotations.items():
        category = info["category"]
        description = info.get("description", "").lower()

        # Exact category match
        if category == query:
            matches.append((asset_id, info["glb_path"], 1.0))
            continue

        # Category contains query
        if query in category:
            matches.append((asset_id, info["glb_path"], 0.9))
            continue

        # Query contains category
        if category in query:
            matches.append((asset_id, info["glb_path"], 0.8))
            continue

        # Word overlap score
        category_words = set(category.split())
        common_words = query_words & category_words
        if common_words:
            score = len(common_words) / max(len(query_words), len(category_words))
            if score > 0.3:
                matches.append((asset_id, info["glb_path"], score * 0.7))
                continue

        # Check description
        if query in description:
            matches.append((asset_id, info["glb_path"], 0.5))

    # Sort by score and return top k
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:top_k]


def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieve_holodeck.py <save_dir>")
        sys.exit(1)

    save_dir = sys.argv[1]

    # Find data directory
    data_dir = find_data_dir()
    if not data_dir:
        print("Error: Could not find objathor/layoutvlm data directory.")
        print("Please set OBJATHOR_ASSETS_DIR environment variable or download the data.")
        print("Checked locations:", POSSIBLE_DATA_DIRS)
        sys.exit(1)

    print(f"Using data directory: {data_dir}")

    # Load categories to retrieve
    objav_cnts_file = f"{save_dir}/objav_cnts.json"
    if not os.path.exists(objav_cnts_file):
        print(f"Error: {objav_cnts_file} not found")
        sys.exit(1)

    with open(objav_cnts_file, "r") as f:
        objav_cnts = json.load(f)

    print(f"Categories to retrieve: {list(objav_cnts.keys())}")

    # Load all annotations
    print("Loading asset annotations...")
    annotations = load_annotations(data_dir)
    print(f"Loaded {len(annotations)} assets")

    if not annotations:
        print("Error: No assets found in data directory")
        sys.exit(1)

    # Find matching assets for each category
    result = {}
    for category in objav_cnts.keys():
        print(f"\nSearching for: {category}")
        matches = find_matching_assets(category, annotations)

        if matches:
            # Return GLB file paths
            result[category] = [m[1] for m in matches]
            print(f"  Found {len(matches)} matches: {[m[0] for m in matches[:3]]}")
        else:
            result[category] = []
            print(f"  No matches found")

    # Write results
    output_file = f"{save_dir}/objav_files.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
