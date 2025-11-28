#!/usr/bin/env python
"""
Download SD 3.5 models from HuggingFace.

REQUIREMENTS:
1. Accept the license at: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
2. Login to HuggingFace: huggingface-cli login

Usage:
    python download_models.py [--model medium|large|large-turbo]
"""
import os
import sys
import argparse

def check_access():
    """Check if user has access to the gated model."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
        return True
    except Exception as e:
        print(f"Not logged in to HuggingFace: {e}")
        print("\nPlease run: huggingface-cli login")
        return False

def download_models(model_variant="medium"):
    """Download SD 3.5 models."""
    from huggingface_hub import hf_hub_download, snapshot_download

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_map = {
        "medium": "stabilityai/stable-diffusion-3.5-medium",
        "large": "stabilityai/stable-diffusion-3.5-large",
        "large-turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
    }

    model_files = {
        "medium": "sd3.5_medium.safetensors",
        "large": "sd3.5_large.safetensors",
        "large-turbo": "sd3.5_large_turbo.safetensors",
    }

    repo_id = model_map.get(model_variant, model_map["medium"])
    model_file = model_files.get(model_variant, model_files["medium"])

    print(f"\nDownloading SD 3.5 {model_variant} model...")
    print(f"  Repository: {repo_id}")
    print(f"  Target: {models_dir}/")
    print()

    try:
        # Download main model
        print(f"1. Downloading {model_file}...")
        hf_hub_download(repo_id, model_file, local_dir=models_dir)
        print("   Done!")

        # Download text encoders
        print("2. Downloading CLIP-L encoder...")
        hf_hub_download(repo_id, "text_encoders/clip_l.safetensors", local_dir=models_dir)
        print("   Done!")

        print("3. Downloading CLIP-G encoder...")
        hf_hub_download(repo_id, "text_encoders/clip_g.safetensors", local_dir=models_dir)
        print("   Done!")

        print("4. Downloading T5-XXL encoder...")
        hf_hub_download(repo_id, "text_encoders/t5xxl_fp16.safetensors", local_dir=models_dir)
        print("   Done!")

        # Create symlinks for sd3_infer.py compatibility
        # The inference script expects files directly in models/, not in text_encoders/
        print("5. Creating symlinks for inference compatibility...")
        symlinks = {
            "t5xxl.safetensors": "text_encoders/t5xxl_fp16.safetensors",
            "clip_l.safetensors": "text_encoders/clip_l.safetensors",
            "clip_g.safetensors": "text_encoders/clip_g.safetensors",
        }
        for link_name, target in symlinks.items():
            link_path = os.path.join(models_dir, link_name)
            if os.path.islink(link_path):
                os.remove(link_path)
            elif os.path.exists(link_path):
                continue  # Don't overwrite actual files
            os.symlink(target, link_path)
        print("   Done!")

        print("\nAll models downloaded successfully!")
        print(f"Models are in: {models_dir}")

    except Exception as e:
        if "403" in str(e) or "GatedRepoError" in str(type(e).__name__):
            print(f"\nERROR: Access denied to {repo_id}")
            print("\nTo fix this:")
            print(f"  1. Go to: https://huggingface.co/{repo_id}")
            print("  2. Click 'Agree and access repository' to accept the license")
            print("  3. Run this script again")
            return False
        else:
            print(f"\nERROR: {e}")
            return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Download SD 3.5 models")
    parser.add_argument("--model", choices=["medium", "large", "large-turbo"],
                       default="medium", help="Model variant to download")
    args = parser.parse_args()

    print("=" * 60)
    print("Stable Diffusion 3.5 Model Downloader")
    print("=" * 60)

    if not check_access():
        return 1

    if not download_models(args.model):
        return 1

    print("\n" + "=" * 60)
    print("Setup complete! You can now use SD 3.5 for image generation.")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
