#!/bin/bash
# SDXL Dependencies Setup (uv-only, no conda)
# This script sets up Stable Diffusion XL for image generation.
# SDXL does NOT require HuggingFace license acceptance (unlike SD 3.5)

set -e

# Get script directory and SceneWeaver root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENEWEAVER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() { echo -e "\n${BLUE}======== $1 ========${NC}\n"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

# Load .env if exists
if [ -f "${SCENEWEAVER_DIR}/.env" ]; then
    set -a
    source "${SCENEWEAVER_DIR}/.env"
    set +a
fi

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-${HOME}/workspace}"
SDXL_DIR="${SDXL_DIR:-${WORKSPACE_DIR}/sdxl}"

print_header "SDXL Dependencies Setup (uv)"
print_info "SDXL directory: ${SDXL_DIR}"
print_info "NOTE: SDXL does NOT require HuggingFace license acceptance!"

# Check for uv
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_success "uv found"

# 1. Create or verify SDXL directory exists
print_header "Setting Up SDXL Directory"
mkdir -p "${SDXL_DIR}"

# 2. Create uv venv for SDXL
print_header "Creating SDXL Virtual Environment"
cd "${SDXL_DIR}"

if [ -d ".venv" ]; then
    print_warning "SDXL .venv exists"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        uv venv .venv --python 3.10
        print_success "Recreated SDXL venv"
    else
        print_info "Using existing venv"
    fi
else
    uv venv .venv --python 3.10
    print_success "Created SDXL venv"
fi

# Activate venv for subsequent installs
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.1
print_header "Installing PyTorch (CUDA 12.1)"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
print_success "PyTorch installed"

# 4. Install diffusers and dependencies
print_header "Installing Diffusers and Dependencies"
uv pip install diffusers accelerate transformers
print_success "Diffusers installed"

# 5. Create inference script
print_header "Creating Inference Script"
cat > "${SDXL_DIR}/sdxl_infer.py" << 'INFER_EOF'
#!/usr/bin/env python
"""
SDXL Image Generation Script
No gated access required - downloads model automatically from HuggingFace

Usage:
    python sdxl_infer.py --prompt "A kitchen countertop with objects" --out output.png
"""
import argparse
import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# Cache directory for models
CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_pipeline(use_fp16=True, low_vram=False):
    """Load SDXL pipeline with optimizations."""
    dtype = torch.float16 if use_fp16 else torch.float32

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=CACHE_DIR,
    )

    # Use DPM++ scheduler for better quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if low_vram:
        # Enable memory efficient attention
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    pipe = pipe.to("cuda")

    print("SDXL pipeline loaded!")
    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = None,
):
    """Generate an image from prompt."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"Generating image with prompt: {prompt[:100]}...")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return image


def main():
    parser = argparse.ArgumentParser(description="SDXL Image Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted", help="Negative prompt")
    parser.add_argument("--out", type=str, default="output.png", help="Output image path")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--low_vram", action="store_true", help="Enable memory optimizations for low VRAM GPUs")

    args = parser.parse_args()

    # Load pipeline
    pipe = load_pipeline(use_fp16=True, low_vram=args.low_vram)

    # Generate image
    image = generate_image(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
    )

    # Save image
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    image.save(args.out)
    print(f"Image saved to: {args.out}")


if __name__ == "__main__":
    main()
INFER_EOF

print_success "Created sdxl_infer.py"

# 6. Create run.sh script for ACDC integration
print_header "Creating Run Script"
cat > "${SDXL_DIR}/run.sh" << 'RUN_EOF'
#!/bin/bash
# Run SDXL inference from prompt.json
# Used by SceneWeaver ACDC tool

SDXL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate venv
if [ -d "${SDXL_DIR}/.venv" ]; then
    source "${SDXL_DIR}/.venv/bin/activate"
else
    echo "Error: SDXL venv not found at ${SDXL_DIR}/.venv"
    echo "Run install_sdxl_dependencies.sh first"
    exit 1
fi

# Read prompt from prompt.json
if [ -f "${SDXL_DIR}/prompt.json" ]; then
    PROMPT=$(python -c "import json; f=open('${SDXL_DIR}/prompt.json'); d=json.load(f); print(d.get('prompt', ''))")
    OUTPUT=$(python -c "import json; f=open('${SDXL_DIR}/prompt.json'); d=json.load(f); print(d.get('img_savedir', 'output.png'))")

    # Run inference
    cd "${SDXL_DIR}"
    python sdxl_infer.py \
        --prompt "${PROMPT}" \
        --out "${OUTPUT}" \
        --steps 30 \
        --cfg 7.5
else
    echo "Error: prompt.json not found"
    exit 1
fi
RUN_EOF

chmod +x "${SDXL_DIR}/run.sh"
print_success "Created run.sh"

# 7. Create activation script
print_header "Creating Activation Script"
cat > "${SDXL_DIR}/activate_sdxl.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Activate SDXL environment
# Usage: source /path/to/sdxl/activate_sdxl.sh

SDXL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "${SDXL_DIR}/.venv" ]; then
    echo "Error: SDXL venv not found at ${SDXL_DIR}/.venv"
    echo "Run install_sdxl_dependencies.sh first"
    return 1
fi

source "${SDXL_DIR}/.venv/bin/activate"

echo "SDXL environment activated"
echo "  SDXL_DIR: ${SDXL_DIR}"
ACTIVATE_EOF

chmod +x "${SDXL_DIR}/activate_sdxl.sh"
print_success "Created activate_sdxl.sh"

deactivate

# Final summary
print_header "SDXL Setup Complete!"
echo ""
echo "Installation summary:"
echo "  SDXL directory:        ${SDXL_DIR}"
echo "  Virtual environment:    ${SDXL_DIR}/.venv"
echo ""
echo "To use SDXL manually:"
echo "  source ${SDXL_DIR}/activate_sdxl.sh"
echo "  python sdxl_infer.py --prompt 'your prompt here' --out output.png"
echo ""
echo "To verify installation:"
echo "  source ${SDXL_DIR}/activate_sdxl.sh"
echo "  python -c 'from diffusers import StableDiffusionXLPipeline; print(\"SDXL ready!\")'"
echo ""
print_success "SDXL is ready to use - no HuggingFace license required!"
