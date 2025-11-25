#!/bin/bash

setup_environments() {
    local SCENEWEAVER_DIR="$1"

    print_header "Setting Up Python Environments"

    # 1. SceneWeaver Planner Environment
    print_info "Creating sceneweaver conda environment..."
    if conda env list | grep -q "^sceneweaver "; then
        print_warning "sceneweaver environment already exists, skipping..."
    else
        conda env create -f "${SCENEWEAVER_DIR}/environment_sceneweaver.yml"
        print_success "sceneweaver environment created"
    fi

    # 2. Infinigen Executor Environment
    print_info "Setting up infinigen environment with uv..."

    cd "${SCENEWEAVER_DIR}"

    # Initialize git submodules
    print_info "Initializing git submodules..."
    if [ -d "infinigen/OcMesher/.git" ]; then
        print_success "Submodules already initialized"
    else
        git submodule update --init --recursive
        print_success "Git submodules initialized"
    fi

    # Create .python-version if it doesn't exist
    if [ ! -f ".python-version" ]; then
        echo "3.10" > .python-version
        print_success "Created .python-version file"
    fi

    # Install Python dependencies with uv
    print_info "Installing dependencies with uv sync..."
    if [ -d ".venv" ]; then
        print_warning ".venv already exists"
        read -p "Remove and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf .venv
            uv sync
            print_success "Dependencies installed with uv"
        else
            print_warning "Skipping dependency installation"
        fi
    else
        uv sync
        print_success "Dependencies installed with uv"
    fi

    # Install Infinigen with Blender
    print_info "Installing Infinigen with Blender..."
    if [ -d "${SCENEWEAVER_DIR}/blender" ]; then
        print_warning "Blender already installed, skipping..."
    else
        INFINIGEN_MINIMAL_INSTALL=True bash "${SCENEWEAVER_DIR}/scripts/install/interactive_blender.sh"
        print_success "Infinigen installed with Blender"
    fi

    print_success "Environment setup complete!"
    echo ""
    print_info "Available environments:"
    echo "  - sceneweaver: Planner environment (Python 3.8)"
    echo "  - .venv: Infinigen executor environment (Python 3.10)"
}
