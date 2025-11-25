#!/bin/bash

download_datasets() {
    local WORKSPACE_DIR="$1"

    print_header "Downloading Datasets"

    cd "${WORKSPACE_DIR}"
    mkdir -p datasets

    # 1. 3D FUTURE Dataset
    print_info "Downloading 3D FUTURE dataset from HuggingFace..."

    if [ -n "${EXISTING_3D_FUTURE}" ] && [ -d "${EXISTING_3D_FUTURE}" ]; then
        print_info "Using existing 3D FUTURE dataset at: ${EXISTING_3D_FUTURE}"
        ln -sf "${EXISTING_3D_FUTURE}" "${WORKSPACE_DIR}/datasets/3D-FUTURE"
        print_success "Linked to existing 3D FUTURE dataset"
    elif [ -d "${WORKSPACE_DIR}/datasets/3D-FUTURE" ]; then
        print_warning "3D FUTURE dataset already exists, skipping..."
    else
        print_info "Downloading 3D-FUTURE-model.zip (this may take a while)..."
        mkdir -p "${WORKSPACE_DIR}/datasets/3D-FUTURE"
        cd "${WORKSPACE_DIR}/datasets"

        # Download from HuggingFace
        wget https://huggingface.co/datasets/yangyandan/PhyScene/resolve/main/dataset/3D-FUTURE-model.zip

        print_info "Extracting 3D FUTURE dataset..."
        unzip -q 3D-FUTURE-model.zip -d 3D-FUTURE/
        rm 3D-FUTURE-model.zip

        print_success "3D FUTURE dataset downloaded and extracted"
    fi

    # 2. MetaScenes Dataset (Optional - requires manual download)
    print_info "MetaScenes dataset needs to be obtained separately"

    if [ -n "${EXISTING_METASCENES}" ] && [ -d "${EXISTING_METASCENES}" ]; then
        print_info "Using existing MetaScenes dataset at: ${EXISTING_METASCENES}"
        ln -sf "${EXISTING_METASCENES}" "${WORKSPACE_DIR}/datasets/metascenes"
        print_success "Linked to existing MetaScenes dataset"
    else
        print_warning "MetaScenes dataset not found"
        print_info "If you have MetaScenes, set EXISTING_METASCENES=/path/to/metascenes and re-run"
        print_info "Or contact the authors to obtain the dataset"
    fi

    # 3. PhyScene sample data (already in repo)
    print_success "PhyScene sample data already present in data/physcene/"

    # 4. Objaverse (downloaded on-demand)
    print_info "Objaverse assets will be downloaded on-demand during scene generation"

    if [ -n "${EXISTING_OBJAVERSE}" ] && [ -d "${EXISTING_OBJAVERSE}" ]; then
        print_info "Using existing Objaverse cache at: ${EXISTING_OBJAVERSE}"
        mkdir -p "${HOME}/.objaverse"
        ln -sf "${EXISTING_OBJAVERSE}" "${HOME}/.objaverse/hf-objaverse-v1"
        print_success "Linked to existing Objaverse cache"
    else
        print_info "Objaverse will cache to: ~/.objaverse/hf-objaverse-v1/"
    fi

    print_success "Dataset download complete"
}
