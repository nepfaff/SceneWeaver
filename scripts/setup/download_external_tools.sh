#!/bin/bash

download_external_tools() {
    local WORKSPACE_DIR="$1"

    print_header "Downloading External Tools"

    cd "${WORKSPACE_DIR}"

    # 1. Stable Diffusion 3.5
    print_info "Cloning SD 3.5 repository..."
    if [ -d "sd3.5" ]; then
        print_warning "sd3.5 already exists, skipping..."
    else
        git clone https://github.com/Scene-Weaver/sd3.5.git
        print_success "SD 3.5 cloned"
        print_info "Setting up SD 3.5..."
        cd sd3.5
        # Add setup instructions if needed
        cd ..
    fi

    # 2. Tabletop Digital Cousins
    print_info "Cloning Tabletop Digital Cousins repository..."
    if [ -d "Tabletop-Digital-Cousins" ]; then
        print_warning "Tabletop-Digital-Cousins already exists, skipping..."
    else
        git clone https://github.com/Scene-Weaver/Tabletop-Digital-Cousins.git
        print_success "Tabletop Digital Cousins cloned"
        print_warning "You need to set up the acdc2 conda environment separately"
        print_info "See: https://github.com/Scene-Weaver/Tabletop-Digital-Cousins for setup instructions"
    fi

    # 3. IDesign/OpenShape (Optional)
    print_info "Would you like to install IDesign/OpenShape for Objaverse retrieval? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        if [ -d "IDesign" ]; then
            print_warning "IDesign already exists, skipping..."
        else
            git clone https://github.com/atcelen/IDesign.git
            print_success "IDesign cloned"
            print_warning "You need to set up the idesign conda environment and download OpenShape embeddings"
            print_info "See: https://github.com/atcelen/IDesign for setup instructions"
        fi
    fi

    # 4. Holodeck (Optional)
    print_info "Would you like to install Holodeck for Objaverse retrieval? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        if [ -d "Holodeck" ]; then
            print_warning "Holodeck already exists, skipping..."
        else
            git clone https://github.com/allenai/Holodeck.git
            print_success "Holodeck cloned"
            print_warning "You need to download Holodeck data separately"
            print_info "See: https://github.com/allenai/Holodeck for data download instructions"
        fi
    fi

    print_success "External tools downloaded"
}
