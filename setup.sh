#!/bin/bash
# HSM Setup Script
# This script sets up the HSM environment with mamba or conda
#
# Usage:
#   ./setup.sh                    # Run
#   ./setup.sh --force           # Force recreation of environment

set -e  # Exit on any error

REPO_NAME="${REPO_NAME:-hsm}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1 \n"
}

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))

    printf "\r${BLUE}[PROGRESS]${NC} ["
    for ((i=1; i<=completed; i++)); do printf "="; done
    for ((i=completed+1; i<=width; i++)); do printf " "; done
    printf "] %d%% (%d/%d)" $percentage $current $total
}

# Setup step tracker
TOTAL_STEPS=7
CURRENT_STEP=0
DOWNLOAD_FAILED=false

next_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    show_progress $CURRENT_STEP $TOTAL_STEPS
    echo -e "\n${BLUE}[STEP $CURRENT_STEP/$TOTAL_STEPS]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate .env file configuration
validate_env_file() {
    ensure_project_root
    log_info "Validating .env file configuration..."

    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_error ".env file not found!"
        log_info "Please create a .env file based on .env.example:"
        log_info "  cp .env.example .env"
        log_info "  # Then edit .env with your API keys"
        return 1
    fi

    # Check OPENAI_API_KEY
    OPENAI_KEY=$(grep "^OPENAI_API_KEY=" .env | cut -d'=' -f2 | sed 's/^"//' | sed 's/"$//')
    if [ -z "$OPENAI_KEY" ] || [ "$OPENAI_KEY" = "your_openai_api_key_here" ]; then
        log_error "OPENAI_API_KEY is not set or using default placeholder!"
        log_info "Please set your actual OpenAI API key in .env file"
        log_info "Get your key from: https://platform.openai.com/api-keys"
        return 1
    fi

    # Check HF_TOKEN (required for downloads)
    HF_TOKEN_VALUE=$(grep "^HF_TOKEN=" .env | cut -d'=' -f2 | sed 's/^"//' | sed 's/"$//')
    if [ -z "$HF_TOKEN_VALUE" ] || [ "$HF_TOKEN_VALUE" = "your_huggingface_token_here" ]; then
        log_error "HF_TOKEN is not set or using default placeholder!"
        log_info "Please set your actual Hugging Face token in .env file"
        log_info "Get your token from: https://huggingface.co/settings/tokens"
        return 1
    fi

    log_success "Environment configuration validated successfully \n"
    return 0
}

# Check system requirements
check_requirements() {
    ensure_project_root
    log_info "Checking system requirements..."

    local missing_tools=()

    # Check for curl or wget
    if ! command_exists curl && ! command_exists wget; then
        missing_tools+=("curl or wget")
    fi

    # Check for git
    if ! command_exists git; then
        missing_tools+=("git")
    fi

    # Check for conda/mamba
    if ! command_exists conda && ! command_exists mamba; then
        missing_tools+=("conda or mamba")
    fi

    # Check for unzip
    if ! command_exists unzip; then
        missing_tools+=("unzip")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install them and run this script again."
        log_info "On Ubuntu/Debian: sudo apt-get install curl git unzip"
        log_info "For mamba (recommended - faster than conda):"
        log_info "  1. Install conda/miniconda first: https://docs.conda.io/projects/miniconda/"
        log_info "  2. Install mamba: conda install mamba -n base -c conda-forge"
        log_info "For conda (alternative): https://docs.conda.io/projects/conda/en/latest/user-guide/install/"
        exit 1
    fi

    log_success "All required tools are available \n"
}

activate_environment() {
    log_info "Activating conda environment..."

    # Prefer conda's shell hook for activation
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate hsm
    else
        # No conda on PATH: source conda.sh from the base
        CONDA_BASE="$(${CONDA_CMD:-mamba} info --base 2>/dev/null || echo "$HOME/miniconda3")"
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            . "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate hsm
        elif command -v micromamba >/dev/null 2>&1; then
            # Fallback if you're actually on micromamba
            eval "$(micromamba shell hook -s bash)"
            micromamba activate hsm
        else
            log_error "No activation hook found (conda.sh or micromamba). Run 'conda init bash' once or install conda."
            exit 1
        fi
    fi

    if [[ "$CONDA_DEFAULT_ENV" == "hsm" || "$CONDA_PREFIX" == *"/hsm" ]]; then
        log_success "Environment activated successfully! \n"
    else
        log_error "Activation failed, please follow the instructions in the README.md to manually setup the environment."
        exit 1
    fi
}

# Setup conda environment
setup_environment() {
    log_info "Setting up conda environment..."

    ensure_project_root

    # Detect conda/mamba
    if command_exists mamba; then
        CONDA_CMD="mamba"
    elif command_exists conda; then
        CONDA_CMD="conda"
    else
        log_error "Neither conda nor mamba found in PATH"
        exit 1
    fi

    # Check if environment already exists
    if $CONDA_CMD env list | awk '{print $1}' | grep -qx "hsm"; then
        log_info "Environment 'hsm' already exists"
        read -rp "Do you want to recreate it? (y/N): " REPLY
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing environment"
            activate_environment
            return 0
        fi
        log_info "Removing existing environment..."
        $CONDA_CMD env remove -n hsm --yes || {
            log_error "Failed to remove existing environment"
            exit 1
        }
    fi

    # Create environment
    log_info "Creating conda environment 'hsm'..."
    if $CONDA_CMD env create -f environment.yml --yes; then
        log_success "Environment created successfully \n"
    else
        log_error "Failed to create environment"
        exit 1
    fi

    activate_environment
}

# Download file with progress
download_file() {
    local url="$1"
    local output="$2"

    log_info "Downloading: $url"

    if command_exists wget; then
        wget --no-check-certificate -O "$output" "$url"
    elif command_exists curl; then
        curl -L -k -o "$output" "$url"
    else
        log_error "Neither curl nor wget available"
        exit 1
    fi
}

# Download HSSD models from Hugging Face
download_hssd_models() {
    log_info "Downloading HSSD models from Hugging Face..."

    mkdir -p data
    cd data

    # Store original directory
    ORIGINAL_DIR="$(pwd)"

    # Download HSSD models (this is a large download ~72GB)
    # Check if HSSD models are already downloaded (has .git directory)
    if [ ! -d "hssd-models/.git" ]; then
        log_info "Downloading HSSD models (~72GB)... This may take a while!"

        # Check if git-lfs is available
        if ! command_exists git-lfs; then
            log_warning "git-lfs not found. Installing..."
            if ! git lfs install; then
                log_error "Failed to install git-lfs"
                return 1
            fi
        fi

        # Login to HF using token from .env file
        if [ -f "../.env" ] && grep -q "HF_TOKEN=" ../.env; then
            HF_TOKEN=$(grep "HF_TOKEN=" ../.env | cut -d'=' -f2 | sed 's/^"//' | sed 's/"$//')
            if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "your_huggingface_token_here" ]; then
                log_info "Using Hugging Face token from .env file"
                if ! hf auth login --token "$HF_TOKEN"; then
                    log_error "Failed to login with HF token"
                    return 1
                fi
            else
                log_warning "HF_TOKEN not set in .env file, using interactive login"
                if ! hf auth login; then
                    log_error "Failed to login interactively"
                    return 1
                fi
            fi
        else
            log_warning ".env file not found or HF_TOKEN not set, using interactive login"
            if ! hf auth login; then
                log_error "Failed to login interactively"
                return 1
            fi
        fi

        # Clone the dataset using git (try SSH first, fallback to HTTPS)
        if ! git clone git@hf.co:datasets/hssd/hssd-models; then
            log_warning "SSH clone failed, trying HTTPS..."
            if ! git clone https://huggingface.co/datasets/hssd/hssd-models; then
                log_error "Failed to clone HSSD models repository with both SSH and HTTPS"
                log_info "Please check your internet connection and authentication setup"
                return 1
            fi
        fi

        log_success "HSSD models downloaded \n"
    else
        log_success "HSSD models repository already cloned \n"
    fi

    cd ..
}

# Helper function to ensure we're back in project root
ensure_project_root() {
    # Change to the directory where setup.sh is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
}

# Download decomposed models from Hugging Face
download_decomposed_models() {
    # Check if decomposed models already exist
    if [ -d "data/hssd-models/objects/decomposed" ]; then
        log_success "Decomposed models already downloaded \n"
        return 0
    fi
    
    log_info "Downloading decomposed models from Hugging Face..."
    mkdir -p data/hssd-models/objects

    # Prepare token argument if available
    HF_TOKEN_ARG=""
    if [ -f ".env" ] && grep -q "HF_TOKEN=" .env; then
        HF_TOKEN=$(grep "HF_TOKEN=" .env | cut -d'=' -f2 | sed 's/^"//' | sed 's/"$//')
        if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "your_huggingface_token_here" ]; then
            HF_TOKEN_ARG="--token $HF_TOKEN"
        fi
    fi

    # Download decomposed models
    if ! hf download hssd/hssd-hab \
        --repo-type=dataset \
        --include "objects/decomposed/**/*_part_*.glb" \
        --exclude "objects/decomposed/**/*_part.*.glb" \
        --local-dir "data/hssd-models" \
        $HF_TOKEN_ARG; then
        log_error "Failed to download decomposed models"
        return 1
    fi

    # Move decomposed folder to correct location
    if [ -d "data/hssd-models/objects/decomposed" ]; then
        log_success "Decomposed models downloaded \n"
    else
        log_warning "Decomposed models not found in expected location"
    fi
}

# Download data from GitHub releases
download_github_data() {
    # Check if data directory already has required contents
    if [ -d "data/motif_library" ] && [ -d "data/preprocessed" ]; then
        log_success "Core data already exists locally \n"
        return 0
    fi

    log_info "Processing data from GitHub releases..."
    mkdir -p data

    # Check if zip file already exists (look in multiple locations)
    local data_zip=""
    local found_local=false

    # First, check in project root
    if [ -f "data.zip" ]; then
        data_zip="data.zip"
        found_local=true
        log_success "Using local data.zip file from project root \n"
    # Then check inside data/ directory
    elif [ -f "data/data.zip" ]; then
        data_zip="data/data.zip"
        found_local=true
        log_success "Using local data.zip file from data/ directory \n"
    fi

    if [ "$found_local" = true ]; then
        # Check if the file is actually a valid zip
        if file "$data_zip" | grep -q "Zip archive"; then
            # Extract data
            log_info "Extracting data..."
            unzip -q "$data_zip"
            rm "$data_zip"
            log_success "GitHub data extracted successfully \n"
        else
            file_type=$(file "$data_zip" | head -1)
            log_error "Local data.zip is not a valid zip archive: $file_type"
            log_info "Please download a valid data.zip file from:"
            log_info "https://github.com/3dlg-hcvc/${REPO_NAME}/releases"
            return 1
        fi
    else
        log_info "No local data.zip found, downloading from GitHub..."
        local data_zip="data.zip"
        local data_url="https://github.com/3dlg-hcvc/${REPO_NAME}/releases/latest/download/data.zip"
        if ! download_file "$data_url" "$data_zip"; then
            log_warning "Failed to download data.zip. Please download manually:"
            log_info "wget --no-check-certificate -O data.zip '$data_url'"
            log_info "Then extract: unzip data.zip && rm data.zip"
            return 1
        fi

        # Check and extract the downloaded file
        if file "$data_zip" | grep -q "Zip archive"; then
            log_info "Extracting data..."
            unzip -q "$data_zip"
            rm "$data_zip"
            log_success "GitHub data extracted successfully \n"
        else
            file_type=$(file "$data_zip" | head -1)
            log_error "Downloaded data.zip is not a valid zip archive: $file_type"
            rm "$data_zip"
            return 1
        fi
    fi
}

# Download support surface data
download_support_surfaces() {
    # Check if support surface data already exists
    if [ -d "data/hssd-models/support-surfaces" ]; then
        log_success "Support surface data already exists locally \n"
        return 0
    fi

    log_info "Processing support surface data..."
    mkdir -p data/hssd-models

    # Check if zip file already exists (look in multiple locations)
    local support_zip=""
    local found_local=false

    # First, check in project root
    if [ -f "support-surfaces.zip" ]; then
        support_zip="support-surfaces.zip"
        found_local=true
        log_success "Using local support-surfaces.zip file from project root \n"
    # Then check inside data/ directory
    elif [ -f "data/support-surfaces.zip" ]; then
        support_zip="data/support-surfaces.zip"
        found_local=true
        log_success "Using local support-surfaces.zip file from data/ directory \n"
    fi

    if [ "$found_local" = true ]; then
        # Check if the file is actually a valid zip
        if file "$support_zip" | grep -q "Zip archive"; then
            # Extract to hssd-models
            log_info "Extracting support surfaces..."
            unzip -q "$support_zip" -d data/hssd-models
            rm "$support_zip"
            log_success "Support surface data extracted successfully \n"
        else
            file_type=$(file "$support_zip" | head -1)
            log_error "Local support-surfaces.zip is not a valid zip archive: $file_type"
            log_info "Please download a valid support-surfaces.zip file from:"
            log_info "https://github.com/3dlg-hcvc/${REPO_NAME}/releases"
            return 1
        fi
    else
        log_info "No local support-surfaces.zip found, downloading from GitHub..."
        local support_zip="support-surfaces.zip"
        local support_url="https://github.com/3dlg-hcvc/${REPO_NAME}/releases/latest/download/support-surfaces.zip"
        if ! download_file "$support_url" "$support_zip"; then
            log_warning "Failed to download support-surfaces.zip. Please download manually:"
            log_info "wget --no-check-certificate -O support-surfaces.zip '$support_url'"
            log_info "Then extract: unzip support-surfaces.zip -d data/hssd-models && rm support-surfaces.zip"
            return 1
        fi

        # Check and extract the downloaded file
        if file "$support_zip" | grep -q "Zip archive"; then
            log_info "Extracting support surfaces..."
            unzip -q "$support_zip" -d data/hssd-models
            rm "$support_zip"
            log_success "Support surface data extracted successfully \n"
        else
            file_type=$(file "$support_zip" | head -1)
            log_error "Downloaded support-surfaces.zip is not a valid zip archive: $file_type"
            rm "$support_zip"
            return 1
        fi
    fi
}

# Verify setup
verify_setup() {
    ensure_project_root
    log_info "Verifying setup..."

    local issues=0

    # Check conda environment
    if ! $CONDA_CMD env list | grep -q "^hsm "; then
        log_error "Conda environment 'hsm' not found"
        issues=$((issues + 1))
    else
        log_info "Conda environment 'hsm' exists"
    fi

    # Check data directory structure
    if [ ! -d "data" ]; then
        log_error "data directory not found"
        issues=$((issues + 1))
    else
        local required_paths=(
            "data/hssd-models/objects"
            "data/hssd-models/objects/decomposed"
            "data/hssd-models/support-surfaces"
            "data/motif_library/meta_programs"
            "data/preprocessed"
        )

        for path in "${required_paths[@]}"; do
            if [ ! -d "$path" ]; then
                log_error "Required path not found: $path"
                issues=$((issues + 1))
            fi
        done

        if [ $issues -eq 0 ]; then
            log_info "Data directory structure is correct"
        fi
    fi

    if [ $issues -gt 0 ]; then
        log_info "Setup verification found $issues issue(s)"
        return 1
    else
        log_success "Setup verification passed \n"
        return 0
    fi
}

# Main setup function
main() {
    echo -e "\n${BLUE}=======================================${NC}"
    log_info "Starting HSM setup..."
    log_info "========================================"
    echo ""

    # Parse command line arguments
    local force=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                force=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --force      Force recreation of environment"
                echo "  --help       Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Run setup steps
    next_step "Checking system requirements..."
    check_requirements

    next_step "Validating environment configuration..."
    ensure_project_root
    if ! validate_env_file; then
        log_error "Environment configuration validation failed!"
        log_info "Please fix the issues above and run the setup script again."
        exit 1
    fi

    next_step "Setting up conda environment..."
    ensure_project_root
    setup_environment

    next_step "Downloading HSSD models (~72GB)..."
    log_info "HSM by default uses HSSD models for 3D model retrieval."
    ensure_project_root
    if ! download_hssd_models; then
        DOWNLOAD_FAILED=true
    fi
    ensure_project_root
    if ! download_decomposed_models; then
        DOWNLOAD_FAILED=true
    fi

    next_step "Downloading preprocessed data from GitHub..."
    ensure_project_root
    if ! download_github_data; then
        log_warning "Preprocessed data download failed. Please download it manually to the root directory then run setup.sh again."
        DOWNLOAD_FAILED=true
    fi

    next_step "Downloading support surface data..."
    ensure_project_root
    if ! download_support_surfaces; then
        log_warning "Support surface download failed. Please download it manually to the root directory then run setup.sh again."
        DOWNLOAD_FAILED=true
    fi

    next_step "Verifying setup..."
    ensure_project_root
    if verify_setup; then
        show_progress $TOTAL_STEPS $TOTAL_STEPS
        echo -e "\n"

        if [ "$DOWNLOAD_FAILED" = true ]; then
            log_warning "HSM setup failed with download failures!"
            log_info ""
            log_info "Some downloads failed. Please download the missing data manually as above."
            log_info "Then run the setup script again."
            log_info "If problem persists, follow the manual setup instructions in the README.md."
            log_info ""
            exit 1
        else
            log_success "HSM setup completed successfully!"
            log_info ""
            log_info "To generate a scene with a description using HSM, run the following commands:"
            log_info "1. $CONDA_CMD activate hsm"
            log_info "2. python main.py -d 'your description'"
            log_info "The default output directory is results/single_run"
            log_info "For more details, please refer to the README.md."
            log_info ""
        fi
    else
        echo -e "\n"
        log_warning "Setup failed with issues. Please review the errors above."
        exit 1
    fi
}

# Run main function
main "$@"
