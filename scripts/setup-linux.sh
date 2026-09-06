#!/bin/bash
# Hydra V3 Setup Script for Linux (Ubuntu/Debian)
set -e

echo "=========================================="
echo "  Hydra V3 Setup - Linux"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
else
    echo -e "${RED}Unsupported package manager. Please install dependencies manually.${NC}"
    exit 1
fi

echo -e "${GREEN}Detected package manager: $PKG_MANAGER${NC}"

# Install system dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"

case $PKG_MANAGER in
    apt)
        sudo apt-get update
        sudo apt-get install -y build-essential libzmq3-dev pkg-config python3 python3-pip python3-venv golang-go curl
        ;;
    dnf)
        sudo dnf install -y gcc gcc-c++ zeromq-devel pkgconfig python3 python3-pip golang curl
        ;;
    yum)
        sudo yum install -y gcc gcc-c++ zeromq-devel pkgconfig python3 python3-pip golang curl
        ;;
    pacman)
        sudo pacman -Syu --noconfirm base-devel zeromq pkgconf python python-pip go curl
        ;;
esac

# Check Go version and install newer if needed
GO_VERSION=$(go version 2>/dev/null | grep -oP 'go\K[0-9]+\.[0-9]+' || echo "0.0")
REQUIRED_GO="1.22"

if [ "$(printf '%s\n' "$REQUIRED_GO" "$GO_VERSION" | sort -V | head -n1)" != "$REQUIRED_GO" ]; then
    echo -e "${YELLOW}Go version $GO_VERSION is older than required $REQUIRED_GO. Installing newer Go...${NC}"

    # Install Go from official source
    GO_LATEST="1.22.0"
    curl -LO "https://go.dev/dl/go${GO_LATEST}.linux-amd64.tar.gz"
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "go${GO_LATEST}.linux-amd64.tar.gz"
    rm "go${GO_LATEST}.linux-amd64.tar.gz"

    export PATH=$PATH:/usr/local/go/bin
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
fi

echo -e "${GREEN}Go version: $(go version)${NC}"
echo -e "${GREEN}Python version: $(python3 --version)${NC}"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Build Go coordinator
echo -e "${GREEN}Building Go coordinator...${NC}"
go mod download
go mod tidy
mkdir -p build/bin
go build -o build/bin/hydra ./cmd/hydra

if [ -f "build/bin/hydra" ]; then
    echo -e "${GREEN}Coordinator built successfully: build/bin/hydra${NC}"
else
    echo -e "${RED}Coordinator build failed${NC}"
    exit 1
fi

# Setup Python virtual environment
echo -e "${GREEN}Setting up Python virtual environment...${NC}"
cd worker

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install worker package
echo -e "${GREEN}Installing Python worker...${NC}"
pip install -e .

# Check for CUDA
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}CUDA is available${NC}"

    # Show GPU info
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || true
else
    echo -e "${YELLOW}CUDA not detected. Install PyTorch with CUDA support for GPU acceleration:${NC}"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu118"
fi

deactivate
cd "$PROJECT_ROOT"

# Create default config if not exists
if [ ! -f "config.toml" ]; then
    echo -e "${GREEN}Creating default configuration...${NC}"
    cp config.example.toml config.toml
fi

# Make run scripts executable
chmod +x scripts/*.sh

echo ""
echo -e "${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To start the coordinator:"
echo "  ./scripts/run-coordinator.sh"
echo ""
echo "To start a worker:"
echo "  ./scripts/run-worker.sh --node-id worker-1"
echo ""
echo "Or manually:"
echo "  ./build/bin/hydra -config config.toml"
echo "  cd worker && source venv/bin/activate && hydra-worker start --node-id worker-1"
echo ""
