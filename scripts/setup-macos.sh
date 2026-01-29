#!/bin/bash
# Hydra V3 Setup Script for macOS
set -e

echo "=========================================="
echo "  Hydra V3 Setup - macOS"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"
brew install zeromq pkg-config go python@3.11

# Verify Go installation
if ! command -v go &> /dev/null; then
    echo -e "${RED}Go installation failed. Please install manually.${NC}"
    exit 1
fi
echo -e "${GREEN}Go version: $(go version)${NC}"

# Verify Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python installation failed. Please install manually.${NC}"
    exit 1
fi
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
elif python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}Apple MPS is available${NC}"
else
    echo -e "${YELLOW}No GPU acceleration detected. Will use CPU.${NC}"
fi

deactivate
cd "$PROJECT_ROOT"

# Create default config if not exists
if [ ! -f "config.toml" ]; then
    echo -e "${GREEN}Creating default configuration...${NC}"
    cp config.example.toml config.toml
fi

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
