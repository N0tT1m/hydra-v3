# Hydra V3

A distributed LLM inference platform that splits large language models across multiple GPUs/nodes for efficient inference. Built with Go (coordinator) and Python/PyTorch (GPU workers), using ZeroMQ for high-performance local network communication.

## Features

- **Distributed Inference**: Split models across heterogeneous GPUs (e.g., RTX 2080Ti 11GB + RTX 5090 32GB)
- **VRAM-Proportional Distribution**: Automatically allocates layers based on available GPU memory
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint
- **Streaming Support**: Server-Sent Events (SSE) for real-time token streaming
- **Pipeline Parallelism**: Double-buffered hidden state forwarding for compute/transfer overlap
- **Partial Model Loading**: Load only assigned layers from SafeTensors files
- **Multi-Architecture Support**: Llama, Mistral, Qwen, Phi models
- **Dynamic Cluster Management**: Nodes can join/leave with automatic rebalancing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Go Coordinator (:8080)                    │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ HTTP API     │  │ Model Mgmt  │  │ Cluster Mgmt      │   │
│  │ (OpenAI)     │  │ (HF Hub)    │  │ (Health, Rebal)   │   │
│  └──────────────┘  └─────────────┘  └───────────────────┘   │
│                           │                                  │
│  ┌────────────────────────┴─────────────────────────────┐   │
│  │              ZeroMQ Broker                            │   │
│  │  ROUTER:5555 (commands)  PULL:5556 (metrics)         │   │
│  │  PUB:5557 (broadcasts)                               │   │
│  └──────────────────────────┬───────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Python Worker │     │ Python Worker │     │ Python Worker │
│ (RTX 2080Ti)  │────▶│ (RTX 5090)    │────▶│ (RTX 5090)    │
│ Layers 0-4    │PUSH │ Layers 5-17   │PUSH │ Layers 18-31  │
│ 11GB VRAM     │PULL │ 32GB VRAM     │PULL │ 32GB VRAM     │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Prerequisites

### System Requirements

- Go 1.22+
- Python 3.10+
- CUDA 11.8+ (for NVIDIA GPUs) or MPS (for Apple Silicon)
- ZeroMQ library

### Install ZeroMQ

**macOS:**
```bash
brew install zeromq pkg-config
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y libzmq3-dev pkg-config
```

**RHEL/CentOS:**
```bash
sudo yum install -y zeromq-devel pkgconfig
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/hydra-v3.git
cd hydra-v3
```

### 2. Build the Coordinator

```bash
# Download Go dependencies
make deps

# Build the coordinator binary
make build
```

### 3. Install the Python Worker

```bash
cd worker

# Install in development mode
pip install -e .

# Or with training support (LoRA, QLoRA)
pip install -e ".[training]"

# Or with Flash Attention
pip install -e ".[flash]"
```

## Configuration

Copy the example configuration:

```bash
cp config.example.toml config.toml
```

### Configuration Options

```toml
[server]
http_addr = "0.0.0.0:8080"      # HTTP API address
metrics_addr = "0.0.0.0:9090"   # Prometheus metrics

[cluster]
node_id = "coordinator"
heartbeat_interval = "500ms"    # Worker heartbeat frequency
unhealthy_threshold = 3         # Missed heartbeats before unhealthy
reserved_vram_gb = 2.0          # VRAM reserved for KV cache/activations
memory_per_layer_gb = 0.5       # Estimated memory per transformer layer

[auth]
enabled = false                 # Enable API key authentication
api_keys = ["sk-xxx"]           # Valid API keys
rate_limit = 100                # Requests per window
rate_window = "1m"              # Rate limit window

[zmq]
router_addr = "tcp://*:5555"    # Command socket
metrics_addr = "tcp://*:5556"   # Metrics socket
broadcast_addr = "tcp://*:5557" # Broadcast socket
high_water_mark = 1000          # ZMQ message queue size

[model]
cache_dir = "~/.cache/hydra/models"
hf_token = ""                   # HuggingFace token for gated models
max_cache_gb = 100
```

## Usage

### Starting the Coordinator

```bash
# With config file
./build/bin/hydra -config config.toml

# Or with defaults
make run-dev
```

The coordinator will start and listen on:
- HTTP API: `http://localhost:8080`
- ZMQ Router: `tcp://*:5555`
- ZMQ Metrics: `tcp://*:5556`
- ZMQ Broadcast: `tcp://*:5557`

### Starting Workers

Each GPU should run one worker:

```bash
# Worker 1 (first GPU)
hydra-worker start \
  --node-id worker-1 \
  --coordinator tcp://localhost:5555 \
  --device cuda:0 \
  --pipeline-port 6000

# Worker 2 (second GPU)
hydra-worker start \
  --node-id worker-2 \
  --coordinator tcp://localhost:5555 \
  --device cuda:1 \
  --pipeline-port 6001
```

**Worker Options:**
- `--node-id`: Unique identifier for this worker
- `--coordinator`: Coordinator ZMQ address
- `--device`: GPU device (`auto`, `cuda:0`, `cuda:1`, `mps`, `cpu`)
- `--dtype`: Model precision (`float16`, `bfloat16`, `float32`)
- `--pipeline-port`: Port for hidden state forwarding

### Loading a Model

```bash
curl -X POST http://localhost:8080/api/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "meta-llama/Llama-2-7b-hf",
    "model_id": "llama-7b",
    "total_layers": 32
  }'
```

The coordinator will:
1. Calculate layer distribution based on worker VRAM
2. Broadcast topology to all workers
3. Send load commands with layer assignments
4. Workers load their assigned layers from SafeTensors

### Running Inference

**Chat Completion:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Streaming:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 200,
    "stream": true
  }'
```

## API Reference

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List loaded models |

### Cluster Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cluster/status` | GET | Cluster status and node info |
| `/api/cluster/rebalance` | POST | Trigger layer rebalancing |
| `/api/models/load` | POST | Load model across workers |
| `/api/models/unload` | POST | Unload model |
| `/api/models/hot-swap` | POST | Hot-swap model without downtime |

### Health & Metrics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check (workers available) |
| `/metrics` | GET | Prometheus metrics |

## Examples

### Python Client

```python
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        # Load model
        await client.post("http://localhost:8080/api/models/load", json={
            "model_path": "meta-llama/Llama-2-7b-hf",
            "model_id": "llama-7b",
            "total_layers": 32
        })

        # Chat completion
        resp = await client.post("http://localhost:8080/v1/chat/completions", json={
            "model": "llama-7b",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 100
        })
        print(resp.json()["choices"][0]["message"]["content"])

asyncio.run(main())
```

### Streaming with Python

```python
import httpx
import json

with httpx.stream("POST", "http://localhost:8080/v1/chat/completions", json={
    "model": "llama-7b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": True
}) as response:
    for line in response.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            content = chunk["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
```

### Using with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Unless auth is enabled
)

response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Layer Distribution Algorithm

The coordinator distributes layers proportionally based on available VRAM:

**Example: 32-layer model across 3 GPUs**
- GPU 1: 11GB VRAM (RTX 2080Ti)
- GPU 2: 32GB VRAM (RTX 5090)
- GPU 3: 32GB VRAM (RTX 5090)

**Calculation:**
1. Subtract reserved VRAM (2GB default): 9GB, 30GB, 30GB
2. Total effective: 69GB
3. Proportions: 13%, 43.5%, 43.5%
4. Layer allocation: 5, 13, 14 layers

**Result:**
- Worker 1: Layers 0-4 (embedding + first 5 layers)
- Worker 2: Layers 5-17 (middle 13 layers)
- Worker 3: Layers 18-31 (last 14 layers + lm_head)

## Project Structure

```
hydra-v3/
├── cmd/hydra/main.go              # Go coordinator entry point
├── internal/
│   ├── api/                       # HTTP API
│   │   ├── handlers/              # Request handlers
│   │   ├── middleware/            # Auth, rate limiting
│   │   └── types/                 # Request/response types
│   ├── coordinator/               # Core orchestration
│   │   ├── coordinator.go         # Message routing
│   │   ├── inference.go           # Generation management
│   │   ├── model_manager.go       # Model distribution
│   │   └── layer_distribution.go  # VRAM-aware allocation
│   ├── cluster/                   # Node registry
│   ├── zmq/                       # ZeroMQ broker
│   └── config/                    # Configuration
├── worker/                        # Python GPU worker
│   ├── hydra_worker/
│   │   ├── cli.py                 # CLI entry point
│   │   ├── core/                  # Device detection
│   │   ├── comm/                  # ZMQ, tensor protocol
│   │   ├── models/                # Partial model loading
│   │   ├── distributed/           # Pipeline, worker
│   │   └── inference/             # Generation engine
│   └── pyproject.toml
├── examples/                      # Usage examples
├── config.example.toml            # Example configuration
└── Makefile                       # Build commands
```

## Development

### Running Tests

```bash
# Go tests
make test

# Python tests
cd worker && pytest
```

### Code Formatting

```bash
# Go
make fmt

# Python
cd worker && black . && ruff check .
```

### Building Docker Images

```bash
make docker-build
```

## Troubleshooting

### "No healthy workers available"

Workers haven't registered yet. Check:
1. Workers are running and connecting to the correct coordinator address
2. ZMQ ports (5555-5557) are accessible
3. No firewall blocking connections

### "Failed to load model"

1. Ensure the model path is correct (HuggingFace ID or local path)
2. Check workers have enough combined VRAM
3. For gated models, set `hf_token` in config

### "Timeout waiting for logits"

The pipeline is stuck. Check:
1. All workers in the pipeline are healthy
2. Pipeline ports (6000+) are accessible between workers
3. No GPU memory errors on workers

### Worker crashes with OOM

Reduce layers per worker by:
1. Increasing `reserved_vram_gb` in config
2. Adding more workers to split the model further
3. Using a smaller model

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
