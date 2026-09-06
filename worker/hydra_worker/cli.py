"""CLI entry point for hydra worker."""

import asyncio
import logging
import os
import platform
import socket
import click
import structlog
import signal
import sys

# Fix Windows asyncio + ZMQ compatibility
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from hydra_worker.distributed.worker import DistributedWorker, DistributedWorkerConfig

# den-den-mushi app tag for this process. The Go coordinator tags itself
# "robin-hydra"; the worker gets its own tag so the two are separable in the hub.
DEFAULT_APP = "robin-hydra-worker"


def _denden_processor(app: str, host: str):
    """structlog processor: stamp den-den-mushi fields and normalize keys.

    Produces the hub schema — {ts, level, app, host, msg, ...} — matching the
    den-den-mushi Python shim: `event` -> `msg`, `exception` -> `error`, and
    the "warning" level renamed to "warn".
    """

    def process(logger, method_name, event_dict):
        event_dict.setdefault("app", app)
        event_dict.setdefault("host", host)
        if event_dict.get("level") == "warning":
            event_dict["level"] = "warn"
        if "event" in event_dict:
            event_dict["msg"] = event_dict.pop("event")
        if "exception" in event_dict:
            event_dict["error"] = event_dict.pop("exception")
        return event_dict

    return process


def setup_logging(verbose: bool = False, fmt: str = None, app: str = None, stream=None):
    """Configure structured logging.

    Default ("json") emits den-den-mushi lines to stdout so a Dockerized deploy
    is captured by the hub forwarder with no extra wiring. "console" gives
    colorized human output for local dev. HYDRA_LOG_FORMAT / HYDRA_LOG_APP
    override the args.
    """
    fmt = (fmt or os.environ.get("HYDRA_LOG_FORMAT") or "json").lower()
    app = app or os.environ.get("HYDRA_LOG_APP") or DEFAULT_APP
    host = os.environ.get("DENDEN_HOST") or socket.gethostname()
    stream = stream if stream is not None else sys.stdout

    common = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "console":
        processors = [
            structlog.processors.TimeStamper(fmt="iso"),
            *common,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = [
            structlog.processors.TimeStamper(fmt="iso", key="ts"),
            *common,
            _denden_processor(app, host),
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if verbose else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=False,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--log-format",
    type=click.Choice(["json", "console"]),
    default=None,
    help="Log format: json (den-den-mushi, default) or console (dev). "
    "Overrides HYDRA_LOG_FORMAT.",
)
def cli(verbose: bool, log_format: str):
    """Hydra distributed worker CLI."""
    setup_logging(verbose, fmt=log_format)


@cli.command()
@click.option(
    "--node-id",
    "-n",
    required=True,
    help="Unique identifier for this worker node",
)
@click.option(
    "--coordinator",
    "-c",
    default="tcp://localhost:5555",
    help="Coordinator address (default: tcp://localhost:5555)",
)
@click.option(
    "--device",
    "-d",
    default="auto",
    help="Device to use: auto, cuda:0, cuda:1, mps, cpu (default: auto)",
)
@click.option(
    "--dtype",
    default="bfloat16",
    type=click.Choice(["float16", "bfloat16", "float32", "int8", "int4", "fp8", "q8", "q4"]),
    help="Data type for model weights (default: bfloat16). Use int8/int4/q8/q4 for quantization.",
)
@click.option(
    "--pipeline-port",
    "-p",
    default=6000,
    type=int,
    help="Port for pipeline communication (default: 6000)",
)
def start(
    node_id: str,
    coordinator: str,
    device: str,
    dtype: str,
    pipeline_port: int,
):
    """Start the distributed worker.

    Example:
        hydra-worker start --node-id worker-1 --coordinator tcp://192.168.1.100:5555
    """
    # Normalize dtype aliases
    dtype_map = {"q8": "int8", "q4": "int4"}
    dtype = dtype_map.get(dtype, dtype)

    log = structlog.get_logger()
    log.info(
        "Starting worker",
        node_id=node_id,
        coordinator=coordinator,
        device=device,
        dtype=dtype,
        pipeline_port=pipeline_port,
    )

    config = DistributedWorkerConfig(
        node_id=node_id,
        coordinator_addr=coordinator,
        device=device,
        dtype=dtype,
        pipeline_port=pipeline_port,
    )

    worker = DistributedWorker(config)

    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        log.info("Received shutdown signal")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(worker.start())
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        worker.stop()
    finally:
        loop.close()


@cli.command()
@click.option(
    "--device",
    "-d",
    default="auto",
    help="Device to check (default: auto)",
)
def info(device: str):
    """Display device information.

    Example:
        hydra-worker info --device cuda:0
    """
    from hydra_worker.core.device import detect_device

    info = detect_device(device)

    click.echo(f"Device Type: {info.device_type}")
    click.echo(f"Device Index: {info.device_index}")
    click.echo(f"Name: {info.name}")
    click.echo(f"Total Memory: {info.total_memory / (1024**3):.2f} GB")
    click.echo(f"Free Memory: {info.free_memory / (1024**3):.2f} GB")

    if info.compute_capability:
        click.echo(f"Compute Capability: {info.compute_capability[0]}.{info.compute_capability[1]}")


@cli.command()
@click.argument("model_path")
@click.option(
    "--layer-start",
    "-s",
    default=0,
    type=int,
    help="First layer index (default: 0)",
)
@click.option(
    "--layer-end",
    "-e",
    default=8,
    type=int,
    help="Last layer index (default: 8)",
)
@click.option(
    "--device",
    "-d",
    default="auto",
    help="Device to use (default: auto)",
)
def test_load(model_path: str, layer_start: int, layer_end: int, device: str):
    """Test partial model loading.

    Example:
        hydra-worker test-load meta-llama/Llama-2-7b-hf --layer-start 0 --layer-end 8
    """
    import torch
    from hydra_worker.core.device import detect_device
    from hydra_worker.models.partial_loader import PartialModelLoader

    device_info = detect_device(device)
    device_obj = torch.device(
        f"{device_info.device_type}:{device_info.device_index}"
        if device_info.device_type != "cpu"
        else "cpu"
    )
    dtype = torch.float16 if device_info.device_type in ("cuda", "mps") else torch.float32

    click.echo(f"Device: {device_obj}")
    click.echo(f"Model: {model_path}")
    click.echo(f"Layers: {layer_start} to {layer_end}")

    loader = PartialModelLoader(model_path, device_obj, dtype)

    click.echo(f"\nModel config:")
    click.echo(f"  Architecture: {loader.arch}")
    click.echo(f"  Total layers: {loader.config.num_hidden_layers}")
    click.echo(f"  Hidden size: {loader.config.hidden_size}")

    mem_bytes = loader.estimate_memory(layer_start, layer_end)
    click.echo(f"  Estimated memory: {mem_bytes / 1024**3:.2f} GB")

    click.echo(f"\nLoading layers {layer_start} to {layer_end}...")

    is_first = layer_start == 0
    is_last = layer_end == loader.config.num_hidden_layers

    model, tokenizer = loader.load_partial_model(
        layer_start=layer_start,
        layer_end=layer_end,
        include_embedding=is_first,
        include_lm_head=is_last,
    )

    click.echo(f"  Loaded {len(model.layers)} layers")
    click.echo(f"  Has embedding: {model.has_embedding}")
    click.echo(f"  Has lm_head: {model.has_lm_head}")
    click.echo("\nPartial loading test passed!")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
