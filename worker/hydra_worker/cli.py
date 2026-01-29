"""CLI entry point for hydra worker."""

import asyncio
import click
import structlog
import signal
import sys

from hydra_worker.distributed.worker import DistributedWorker, DistributedWorkerConfig


def setup_logging(verbose: bool = False):
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """Hydra distributed worker CLI."""
    setup_logging(verbose)


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
    default="float16",
    type=click.Choice(["float16", "bfloat16", "float32"]),
    help="Data type for model weights (default: float16)",
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
