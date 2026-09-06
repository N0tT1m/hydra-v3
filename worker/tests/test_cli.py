"""CLI surface: option parsing, device reporting, and worker startup wiring.

The logging processors are covered by test_logging.py; this file covers the
click commands themselves — the layer that turns operator input into a
DistributedWorkerConfig.
"""

import pytest
from click.testing import CliRunner

import hydra_worker.cli as cli_module
from hydra_worker.cli import cli
from hydra_worker.core.device import DeviceInfo


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def captured_worker(monkeypatch):
    """Replace DistributedWorker so `start` never opens a socket."""
    captured = {}

    class FakeWorker:
        def __init__(self, config):
            captured["config"] = config
            self.stopped = False

        async def start(self):
            captured["started"] = True

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(cli_module, "DistributedWorker", FakeWorker)
    return captured


# --- start ------------------------------------------------------------------


def test_start_requires_a_node_id(runner, captured_worker):
    result = runner.invoke(cli, ["start"])
    assert result.exit_code != 0
    assert "node-id" in result.output


def test_start_builds_a_config_from_defaults(runner, captured_worker):
    result = runner.invoke(cli, ["start", "--node-id", "worker-1"])

    assert result.exit_code == 0, result.output
    config = captured_worker["config"]
    assert config.node_id == "worker-1"
    assert config.coordinator_addr == "tcp://localhost:5555"
    assert config.device == "auto"
    assert config.dtype == "bfloat16"
    assert config.pipeline_port == 6000
    assert captured_worker["started"] is True


def test_start_passes_through_explicit_options(runner, captured_worker):
    result = runner.invoke(
        cli,
        [
            "start",
            "--node-id", "gpu-2",
            "--coordinator", "tcp://10.0.0.1:5555",
            "--device", "cuda:1",
            "--dtype", "float16",
            "--pipeline-port", "6010",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured_worker["config"]
    assert config.node_id == "gpu-2"
    assert config.coordinator_addr == "tcp://10.0.0.1:5555"
    assert config.device == "cuda:1"
    assert config.dtype == "float16"
    assert config.pipeline_port == 6010


@pytest.mark.parametrize("alias,expected", [("q8", "int8"), ("q4", "int4")])
def test_start_normalizes_quantization_aliases(runner, captured_worker, alias, expected):
    result = runner.invoke(cli, ["start", "--node-id", "w", "--dtype", alias])

    assert result.exit_code == 0, result.output
    assert captured_worker["config"].dtype == expected


def test_start_rejects_an_unknown_dtype(runner, captured_worker):
    result = runner.invoke(cli, ["start", "--node-id", "w", "--dtype", "float64"])
    assert result.exit_code != 0
    assert "float64" in result.output


def test_start_stops_the_worker_on_keyboard_interrupt(runner, monkeypatch):
    stopped = []

    class InterruptingWorker:
        def __init__(self, config):
            pass

        async def start(self):
            raise KeyboardInterrupt

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(cli_module, "DistributedWorker", InterruptingWorker)

    result = runner.invoke(cli, ["start", "--node-id", "w"])

    assert result.exit_code == 0, result.output
    assert stopped == [True], "Ctrl-C must release the worker's sockets"


def test_verbose_flag_is_accepted(runner, captured_worker):
    result = runner.invoke(cli, ["--verbose", "start", "--node-id", "w"])
    assert result.exit_code == 0, result.output


def test_log_format_option_is_accepted(runner, captured_worker):
    result = runner.invoke(cli, ["--log-format", "console", "start", "--node-id", "w"])
    assert result.exit_code == 0, result.output


# --- info -------------------------------------------------------------------


def fake_device(**kwargs):
    defaults = dict(
        device_type="cuda",
        device_index=0,
        name="Fake GPU",
        total_memory=16 * 1024**3,
        free_memory=8 * 1024**3,
        compute_capability=(8, 6),
    )
    defaults.update(kwargs)
    return DeviceInfo(**defaults)


def test_info_reports_device_details(runner, monkeypatch):
    monkeypatch.setattr(
        "hydra_worker.core.device.detect_device", lambda device: fake_device()
    )

    result = runner.invoke(cli, ["info", "--device", "cuda:0"])

    assert result.exit_code == 0, result.output
    assert "Fake GPU" in result.output
    assert "16.00 GB" in result.output
    assert "8.00 GB" in result.output
    assert "8.6" in result.output


def test_info_omits_compute_capability_when_unavailable(runner, monkeypatch):
    monkeypatch.setattr(
        "hydra_worker.core.device.detect_device",
        lambda device: fake_device(device_type="cpu", name="CPU", compute_capability=None),
    )

    result = runner.invoke(cli, ["info"])

    assert result.exit_code == 0, result.output
    assert "CPU" in result.output
    assert "Compute Capability" not in result.output


# --- test-load --------------------------------------------------------------


class FakeConfig:
    num_hidden_layers = 32
    hidden_size = 4096


class FakePartialModel:
    def __init__(self, layer_count, has_embedding, has_lm_head):
        self.layers = list(range(layer_count))
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head


class FakeLoader:
    """Stands in for PartialModelLoader; records what it was asked to load."""

    last = None

    def __init__(self, model_path, device, dtype):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.arch = "llama"
        self.config = FakeConfig()
        FakeLoader.last = self

    def estimate_memory(self, layer_start, layer_end):
        return (layer_end - layer_start) * 1024**3

    def load_partial_model(self, layer_start, layer_end, include_embedding, include_lm_head):
        self.requested = dict(
            layer_start=layer_start,
            layer_end=layer_end,
            include_embedding=include_embedding,
            include_lm_head=include_lm_head,
        )
        return FakePartialModel(layer_end - layer_start, include_embedding, include_lm_head), object()


def test_test_load_reports_the_loaded_slice(runner, monkeypatch):
    monkeypatch.setattr(
        "hydra_worker.core.device.detect_device",
        lambda device: fake_device(device_type="cpu", name="CPU", compute_capability=None),
    )
    monkeypatch.setattr("hydra_worker.models.partial_loader.PartialModelLoader", FakeLoader)

    result = runner.invoke(cli, ["test-load", "org/model", "-s", "0", "-e", "8"])

    assert result.exit_code == 0, result.output
    assert "Loaded 8 layers" in result.output
    assert "Partial loading test passed!" in result.output
    # Layer 0 means this slice owns the embedding; layer 8 of 32 does not own
    # the head.
    assert FakeLoader.last.requested["include_embedding"] is True
    assert FakeLoader.last.requested["include_lm_head"] is False


def test_test_load_marks_the_final_slice_as_owning_the_lm_head(runner, monkeypatch):
    monkeypatch.setattr(
        "hydra_worker.core.device.detect_device",
        lambda device: fake_device(device_type="cpu", name="CPU", compute_capability=None),
    )
    monkeypatch.setattr("hydra_worker.models.partial_loader.PartialModelLoader", FakeLoader)

    result = runner.invoke(cli, ["test-load", "org/model", "-s", "24", "-e", "32"])

    assert result.exit_code == 0, result.output
    assert FakeLoader.last.requested["include_embedding"] is False
    assert FakeLoader.last.requested["include_lm_head"] is True


def test_host_defaults_to_empty_so_the_worker_resolves_it(runner, captured_worker):
    result = runner.invoke(cli, ["start", "--node-id", "worker-1"])

    assert result.exit_code == 0, result.output
    # Empty means "resolve from the local hostname" in _get_host_address().
    assert captured_worker["config"].host == ""


def test_host_flag_sets_the_advertised_address(runner, captured_worker):
    result = runner.invoke(
        cli, ["start", "--node-id", "worker-1", "--host", "192.168.1.90"]
    )

    assert result.exit_code == 0, result.output
    assert captured_worker["config"].host == "192.168.1.90"


def test_host_can_come_from_the_environment(runner, captured_worker, monkeypatch):
    monkeypatch.setenv("HYDRA_WORKER_HOST", "10.0.0.5")
    result = runner.invoke(cli, ["start", "--node-id", "worker-1"])

    assert result.exit_code == 0, result.output
    assert captured_worker["config"].host == "10.0.0.5"
