"""Tests for den-den-mushi log wiring in the worker CLI.

The worker must emit hub-shaped JSON ({ts, level, app, host, msg, ...}) to
stdout by default so the fleet's Vector docker_logs forwarder ships it with no
extra wiring. These tests pin that contract.
"""

import io
import json

import structlog

from hydra_worker.cli import setup_logging, DEFAULT_APP


def _emit(**setup_kwargs):
    """Configure logging into a buffer and return (logger, buffer)."""
    buf = io.StringIO()
    setup_logging(stream=buf, **setup_kwargs)
    return structlog.get_logger(), buf


def test_json_is_default_and_matches_denden_schema():
    log, buf = _emit(app="robin-test")
    log.info("hello world", creator="boa", n=3)
    line = buf.getvalue().strip().splitlines()[-1]
    rec = json.loads(line)  # must be valid JSON

    assert rec["level"] == "info"
    assert rec["app"] == "robin-test"
    assert rec["msg"] == "hello world"       # `event` renamed to `msg`
    assert "ts" in rec and "host" in rec
    assert "event" not in rec
    # App-specific keys survive as top-level fields (queryable in ClickHouse).
    assert rec["creator"] == "boa"
    assert rec["n"] == 3


def test_default_app_tag():
    log, buf = _emit()
    log.info("tick")
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rec["app"] == DEFAULT_APP


def test_warning_level_canonicalized_to_warn():
    # den-den-mushi / the shim use "warn", not Python's "warning".
    log, buf = _emit()
    log.warning("drift high")
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rec["level"] == "warn"


def test_console_format_is_not_json():
    log, buf = _emit(fmt="console")
    log.info("human readable")
    out = buf.getvalue()
    assert "human readable" in out
    assert not out.strip().startswith("{")


def test_env_var_selects_format(monkeypatch):
    monkeypatch.setenv("HYDRA_LOG_FORMAT", "json")
    monkeypatch.setenv("HYDRA_LOG_APP", "robin-env")
    log, buf = _emit()  # no explicit fmt/app -> env wins
    log.info("via env")
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rec["app"] == "robin-env"
    assert rec["msg"] == "via env"
