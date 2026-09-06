"""Tests for ZMQ address parsing and port shifting.

The metrics and broadcast endpoints are derived by shifting the coordinator's
port by +1 and +2. The prior implementation used ``rsplit(":", 1)`` which
silently corrupted IPv6 addresses like ``tcp://[::1]:5555``.
"""

import pytest

from hydra_worker.comm.zmq_handler import shift_port


@pytest.mark.parametrize(
    "addr,delta,expected",
    [
        ("tcp://localhost:5555", 1, "tcp://localhost:5556"),
        ("tcp://localhost:5555", 2, "tcp://localhost:5557"),
        ("tcp://127.0.0.1:5555", 1, "tcp://127.0.0.1:5556"),
        ("tcp://*:5555", 1, "tcp://*:5556"),
        ("tcp://*:5555", 2, "tcp://*:5557"),
        ("tcp://192.168.1.10:5555", 1, "tcp://192.168.1.10:5556"),
        # IPv6 loopback
        ("tcp://[::1]:5555", 1, "tcp://[::1]:5556"),
        # Full IPv6
        ("tcp://[2001:db8::1]:5555", 2, "tcp://[2001:db8::1]:5557"),
        # Zero delta
        ("tcp://host:9000", 0, "tcp://host:9000"),
    ],
)
def test_shift_port_valid(addr, delta, expected):
    assert shift_port(addr, delta) == expected


@pytest.mark.parametrize(
    "addr",
    [
        "ipc:///tmp/hydra.sock",
        "inproc://worker",
        # Missing port — returned unchanged.
        "tcp://localhost",
        # Malformed — returned unchanged rather than crashing.
        "garbage",
        "",
    ],
)
def test_shift_port_unparseable_returned_unchanged(addr):
    assert shift_port(addr, 1) == addr


def test_shift_port_preserves_scheme():
    assert shift_port("tcp://host:1000", 5).startswith("tcp://")


def test_shift_port_large_delta():
    # Shifting past common port ranges still produces a well-formed address.
    assert shift_port("tcp://host:60000", 1000) == "tcp://host:61000"
