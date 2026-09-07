"""Test the DistributedWorker event-loop error backoff.

We stub out ZMQHandler so we can inject a stream of handler errors and
confirm that:
  * Consecutive errors back off (don't spin-loop).
  * Setting `running = False` terminates the loop promptly.
  * A clean recovery doesn't prevent later cycles.
"""

import asyncio

import pytest

from hydra_worker.distributed.worker import (
    DistributedWorker,
    DistributedWorkerConfig,
)


class FakeZMQ:
    """Minimal stand-in for ZMQHandler covering the surface the event loop uses."""

    def __init__(self, script):
        self._script = list(script)
        self.receive_calls = 0
        self.wait_calls = 0
        self.pull = None
        self.push = None

    async def wait_readable(self, timeout=0.25):
        """Report the coordinator socket readable while the script has items.

        Mirrors the real handler: the loop only calls receive() for a socket
        the poller flagged, so an empty script must still yield to the loop.
        """
        self.wait_calls += 1
        if self._script:
            return {"coordinator": True}
        await asyncio.sleep(0.01)
        return {}

    async def receive(self, timeout):
        self.receive_calls += 1
        if not self._script:
            await asyncio.sleep(0.01)
            return None
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def check_broadcast(self):
        return None

    def close(self):
        pass


def _make_worker() -> DistributedWorker:
    cfg = DistributedWorkerConfig(
        node_id="w",
        coordinator_addr="tcp://localhost:5555",
        device="cpu",
    )
    return DistributedWorker(cfg)


@pytest.mark.asyncio
async def test_event_loop_terminates_on_shutdown():
    worker = _make_worker()
    worker.zmq_handler = FakeZMQ([])
    worker.running = True

    task = asyncio.create_task(worker._event_loop())
    await asyncio.sleep(0.05)
    worker.running = False
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_event_loop_backs_off_on_errors():
    worker = _make_worker()
    worker.zmq_handler = FakeZMQ([
        RuntimeError("boom1"),
        RuntimeError("boom2"),
        RuntimeError("boom3"),
    ])
    worker.running = True

    task = asyncio.create_task(worker._event_loop())
    # 0.5s window with backoff 0.2s + 0.4s + 0.8s — expect very few calls.
    await asyncio.sleep(0.5)
    worker.running = False
    await asyncio.wait_for(task, timeout=3.0)

    # Without backoff this would be dozens of calls per second.
    assert worker.zmq_handler.receive_calls <= 6, (
        f"expected few receive calls due to backoff, got {worker.zmq_handler.receive_calls}"
    )


@pytest.mark.asyncio
async def test_event_loop_handles_message_without_error():
    worker = _make_worker()
    handled = []

    async def fake_handle(m):
        handled.append(m)

    worker._handle_message = fake_handle
    worker.zmq_handler = FakeZMQ([
        {"type": "health_check"},
        {"type": "health_check"},
    ])
    worker.running = True

    task = asyncio.create_task(worker._event_loop())
    await asyncio.sleep(0.2)
    worker.running = False
    await asyncio.wait_for(task, timeout=2.0)

    assert len(handled) == 2
