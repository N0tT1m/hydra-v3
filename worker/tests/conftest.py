"""Shared pytest configuration.

Adds the worker package to sys.path so tests can run from the worker/ directory
without requiring an editable install.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if WORKER_ROOT not in sys.path:
    sys.path.insert(0, WORKER_ROOT)


import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Seed torch before every test.

    Much of the suite builds inputs and synthetic weights with torch.randn and
    then asserts on tolerances — quantization round-trips especially, where the
    error depends on the draw. Unseeded, those tests fail a small fraction of
    runs. Seeding per test (rather than once per session) keeps each test
    independent of the order the others ran in.
    """
    torch.manual_seed(0)
