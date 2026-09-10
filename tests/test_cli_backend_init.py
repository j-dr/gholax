"""Importing the CLI must not bring up the XLA backend.

jax.distributed.initialize() has to run before anything touches a device, so
gholax.cli defers its heavy imports (which reach interpax, and interpax
materializes a jnp array at module scope) until after maybe_init_distributed.
If that ordering regresses, every multi-node launch dies with "must be called
before any JAX calls that might initialise the XLA backend", and each task on
a node grabs all of that node's GPUs and OOMs.

Runs in a subprocess so a backend initialized by another test can't mask it.
"""

import subprocess
import sys

PROBE = """
import gholax.cli
from jax._src import xla_bridge as xb
backends = list(xb._backends.keys())
print("BACKENDS=" + ",".join(backends))
"""


def test_cli_import_does_not_initialize_backend():
    res = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
    line = [l for l in res.stdout.splitlines() if l.startswith("BACKENDS=")][0]
    backends = [b for b in line.split("=", 1)[1].split(",") if b]
    assert backends == [], (
        f"importing gholax.cli initialized XLA backend(s) {backends}; "
        "multi-node runs require the backend to stay down until "
        "maybe_init_distributed() has run"
    )
