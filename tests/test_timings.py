"""
Timing measurements for ExpansionHistory.compute_analytic and LinearGrowth.compute_ode.

Run with:
    pytest tests/test_timings.py -v -s

Each test JIT-compiles on a warmup call, then times N_REPEAT subsequent calls.
Results are printed to stdout so they are visible with -s.
"""
import gholax.util.likelihood_module  # noqa: F401
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gholax.theory.expansion_history import ExpansionHistory
from gholax.theory.linear_growth import LinearGrowth
from tests.conftest import COSMO_PARAMS

N_REPEAT = 20
# Convert params to jnp arrays so JAX traces them as abstract values rather
# than baking the Python floats as compile-time constants.
PARAMS = {k: jnp.array(v) for k, v in COSMO_PARAMS["standard"].items()}


def _block(result):
    """Call block_until_ready on every leaf array in a pytree."""
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)


def _time_fn(fn, *args, n=N_REPEAT):
    """JIT-compile fn, then time n post-compile calls. Returns (mean_ms, min_ms)."""
    jit_fn = jax.jit(fn)
    _block(jit_fn(*args))          # triggers tracing + XLA compilation; block until done

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _block(jit_fn(*args))      # block_until_ready ensures we measure real wall time
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.mean(times)), float(np.min(times))


# ---------------------------------------------------------------------------
# ExpansionHistory.compute_analytic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nz,label", [
    (125,  "nz=125  (default)"),
    (500,  "nz=500             "),
])
def test_time_compute_analytic(nz, label):
    eh = ExpansionHistory(zmin=0.0, zmax=2.0, nz=nz,
                          use_direct_integration=True,
                          use_emulator=False,
                          use_boltzmann=False)

    mean_ms, min_ms = _time_fn(eh.compute_analytic, PARAMS)
    print(f"\n  compute_analytic  {label}  "
          f"mean={mean_ms:.2f} ms  min={min_ms:.2f} ms  (n={N_REPEAT})")


# ---------------------------------------------------------------------------
# LinearGrowth.compute_ode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_pts,label", [
    (30,  "n_points_ode=30  (default)"),
    (100, "n_points_ode=100           "),
    (500, "n_points_ode=500           "),
])
def test_time_compute_ode(n_pts, label):
    lg = LinearGrowth(
        zmin=0.0, zmax=2.0, nz=125,
        use_emulator=False,
        use_boltzmann=True,
        solve_ode=True,
        n_points_ode=n_pts,
        a_init_ode=1.0 / (1.0 + 200.0),
    )

    mean_ms, min_ms = _time_fn(lg.compute_ode, {}, PARAMS)
    print(f"\n  compute_ode  {label}  "
          f"mean={mean_ms:.2f} ms  min={min_ms:.2f} ms  (n={N_REPEAT})")
