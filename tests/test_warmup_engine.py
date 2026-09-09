"""The warmup engine standing alone: no sampler, no model."""

import jax
import jax.numpy as jnp
import numpy as np

from gholax.sampler.warmup import (
    Warmup,
    WarmupCheckpoint,
    WarmupConfig,
    WarmupHost,
    WarmupRequest,
)

N_DEVICES = jax.local_device_count()
TRUE_VAR = jnp.array([4.0, 0.25])


def _jlp():
    return jax.jit(lambda x: -0.5 * jnp.sum(x**2 / TRUE_VAR))


def _request(tmp_path=None, n_chains=None, seed=0):
    n_chains = n_chains or 2 * N_DEVICES
    key = jax.random.key(seed)
    pos = jax.random.normal(key, (n_chains, 2)) * jnp.sqrt(TRUE_VAR)
    return WarmupRequest(
        _jlp(),
        pos,
        key,
        output_file=str(tmp_path / "engine") if tmp_path else None,
    )


def _engine(**kw):
    cfg = WarmupConfig(
        window_steps=25,
        max_steps=400,
        max_window=100,
        consecutive_windows=1,
        rtol_mass=0.3,
        rtol_step=0.3,
        **kw,
    )
    return Warmup(cfg, WarmupHost(prefix="nuts"))


def test_engine_adapts_diagonal_metric_without_a_sampler(tmp_path):
    result = _engine().run(_request(tmp_path))
    assert result.converged and result.calibrated
    imm = np.asarray(result.inverse_mass_matrix)
    assert np.allclose(imm, np.asarray(TRUE_VAR), rtol=0.6)
    assert float(result.step_size) > 0
    assert result.positions.shape == (2 * N_DEVICES, 2)


def test_engine_writes_a_resumable_checkpoint(tmp_path):
    warm = _engine()
    warm.run(_request(tmp_path))
    ck = WarmupCheckpoint(str(tmp_path / "engine"), warm.host).read_intermediate()
    assert ck["calibrated"] and ck["warmup_converged"]
    assert set(ck) >= {
        "inverse_mass_matrix",
        "step_size",
        "positions",
        "sample_transform",
        "total_steps",
    }


def test_tree_depth_is_opt_out():
    """Samplers with no tree depth (MCLMC) get no depth cap."""
    req = _request()
    assert _engine().run(req).max_num_doublings is not None
    warm = _engine()
    warm.tune_tree_depth = False
    assert warm.run(req).max_num_doublings is None
