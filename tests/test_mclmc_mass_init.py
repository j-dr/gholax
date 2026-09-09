"""Tests for BaseSampler._mclmc_mass_matrix (mass_matrix_init: 'mclmc').

Accuracy calibration: on a dim-16 diagonal Gaussian with variances spanning
[0.25, 4], measured per-element estimates over several seeds land well within
a factor of 2 of truth; the tests assert factor-of-2 agreement per element.
"""

import os
import sys

if "jax" not in sys.modules:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gholax.sampler.base import BaseSampler

DIM = 16
# Known marginal variances spanning a factor of 16.
TRUE_VAR = jnp.array(np.geomspace(0.25, 4.0, DIM))


def _log_posterior(p):
    return -0.5 * jnp.sum(p**2 / TRUE_VAR)


def _sampler():
    return BaseSampler()


def _starts(n=4, seed=0):
    return 0.1 * jax.random.normal(jax.random.key(seed), (n, DIM))


def test_recovers_true_variances():
    imm, info = _sampler()._mclmc_mass_matrix(
        _log_posterior, _starts(), jax.random.key(1)
    )
    assert info["rung"] == "eps/2"
    assert info["n_survivors"] == 8
    ratio = np.asarray(imm / TRUE_VAR)
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)


def test_nan_chains_dropped():
    # Poison some starting rows with inf: those chains go non-finite and
    # must be dropped without corrupting the estimate.
    starts = _starts(8)
    starts = starts.at[1].set(jnp.inf).at[5].set(jnp.inf)
    imm, info = _sampler()._mclmc_mass_matrix(
        _log_posterior, starts, jax.random.key(2)
    )
    assert info["rung"] == "eps/2"
    assert info["n_survivors"] == 6
    ratio = np.asarray(imm / TRUE_VAR)
    assert np.all(np.isfinite(imm))
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)


def test_full_failure_falls_back_to_hessian(capsys):
    # All chains start at inf -> both eps rungs fail -> hessian rung.
    starts = jnp.full((4, DIM), jnp.inf)
    # Hessian is evaluated at starts[0]; give it a finite row so the
    # hessian rung itself succeeds.
    starts = starts.at[0].set(jnp.zeros(DIM))

    class AllInfStarts(BaseSampler):
        pass

    sampler = AllInfStarts()
    # Force estimation chains non-finite regardless of tune by poisoning
    # every start after the hessian anchor row is used: easiest is a
    # logdensity that is nan away from exactly zero jitter cannot hit.
    def poisoned_lp(p):
        lp = _log_posterior(p)
        return jnp.where(jnp.max(jnp.abs(p)) < 1e-12, lp, jnp.nan)

    imm, info = sampler._mclmc_mass_matrix(
        poisoned_lp, starts, jax.random.key(3)
    )
    out = capsys.readouterr().out
    assert info["rung"] in ("hessian", "ones")
    assert info["n_survivors"] == 0
    assert "falling back to Hessian" in out
    assert np.all(np.isfinite(imm))
    assert np.all(np.asarray(imm) >= 1e-8) and np.all(np.asarray(imm) <= 1e8)


needs_chains = pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="convergence loop requires >= 2 devices",
)


@needs_chains
def test_mclmc_sampler_integration(tmp_path):
    from tests.test_sampler_consolidation import ToyModel
    from gholax.sampler import MCLMC

    cfg = {
        "sampler": {
            "MCLMC": {
                "n_steps_warmup": 200,
                "n_steps_min": 20,
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "minimize_and_sample": False,
                "max_warmup_rounds": 2,
                "mass_matrix_init": "mclmc",
            }
        }
    }
    prefix = str(tmp_path / "mclmc_mm")
    samples, param_names = MCLMC(cfg).run(ToyModel(), prefix)
    samples = np.asarray(samples)
    assert np.all(np.isfinite(samples))
    assert param_names[-1] == "log_posterior"

