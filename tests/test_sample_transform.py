"""Logit sample_transform: unbounded reparametrization of uniform-bounded
parameters. Verifies transform math, posterior invariance, end-to-end NUTS
on a box-bounded target, physical-space checkpoints, and restart guards."""

import json
import os
import sys

if "jax" not in sys.modules:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gholax.sampler.priors import Prior

N_DEVICES = jax.local_device_count()
needs_chains = pytest.mark.skipif(N_DEVICES < 2, reason="needs >= 2 devices")

CFG = {
    "a": {"prior": {"dist": "uniform", "min": -2.0, "max": 3.0}, "ref": 0.5},
    "b": {"prior": {"dist": "norm", "loc": 1.0, "scale": 2.0}, "ref": 1.0},
    "c": {"prior": {"dist": "uniform", "min": 0.0, "max": 1.0}, "ref": 0.2},
}


def test_constrain_roundtrip_and_jacobian():
    pr = Prior(dict(CFG), transform=True)
    y = jnp.array([0.3, -1.2, 2.0])
    th = pr.constrain(y)
    # uniform params inside their boxes; norm param affine
    assert -2.0 < float(th[0]) < 3.0
    assert 0.0 < float(th[2]) < 1.0
    assert np.isclose(float(th[1]), -1.2 * 2.0 + 1.0)
    y2 = pr.unconstrain(th)
    assert np.allclose(np.asarray(y2), np.asarray(y), atol=1e-5)
    # analytic Jacobian: sum over uniform dims of log(range*sig*(1-sig))
    s = jax.nn.sigmoid
    lj = (np.log(5.0 * float(s(y[0])) * float(s(-y[0])))
          + np.log(1.0 * float(s(y[2])) * float(s(-y[2]))))
    assert np.isclose(float(pr.log_jacobian(y)), lj, atol=1e-5)
    # transform off: affine both ways, zero Jacobian
    pr0 = Prior(dict(CFG), transform=False)
    th0 = pr0.constrain(y)
    assert np.allclose(np.asarray(pr0.unconstrain(th0)), np.asarray(y))
    assert pr0.log_jacobian(y) == 0.0


def test_initial_position_transformed():
    pr = Prior(dict(CFG), transform=True)
    y = pr.initial_position(random_start=False, normalize=True)
    th = pr.constrain(jnp.array(list(y.values())))
    ref = np.array(list(pr.get_reference_point().values()))
    assert np.allclose(np.asarray(th), ref, atol=1e-5)
    y1 = pr.initial_position(random_start=True, key=jax.random.key(0),
                             normalize=True)
    th1 = np.asarray(pr.constrain(jnp.array(list(y1.values()))))
    assert -2.0 < th1[0] < 3.0 and 0.0 < th1[2] < 1.0


DIM = 8


def make_model(transform):
    """Box-bounded correlated-ish Gaussian via the real Prior/logp plumbing."""
    cfg = {
        f"x{i}": {"prior": {"dist": "uniform", "min": -3.0, "max": 3.0},
                  "ref": 0.0}
        for i in range(DIM)
    }
    prior = Prior(cfg, transform=transform)
    var = jnp.asarray(np.geomspace(0.25, 1.0, DIM), dtype=jnp.float32)

    class ToyModel:
        def __init__(self):
            self.prior = prior

        def log_posterior_scaled_params(self, y):
            th = prior.constrain(y)
            lp = prior.log_prior(dict(zip(prior.params, th)))
            lp += -0.5 * jnp.sum(th**2 / var)
            return lp + prior.log_jacobian(y)

    return ToyModel(), np.asarray(var)


def _nuts(extra=None):
    from gholax.sampler import NUTS

    cfg = {"warmup_algorithm": "pooled_window", "minimize_and_sample": False,
           "chains_per_device": 2, "pooled_window_steps": 5,
           "pooled_window_max_steps": 100, "pooled_window_max_window": 40,
           "pooled_window_min_tail_steps": 16,
           "pooled_window_consecutive_windows": 1,
           "adaptive_warmup_rtol_mass": 0.5, "adaptive_warmup_rtol_step": 0.5,
           "pooled_window_allow_unconverged": True,
           "n_steps_min": 300, "n_steps_incr": 100,
           "target_r_minus_one": 0.1}
    cfg.update(extra or {})
    return NUTS({"sampler": {"NUTS": cfg}})


@needs_chains
def test_nuts_transformed_end_to_end(tmp_path):
    """Transformed NUTS on a box-bounded Gaussian: physical-space moments
    recovered, checkpoint stored physical (inside the box)."""
    model, var = make_model(True)
    prefix = str(tmp_path / "tf")
    samples, names = _nuts().run(model, prefix)
    s = np.asarray(samples)[:, :, :DIM]
    chk = np.load(prefix + ".samples_chk.npy")
    assert chk.min() > -3.0 and chk.max() < 3.0  # physical space
    half = s[:, s.shape[1] // 2:, :].reshape(-1, DIM)
    assert np.max(np.abs(half.mean(axis=0)) / np.sqrt(var)) < 0.35
    assert np.max(np.abs(half.var(axis=0) - var) / var) < 0.5
    with open(prefix + ".nuts_warmup_parameters.json") as fp:
        assert json.load(fp)["sample_transform"] is True


@needs_chains
def test_restart_refuses_convention_mismatch(tmp_path):
    model, _ = make_model(True)
    prefix = str(tmp_path / "mix")
    _nuts().run(model, prefix)
    model0, _ = make_model(False)
    with pytest.raises(RuntimeError, match="sample_transform"):
        _nuts({"restart": True}).run(model0, prefix)
