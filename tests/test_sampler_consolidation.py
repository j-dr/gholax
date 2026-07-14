"""Sampler smoke and checkpoint-format tests on a toy Gaussian model.

Guards the BaseSampler consolidation: NUTS, MCLMC, MetropolisHastings, and
Minimize share chain init, pre-minimization, the pmapped inference loop,
checkpointing, and final unscaling. The checkpoint file names, shapes, and
JSON keys asserted here are a public API consumed by
gholax.util.postprocess_chain (used from external notebooks) and by the
samplers' own restart paths.
"""

import json
import os
import sys

# The convergence loops need >= 2 chains for potential_scale_reduction.
# Force multiple host devices when this module is what first initializes
# JAX (standalone run); in a full-suite run JAX is usually already imported
# with a single device and the chain tests skip below.
if "jax" not in sys.modules:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class ToyPrior:
    def __init__(self):
        self.params = ["x0", "x1"]

    def get_prior_sigmas(self):
        return np.array([1.0, 1.0])

    def get_reference_values(self):
        return np.array([0.0, 0.0])

    def get_proposal_sigmas(self):
        return np.array([0.5, 0.5])

    def initial_position(self, random_start=True, key=None, normalize=True):
        if random_start and key is not None:
            vals = jax.random.normal(key, (2,)) * 0.1 + 0.5
        else:
            vals = jnp.full((2,), 0.5)
        return {"x0": vals[0], "x1": vals[1]}


class ToyModel:
    def __init__(self):
        self.prior = ToyPrior()

    def log_posterior_scaled_params(self, p):
        return -0.5 * jnp.sum(p**2)

    def log_posterior(self, p):
        return -0.5 * jnp.sum(p**2)


N_DEVICES = jax.local_device_count()
N_PARAMS = 2

needs_chains = pytest.mark.skipif(
    N_DEVICES < 2,
    reason="R-hat convergence loop requires >= 2 chains (JAX devices); "
    "run this file standalone to force 4 host devices",
)


def _check_common_output(samples, param_names, n_steps_min):
    samples = np.asarray(samples)
    assert samples.shape[0] == N_DEVICES
    assert samples.shape[1] >= n_steps_min
    # params + appended log_posterior column
    assert samples.shape[2] == N_PARAMS + 1
    assert param_names[-1] == "log_posterior"
    assert np.all(np.isfinite(samples))
    # samples of a unit Gaussian rescaled by unit sigmas stay O(1)
    assert np.max(np.abs(samples[:, :, :N_PARAMS])) < 10


def _check_checkpoints(prefix, expect_logpost=True):
    samples_chk = np.load(f"{prefix}.samples_chk.npy")
    assert samples_chk.ndim == 3
    assert samples_chk.shape[0] == N_DEVICES
    assert samples_chk.shape[2] == N_PARAMS
    if expect_logpost:
        logpost_chk = np.load(f"{prefix}.logposterior_chk.npy")
        assert logpost_chk.shape == samples_chk.shape[:2]


@needs_chains
def test_nuts_run_and_checkpoints(tmp_path):
    from gholax.sampler import NUTS

    cfg = {
        "sampler": {
            "NUTS": {
                "n_steps_warmup": 100,
                "n_steps_min": 20,
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "minimize_and_sample": True,
            }
        }
    }
    prefix = str(tmp_path / "nuts")
    samples, param_names = NUTS(cfg).run(ToyModel(), prefix)

    _check_common_output(samples, param_names, 20)
    _check_checkpoints(prefix)

    with open(f"{prefix}.nuts_warmup_parameters.json") as fp:
        warmup = json.load(fp)
    assert set(warmup) == {"inverse_mass_matrix", "step_size", "initial_state"}

    with open(f"{prefix}.minimization_results.json") as fp:
        mres = json.load(fp)
    assert set(mres) == {"x_opt", "value"}
    # MAP of the unit Gaussian is the origin
    assert np.max(np.abs(np.array(mres["x_opt"]))) < 1e-3


@needs_chains
def test_nuts_restart(tmp_path):
    from gholax.sampler import NUTS

    scfg = {
        "n_steps_warmup": 100,
        "n_steps_min": 20,
        "n_steps_incr": 10,
        "target_r_minus_one": 0.5,
        "minimize_and_sample": False,
    }
    prefix = str(tmp_path / "nuts")
    NUTS({"sampler": {"NUTS": dict(scfg)}}).run(ToyModel(), prefix)
    n_before = np.load(f"{prefix}.samples_chk.npy").shape[1]

    scfg.update({"restart": True, "n_steps_min": n_before + 10})
    samples, _ = NUTS({"sampler": {"NUTS": dict(scfg)}}).run(ToyModel(), prefix)
    n_after = np.load(f"{prefix}.samples_chk.npy").shape[1]
    assert n_after >= n_before + 10


@needs_chains
def test_mclmc_run_and_checkpoints(tmp_path):
    from gholax.sampler import MCLMC

    cfg = {
        "sampler": {
            "MCLMC": {
                "n_steps_warmup": 200,
                "n_steps_min": 20,
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "minimize_and_sample": True,
                "max_warmup_rounds": 2,
            }
        }
    }
    prefix = str(tmp_path / "mclmc")
    samples, param_names = MCLMC(cfg).run(ToyModel(), prefix)

    _check_common_output(samples, param_names, 20)
    _check_checkpoints(prefix)

    with open(f"{prefix}.mclmc_warmup_parameters.json") as fp:
        warmup = json.load(fp)
    assert set(warmup) == {
        "L",
        "step_size",
        "inverse_mass_matrix",
        "initial_state",
        "adjusted",
    }


@needs_chains
def test_metropolis_hastings_run_and_checkpoints(tmp_path):
    from gholax.sampler import MetropolisHastings

    cfg = {
        "sampler": {
            "MetropolisHastings": {
                "n_steps": 10,
                "n_steps_checkpoint": 5,
                "n_steps_incr": 100,
                "target_r_minus_one": 1.0,
                "init_covariance": "None",
                "minimize_and_sample": True,
            }
        }
    }
    prefix = str(tmp_path / "mh")
    samples, param_names = MetropolisHastings(cfg).run(ToyModel(), prefix)

    _check_common_output(samples, param_names, 10)
    _check_checkpoints(prefix)

    proposal_cov = np.load(f"{prefix}.proposal_cov.npy")
    assert proposal_cov.shape == (N_PARAMS, N_PARAMS)
    assert np.all(np.linalg.eigvalsh(proposal_cov) > 0)


def test_minimize_run(tmp_path):
    from gholax.sampler import Minimize

    cfg = {"sampler": {"Minimize": {}}}
    prefix = str(tmp_path / "minimize")
    samples, param_names = Minimize(cfg).run(ToyModel(), prefix)

    samples = np.asarray(samples)
    assert samples.shape == (N_DEVICES, 1, N_PARAMS + 1)
    assert param_names[-1] == "log_posterior"
    # MAP of the unit Gaussian is the origin with log posterior 0
    assert np.max(np.abs(samples[:, 0, :N_PARAMS])) < 1e-3
    assert np.max(np.abs(samples[:, 0, -1])) < 1e-6

    with open(f"{prefix}.minimization_results.json") as fp:
        mres = json.load(fp)
    assert set(mres) == {"x_opt", "value"}
