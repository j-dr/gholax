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

    def constrain(self, y):
        return y

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
    if os.path.exists(f"{prefix}.divergent_chk.npy"):
        div = np.load(f"{prefix}.divergent_chk.npy")
        assert div.dtype == bool and div.shape[0] == N_DEVICES
        assert div.shape[1] <= samples_chk.shape[1]


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
    # max_num_doublings appears only when the pooled warmup converges.
    assert set(warmup) - {"max_num_doublings"} == {
        "inverse_mass_matrix", "step_size", "initial_state", "sample_transform",
    }

    with open(f"{prefix}.minimization_results.json") as fp:
        mres = json.load(fp)
    assert set(mres) == {"x_opt", "x_opt_physical", "value"}
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
    assert set(mres) == {"x_opt", "x_opt_physical", "value"}


@needs_chains
def test_nuts_ess_stopping(tmp_path, capsys):
    """target_min_ess extends sampling past R-hat convergence."""
    from gholax.sampler import NUTS

    cfg = {
        "sampler": {
            "NUTS": {
                "n_steps_warmup": 100,
                "n_steps_min": 20,
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "target_min_ess": 200,
                "minimize_and_sample": True,
            }
        }
    }
    prefix = str(tmp_path / "nuts_ess")
    samples, _ = NUTS(cfg).run(ToyModel(), prefix)
    out = capsys.readouterr().out
    assert "min bulk ESS = " in out and "burn-in fraction" in out
    final_ess = float(out.split("min bulk ESS = ")[-1].split(" ")[0])
    assert final_ess >= 200


def test_nuts_ess_replaces_n_steps_min_default():
    from gholax.sampler import NUTS

    def _n(cfg):
        return NUTS({"sampler": {"NUTS": cfg}}).n_steps_min

    assert _n({}) == 250
    assert _n({"target_min_ess": 100}) == 0
    assert _n({"target_min_ess": 100, "n_steps_min": 50}) == 50


@pytest.mark.parametrize("resample", [True, False])
def test_pathfinder_init_seeds_dispersed_cloud(tmp_path, resample, capsys):
    """Pathfinder seeding returns n_chains finite, dispersed draws whose
    spread matches the (unit) posterior, and saves the init file."""
    from gholax.sampler import NUTS

    s = NUTS({"sampler": {"NUTS": {
        "pathfinder_init": True, "pathfinder_resample": resample,
        "pathfinder_n_paths": 3, "pathfinder_elbo_samples": 10,
    }}})
    jlp = jax.jit(ToyModel().log_posterior_scaled_params)
    prefix = str(tmp_path / "pf")
    pos = s._pathfinder_init(jlp, jnp.array([3.0, -2.0]), 64,
                             jax.random.key(1), prefix)
    pos = np.asarray(pos)
    assert pos.shape == (64, 2) and np.isfinite(pos).all()
    assert 0.5 < pos.std(axis=0).min() and pos.std(axis=0).max() < 1.6
    assert np.abs(pos.mean(axis=0)).max() < 0.6
    assert len(np.unique(pos, axis=0)) > 32
    info = json.load(open(prefix + ".pathfinder_init.json"))
    assert info["resample"] is resample and len(info["elbo"]) == 3
    imm = np.asarray(info["inverse_mass_matrix"])
    assert imm.shape == (2, 2) and np.allclose(imm, s._pathfinder_imm)
    assert np.all(np.abs(np.diag(imm) - 1.0) < 0.5)  # unit posterior
    out = capsys.readouterr().out
    assert ("importance-weight ESS" in out) is resample


def test_nuts_pathfinder_init_run(tmp_path, capsys):
    """End-to-end: pathfinder seeding after minimization, MAP still saved."""
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices")
    from gholax.sampler import NUTS

    cfg = {"sampler": {"NUTS": {
        "n_steps_warmup": 50, "n_steps_min": 10, "n_steps_incr": 10,
        "target_r_minus_one": 0.5, "minimize_and_sample": True,
        "random_start": False,
        "pathfinder_init": True, "pathfinder_n_paths": 2,
        "pathfinder_elbo_samples": 5,
    }}}
    prefix = str(tmp_path / "nuts_pf")
    NUTS(cfg).run(ToyModel(), prefix)
    out = capsys.readouterr().out
    assert "Running Pathfinder" in out
    assert os.path.exists(prefix + ".divergent_chk.npy")
    assert "in parallel with Pathfinder" in out
    assert os.path.exists(prefix + ".minimization_results.json")
    assert os.path.exists(prefix + ".pathfinder_init.json")
    # polished MAP is the saved one: the toy optimum is the origin
    mr = json.load(open(prefix + ".minimization_results.json"))
    assert np.abs(np.array(mr["x_opt"][0])).max() < 1e-2


def test_nuts_pathfinder_mass_matrix_init(tmp_path, capsys):
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices")
    from gholax.sampler import NUTS

    with pytest.raises(ValueError):
        NUTS({"sampler": {"NUTS": {"mass_matrix_init": "pathfinder",
                                   "pathfinder_init": False}}})
    assert NUTS({"sampler": {"NUTS": {}}}).mass_matrix_init == "pathfinder"
    assert NUTS({"sampler": {"NUTS": {"pathfinder_init": False}}}
                ).mass_matrix_init == "hessian_dense"
    cfg = {"sampler": {"NUTS": {
        "n_steps_warmup": 50, "n_steps_min": 10, "n_steps_incr": 10,
        "target_r_minus_one": 0.5, "minimize_and_sample": True,
        "pathfinder_init": True, "pathfinder_n_paths": 2,
        "pathfinder_elbo_samples": 5, "mass_matrix_init": "pathfinder",
    }}}
    NUTS(cfg).run(ToyModel(), str(tmp_path / "nuts_pfimm"))
    out = capsys.readouterr().out
    assert "Dense initial mass matrix from Pathfinder covariance" in out


def test_rank_normalized_diagnostics_transform_invariant():
    """Bulk ESS/R-hat are identical in sampling and physical space, and a
    logit-stretched box edge no longer depresses the stopping ESS."""
    from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
    from gholax.sampler.base import _rank_normalize

    rng = np.random.default_rng(0)
    # AR(1) chains in physical space on a box [0, 1], a few draws near the edge
    x = np.empty((8, 400)); x[:, 0] = 0.5
    for t in range(1, 400):
        x[:, t] = 0.5 + 0.6 * (x[:, t - 1] - 0.5) + 0.1 * rng.standard_normal(8)
    x = np.clip(x, 1e-4, 1 - 1e-4)
    x[0, 100:120] = 1e-3                          # one chain pinned at the edge
    x = x[..., None]
    y = np.log(x / (1 - x))                       # logit (sampling space)
    ess_x = float(effective_sample_size(jnp.asarray(x)).min())
    ess_y = float(effective_sample_size(jnp.asarray(y)).min())
    zx, zy = _rank_normalize(jnp.asarray(x)), _rank_normalize(jnp.asarray(y))
    assert np.allclose(zx, zy)
    ess_z = float(effective_sample_size(zx).min())
    assert ess_y < 0.7 * ess_x                     # raw ESS is transform dependent
    assert 0.7 * ess_x < ess_z < 1.4 * ess_x       # bulk ESS tracks the physical one
    assert np.allclose(potential_scale_reduction(zx), potential_scale_reduction(zy))
