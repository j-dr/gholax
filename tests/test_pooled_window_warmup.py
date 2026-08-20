"""Tests for the pooled cross-chain window warmup (warmup_algorithm='pooled_window').

Uses toy anisotropic Gaussian targets only; CPU-runnable. Run standalone to
force 4 host devices for the chain-parallel tests.
"""

import json
import os
import re
import sys

if "jax" not in sys.modules:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N_DEVICES = jax.local_device_count()
DIM = 32

needs_chains = pytest.mark.skipif(
    N_DEVICES < 2,
    reason="R-hat convergence loop requires >= 2 chains (JAX devices); "
    "run this file standalone to force 4 host devices",
)


def _true_var(cond):
    """Anisotropic diagonal Gaussian variances with condition number `cond`."""
    return np.geomspace(1.0 / np.sqrt(cond), np.sqrt(cond), DIM)


def make_model(cond):
    var = _true_var(cond)
    prec = jnp.asarray(1.0 / var, dtype=jnp.float32)

    class ToyPrior:
        def __init__(self):
            self.params = [f"x{i}" for i in range(DIM)]

        def get_prior_sigmas(self):
            return np.ones(DIM)

        def get_reference_values(self):
            return np.zeros(DIM)

        def initial_position(self, random_start=True, key=None, normalize=True):
            if random_start and key is not None:
                vals = jax.random.normal(key, (DIM,)) * 0.1 + 0.5
            else:
                vals = jnp.full((DIM,), 0.5)
            return {f"x{i}": vals[i] for i in range(DIM)}

    class ToyModel:
        def __init__(self):
            self.prior = ToyPrior()

        def log_posterior_scaled_params(self, p):
            return -0.5 * jnp.sum(prec * p**2)

    return ToyModel()


def _nuts(scfg):
    from gholax.sampler import NUTS

    return NUTS({"sampler": {"NUTS": dict(scfg)}})


POOLED_CFG = {
    "warmup_algorithm": "pooled_window",
    "n_steps_min": 400,
    "n_steps_incr": 100,
    "target_r_minus_one": 0.1,
    "minimize_and_sample": False,
    "chains_per_device": 2,
}

WINDOW_DEFAULT_STEPS = 500  # n_steps_warmup default of the 'window' path


def _warmup_steps_used(out):
    """Sequential warmup steps parsed from the pooled-warmup log."""
    m = re.search(r"Pooled warmup converged after (\d+) steps", out)
    if m:
        return int(m.group(1))
    m = re.search(r"Pooled warmup reached max (\d+) steps", out)
    assert m, "no pooled warmup completion message in output"
    return int(m.group(1))


def test_pooled_window_algorithm_validated():
    nuts = _nuts({"warmup_algorithm": "pooled_window"})
    assert nuts.warmup_algorithm == "pooled_window"
    assert nuts.pooled_window_steps == 25
    assert nuts.pooled_window_max_steps == 200
    with pytest.raises(ValueError, match="warmup_algorithm"):
        _nuts({"warmup_algorithm": "pooled_windows"})


def test_pooled_window_mesh_mode_raises(tmp_path):
    cfg = {
        "warmup_algorithm": "pooled_window",
        "n_chains": N_DEVICES,
        "model_shards": 1,
        "minimize_and_sample": False,
    }
    with pytest.raises(
        ValueError, match="pooled_window.*not.*supported in mesh mode"
    ):
        _nuts(cfg).run(make_model(10.0), str(tmp_path / "mesh_pooled"))


@needs_chains
@pytest.mark.parametrize("cond", [10.0, 100.0], ids=["cond10", "cond100"])
def test_pooled_window_end_to_end(tmp_path, capsys, cond):
    """8 pooled chains: the adapted diagonal mass matrix must match the true
    marginal variances, warmup must use materially fewer sequential steps
    than the 500-step window default, and downstream sampling must converge
    with sane moments."""
    n_total = 2 * N_DEVICES
    prefix = str(tmp_path / f"pooled_c{int(cond)}")
    samples, param_names = _nuts(POOLED_CFG).run(make_model(cond), prefix)
    out = capsys.readouterr().out
    assert f"Running pooled window warmup ({n_total} chains, 25-step windows" in out
    assert "rhat - 1 (latter half)" in out

    steps_used = _warmup_steps_used(out)
    assert steps_used <= 200 < WINDOW_DEFAULT_STEPS

    with open(f"{prefix}.nuts_warmup_parameters.json") as fp:
        warmup = json.load(fp)
    true_var = _true_var(cond)
    mass = np.array(warmup["inverse_mass_matrix"])
    rel_err = np.abs(mass - true_var) / true_var
    print(
        f"[cond={cond}] warmup steps used = {steps_used} "
        f"(window default {WINDOW_DEFAULT_STEPS}); "
        f"mass rel err vs truth: max = {rel_err.max():.4f}, "
        f"mean = {rel_err.mean():.4f}"
    )
    # The final window pools ~200 samples: the mean rel err is the tight
    # "matches truth" check; the max over 32 dims gets a 4-sigma bound.
    assert rel_err.mean() < 0.25
    assert rel_err.max() < 0.6

    # Final chain positions are the per-chain starts: all distinct.
    starts = np.array(warmup["initial_state"])
    assert starts.shape == (n_total, DIM)
    assert len(np.unique(starts, axis=0)) == n_total

    samples = np.asarray(samples)
    assert samples.shape[0] == n_total
    assert samples.shape[2] == DIM + 1
    assert np.all(np.isfinite(samples))

    # Latter half against the known target moments.
    half = samples[:, samples.shape[1] // 2 :, :DIM].reshape(-1, DIM)
    assert np.max(np.abs(half.mean(axis=0)) / np.sqrt(true_var)) < 0.35
    assert np.max(np.abs(half.var(axis=0) - true_var) / true_var) < 0.5


@needs_chains
def test_pooled_window_resume_from_checkpoint(tmp_path, capsys):
    """Warmup interrupted after two windows must resume from the boundary
    checkpoint and end up with parameters equivalent to an uninterrupted
    run's."""
    model = make_model(10.0)
    true_var = _true_var(10.0)
    nuts = _nuts(POOLED_CFG)
    setup = nuts._init_chains(model)
    positions = setup.initial_positions

    # "Killed" run: max_steps == 2 windows stops with the intermediate
    # checkpoint on disk but no final parameters file.
    prefix = str(tmp_path / "pooled")
    interrupted = _nuts({**POOLED_CFG, "pooled_window_max_steps": 50})
    interrupted._pooled_window_warmup(
        setup.jlp, jax.random.key(0), positions, output_file=prefix
    )
    ckpt_file = f"{prefix}.nuts_warmup_intermediate.json"
    assert os.path.exists(ckpt_file)
    with open(ckpt_file) as fp:
        ck = json.load(fp)
    assert set(ck) == {
        "inverse_mass_matrix", "step_size", "positions", "total_steps",
        "between_within_ratio", "window_chunks",
    }
    assert ck["total_steps"] == 50
    assert np.array(ck["positions"]).shape == (2 * N_DEVICES, DIM)

    # Restart: adaptation must resume from the checkpointed window.
    resumed = _nuts({**POOLED_CFG, "restart": True})
    states_r, params_r = resumed._pooled_window_warmup(
        setup.jlp, jax.random.key(1), positions, output_file=prefix
    )
    out = capsys.readouterr().out
    assert "Resuming pooled window warmup from checkpointed step 50" in out
    with open(ckpt_file) as fp:
        assert json.load(fp)["total_steps"] > 50

    # Uninterrupted run for comparison.
    full = _nuts(POOLED_CFG)
    states_f, params_f = full._pooled_window_warmup(
        setup.jlp, jax.random.key(2), positions,
        output_file=str(tmp_path / "pooled_full"),
    )

    mass_r = np.asarray(params_r["inverse_mass_matrix"])
    mass_f = np.asarray(params_f["inverse_mass_matrix"])
    # Both runs estimate the same true variances within sampling noise.
    assert np.all(np.abs(mass_r - true_var) / true_var < 0.75)
    assert np.all(np.abs(mass_f - true_var) / true_var < 0.75)
    assert 0.4 < float(params_r["step_size"]) / float(params_f["step_size"]) < 2.5


@needs_chains
def test_pooled_window_warm_start_seeds_convergence_check(capsys):
    """A warm start (mass + step size) must seed the convergence check so
    the very first window boundary is checked; a cold start prints no check
    on its first boundary."""
    model = make_model(10.0)
    nuts = _nuts({**POOLED_CFG, "pooled_window_max_steps": 25})
    setup = nuts._init_chains(model)

    nuts._pooled_window_warmup(setup.jlp, jax.random.key(0), setup.initial_positions)
    cold_out = capsys.readouterr().out
    assert "Pooled warmup step 25 " not in cold_out

    nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.asarray(_true_var(10.0)),
        initial_step_size=0.5,
    )
    warm_out = capsys.readouterr().out
    assert "Pooled warmup step 25 " in warm_out


@needs_chains
def test_pooled_window_overdispersed_start_keeps_seed_metric(capsys):
    """Chains started far outside the posterior (spread >> width) with a
    known-good seed imm: the boundary update must use the within-chain
    variance shrunk toward the seed, so after 2 windows the imm stays
    within a factor of a few of truth instead of jumping to the
    start-spread scale (the old total-variance estimator inflated it by
    ~spread^2). The unmixed state must be flagged via the between/within
    ratio."""
    model = make_model(10.0)
    true_var = _true_var(10.0)
    nuts = _nuts({**POOLED_CFG, "pooled_window_max_steps": 50})
    setup = nuts._init_chains(model)
    n_total = setup.initial_positions.shape[0]

    spread = 50.0  # start std = 50x posterior scale in every dim
    starts = (
        jax.random.normal(jax.random.key(3), (n_total, DIM))
        * spread * jnp.sqrt(jnp.asarray(true_var))
    )
    _, params = nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), starts,
        initial_inverse_mass_matrix=jnp.asarray(true_var),
        initial_step_size=0.5,
    )
    out = capsys.readouterr().out
    assert "max_between_within_ratio" in out

    mass = np.asarray(params["inverse_mass_matrix"])
    factor = np.maximum(mass / true_var, true_var / mass)
    print(f"imm-vs-truth factor: max = {factor.max():.2f}")
    # Old estimator: median inflation O(spread^2) ~ 2500x. Fixed: a few.
    assert factor.max() < 5.0


@needs_chains
def test_pooled_window_step_size_search(capsys):
    """With no step_size_init in the config, warmup runs the doubling/halving
    search and lands near the toy target's stable eps; pinning step_size_init
    skips the search."""
    model = make_model(10.0)
    nuts = _nuts({**POOLED_CFG, "pooled_window_max_steps": 10})
    assert nuts.step_size_search
    setup = nuts._init_chains(model)
    nuts._pooled_window_warmup(setup.jlp, jax.random.key(0), setup.initial_positions)
    out = capsys.readouterr().out
    m = re.search(r"Initial step size search: ([0-9.e+-]+)", out)
    assert m
    assert 1e-3 < float(m.group(1)) < 10.0

    nuts = _nuts({**POOLED_CFG, "pooled_window_max_steps": 10,
                  "step_size_init": 0.05})
    assert not nuts.step_size_search
    setup = nuts._init_chains(model)
    nuts._pooled_window_warmup(setup.jlp, jax.random.key(0), setup.initial_positions)
    assert "Initial step size search" not in capsys.readouterr().out


@needs_chains
def test_pooled_window_depth_cap(capsys):
    """Auto depth cap: warmup reports depth percentiles and returns a
    max_num_doublings capped at quantile-depth + 1; pinning max_num_doublings
    in the config disables it."""
    model = make_model(10.0)
    nuts = _nuts({**POOLED_CFG, "pooled_window_max_steps": 10})
    assert nuts.max_num_doublings_auto
    setup = nuts._init_chains(model)
    _, params = nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions
    )
    out = capsys.readouterr().out
    assert "Warmup tree depth percentiles" in out
    assert 1 <= params["max_num_doublings"] <= 10
    assert nuts.max_num_doublings == params["max_num_doublings"]

    nuts = _nuts({**POOLED_CFG, "pooled_window_max_steps": 10,
                  "max_num_doublings": 6})
    assert not nuts.max_num_doublings_auto
    setup = nuts._init_chains(model)
    _, params = nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions
    )
    assert "max_num_doublings" not in params
    assert nuts.max_num_doublings == 6


@needs_chains
def test_pooled_window_mixing_gate(capsys):
    """An unreachable mixing gate must block convergence (run to max steps)
    even when mass/step rtols are satisfied."""
    model = make_model(10.0)
    nuts = _nuts({**POOLED_CFG, "pooled_window_max_steps": 20,
                  "adaptive_warmup_rtol_mass": 1e9,
                  "adaptive_warmup_rtol_step": 1e9,
                  "pooled_window_mixing_gate": 0.0})
    setup = nuts._init_chains(model)
    nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.asarray(_true_var(10.0)),
        initial_step_size=0.5,
    )
    out = capsys.readouterr().out
    assert "reached max 20 steps" in out

    nuts = _nuts({**POOLED_CFG, "pooled_window_max_steps": 20,
                  "adaptive_warmup_rtol_mass": 1e9,
                  "adaptive_warmup_rtol_step": 1e9,
                  "pooled_window_mixing_gate": 1e9})
    setup = nuts._init_chains(model)
    nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.asarray(_true_var(10.0)),
        initial_step_size=0.5,
    )
    assert "converged after" in capsys.readouterr().out


@needs_chains
def test_pooled_window_growing_windows(capsys):
    """Window length doubles each boundary from pooled_window_steps up to
    pooled_window_max_window."""
    model = make_model(10.0)
    nuts = _nuts({**POOLED_CFG, "pooled_window_steps": 5,
                  "pooled_window_max_steps": 40,
                  "pooled_window_max_window": 20,
                  "pooled_window_mixing_gate": 0.0,
                  "adaptive_warmup_rtol_mass": 1e9,
                  "adaptive_warmup_rtol_step": 1e9})
    setup = nuts._init_chains(model)
    nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.asarray(_true_var(10.0)),
        initial_step_size=0.5,
    )
    out = capsys.readouterr().out
    for tag in ["step 5 (window 5)", "step 15 (window 10)",
                "step 35 (window 20)", "step 55 (window 20)"]:
        assert f"Pooled warmup {tag}" in out, tag
