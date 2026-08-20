"""Tests for sampler speedup features: intermediate warmup checkpoints,
warmup warm starts, jittered chain starts, hoisted reinit compilation,
batched Hessian mass-matrix estimation, and latter-half R-hat.

Uses toy Gaussian posteriors only; CPU-runnable. Run standalone to force
4 host devices for the chain-parallel tests.
"""

import json
import logging
import os
import re
import sys
import time

if "jax" not in sys.modules:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N_DEVICES = jax.local_device_count()
N_PARAMS = 2

needs_chains = pytest.mark.skipif(
    N_DEVICES < 2,
    reason="R-hat convergence loop requires >= 2 chains (JAX devices); "
    "run this file standalone to force 4 host devices",
)


class ToyPrior:
    def __init__(self):
        self.params = ["x0", "x1"]

    def get_prior_sigmas(self):
        return np.array([1.0, 1.0])

    def get_reference_values(self):
        return np.array([0.0, 0.0])

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


class MeshToyModel(ToyModel):
    """Toy model exposing the mesh-mode posterior hook (model_shards=1)."""

    def sharded_log_posterior_scaled_params(self, mesh):
        return jax.jit(self.log_posterior_scaled_params)


def _toy_jlp():
    return jax.jit(lambda p: -0.5 * jnp.sum(p**2))


def _nuts(scfg):
    from gholax.sampler import NUTS

    return NUTS({"sampler": {"NUTS": dict(scfg)}})


def _mclmc(scfg):
    from gholax.sampler import MCLMC

    return MCLMC({"sampler": {"MCLMC": dict(scfg)}})


# ---------------------------------------------------------------------------
# warmup-intermediate-checkpoints
# ---------------------------------------------------------------------------

ADAPTIVE_CFG = {
    "warmup_algorithm": "adaptive_window",
    "adaptive_warmup_stage_steps": 50,
    "adaptive_warmup_min_steps": 100,
    "adaptive_warmup_max_steps": 400,
    "adaptive_warmup_rtol_mass": 0.5,
    "adaptive_warmup_rtol_step": 0.5,
    "minimize_and_sample": False,
}


def test_nuts_warmup_intermediate_checkpoint_resume(tmp_path, capsys):
    jlp = _toy_jlp()
    pos0 = jnp.zeros(N_PARAMS)
    key = jax.random.key(0)

    # "Killed" run: max_steps == stage_steps stops after stage 1 with the
    # intermediate checkpoint on disk but no final parameters file.
    prefix = str(tmp_path / "nuts")
    interrupted = _nuts({**ADAPTIVE_CFG, "adaptive_warmup_max_steps": 50})
    interrupted._adaptive_window_warmup(jlp, key, pos0, output_file=prefix)

    ckpt_file = f"{prefix}.nuts_warmup_intermediate.json"
    assert os.path.exists(ckpt_file)
    with open(ckpt_file) as fp:
        ck = json.load(fp)
    assert set(ck) == {"inverse_mass_matrix", "step_size", "position", "total_steps"}
    assert ck["total_steps"] == 50

    # Restart: adaptation must resume from the checkpointed stage.
    resumed = _nuts({**ADAPTIVE_CFG, "restart": True})
    state_r, params_r = resumed._adaptive_window_warmup(
        jlp, jax.random.key(1), pos0, output_file=prefix
    )
    out = capsys.readouterr().out
    assert "Resuming adaptive warmup from checkpointed step 50" in out
    with open(ckpt_file) as fp:
        assert json.load(fp)["total_steps"] > 50

    # Uninterrupted run for comparison.
    full = _nuts(ADAPTIVE_CFG)
    state_f, params_f = full._adaptive_window_warmup(
        jlp, jax.random.key(2), pos0, output_file=str(tmp_path / "nuts_full")
    )

    mass_r = np.asarray(params_r["inverse_mass_matrix"])
    mass_f = np.asarray(params_f["inverse_mass_matrix"])
    step_r = float(params_r["step_size"])
    step_f = float(params_f["step_size"])
    # Unit Gaussian target: adapted parameters agree within sampling noise.
    assert np.all(np.abs(mass_r - mass_f) / np.abs(mass_f) < 0.75)
    assert 0.4 < step_r / step_f < 2.5


def _halfway_adapt_stub(target):
    """Deterministic _adapt_unadjusted replacement: params move halfway to
    target each round."""
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    def stub(self, jlp, state, rng_key, params, diagonal_preconditioning=None):
        new = MCLMCAdaptationState(
            L=0.5 * (params.L + target.L),
            step_size=0.5 * (params.step_size + target.step_size),
            inverse_mass_matrix=0.5
            * (params.inverse_mass_matrix + target.inverse_mass_matrix),
        )
        return state, new

    return stub


def test_mclmc_adapt_intermediate_checkpoint_resume(tmp_path, monkeypatch, capsys):
    import blackjax
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    from gholax.sampler import MCLMC

    target = MCLMCAdaptationState(
        L=jnp.asarray(2.0),
        step_size=jnp.asarray(0.1),
        inverse_mass_matrix=jnp.array([4.0, 9.0]),
    )
    monkeypatch.setattr(MCLMC, "_adapt_unadjusted", _halfway_adapt_stub(target))

    jlp = _toy_jlp()
    params0 = MCLMCAdaptationState(
        L=jnp.asarray(1.0),
        step_size=jnp.asarray(0.01),
        inverse_mass_matrix=jnp.ones(2),
    )
    state0 = blackjax.mcmc.mclmc.init(
        position=jnp.zeros(2), logdensity_fn=jlp, rng_key=jax.random.key(0)
    )

    base_cfg = {"warmup_tolerance": 1e-8, "max_warmup_rounds": 5}
    prefix = str(tmp_path / "mclmc")
    ckpt_file = f"{prefix}.mclmc_warmup_intermediate.json"

    # "Killed" run: stops after round 1 with the checkpoint on disk.
    interrupted = _mclmc({**base_cfg, "max_warmup_rounds": 1})
    interrupted._adapt_with_convergence(
        jlp, state0, jax.random.key(1), params0, output_file=prefix
    )
    with open(ckpt_file) as fp:
        ck = json.load(fp)
    assert ck["round"] == 1

    # Restart resumes from round 2 and reproduces the uninterrupted sequence
    # exactly (the stub is deterministic).
    resumed = _mclmc({**base_cfg, "restart": True})
    _, params_r = resumed._adapt_with_convergence(
        jlp, state0, jax.random.key(2), params0, output_file=prefix
    )
    assert "Resuming MCLMC adaptation from checkpointed round 1" in capsys.readouterr().out

    full = _mclmc(base_cfg)
    _, params_f = full._adapt_with_convergence(
        jlp, state0, jax.random.key(3), params0,
        output_file=str(tmp_path / "mclmc_full"),
    )

    assert np.allclose(float(params_r.L), float(params_f.L))
    assert np.allclose(float(params_r.step_size), float(params_f.step_size))
    assert np.allclose(
        np.asarray(params_r.inverse_mass_matrix),
        np.asarray(params_f.inverse_mass_matrix),
    )


# ---------------------------------------------------------------------------
# convergence-loop-recompile
# ---------------------------------------------------------------------------


@needs_chains
def test_mesh_reinit_does_not_recompile_per_batch(tmp_path, caplog):
    """The mapped init must be built once: previously each n_steps_incr batch
    re-traced and recompiled it in mesh mode (one 'Compiling body' per batch)."""
    from gholax.sampler import NUTS

    cfg = {
        "sampler": {
            "NUTS": {
                "n_chains": N_DEVICES,
                "model_shards": 1,
                "n_steps_warmup": 50,
                "n_steps_min": 60,  # 6 batches of n_steps_incr
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "minimize_and_sample": False,
            }
        }
    }
    jax.config.update("jax_log_compiles", True)
    try:
        with caplog.at_level(logging.WARNING, logger="jax._src.interpreters.pxla"):
            NUTS(cfg).run(MeshToyModel(), str(tmp_path / "mesh_recompile"))
    finally:
        jax.config.update("jax_log_compiles", False)

    init_compiles = [
        r.getMessage()
        for r in caplog.records
        if "Compiling body" in r.getMessage()
        and f"[ShapedArray(float32[{N_DEVICES},{N_PARAMS}])]" in r.getMessage()
    ]
    # Warmup-path init plus at most the first reinit (different input
    # sharding); with the per-batch rebuild this was one compile per batch.
    assert len(init_compiles) <= 3, init_compiles


# ---------------------------------------------------------------------------
# overdispersed-chain-starts
# ---------------------------------------------------------------------------


def test_jitter_positions_distinct_and_deterministic():
    from gholax.sampler import NUTS

    sampler = _nuts({})
    key = jax.random.key(0)
    position = jnp.array([1.0, -2.0])
    imm = jnp.array([0.25, 4.0])

    starts = sampler._jitter_positions(key, position, imm, 4)
    assert starts.shape == (4, 2)
    assert len(np.unique(np.asarray(starts), axis=0)) == 4
    # Deterministic given the key.
    assert np.array_equal(
        np.asarray(sampler._jitter_positions(key, position, imm, 4)),
        np.asarray(starts),
    )
    # Jitter scales with sqrt(inverse mass matrix).
    dev = np.asarray(starts) - np.asarray(position)[None, :]
    assert np.max(np.abs(dev[:, 0])) < np.max(np.abs(dev[:, 1]))


@needs_chains
def test_post_warmup_starts_distinct_and_rhat_sane(tmp_path):
    from gholax.sampler import NUTS

    cfg = {
        "sampler": {
            "NUTS": {
                "n_steps_warmup": 100,
                "n_steps_min": 20,
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "minimize_and_sample": False,
            }
        }
    }
    prefix = str(tmp_path / "nuts")
    samples, _ = NUTS(cfg).run(ToyModel(), prefix)

    with open(f"{prefix}.nuts_warmup_parameters.json") as fp:
        warmup = json.load(fp)
    starts = np.array(warmup["initial_state"])
    assert starts.shape[0] == N_DEVICES
    assert len(np.unique(starts, axis=0)) == N_DEVICES

    # R-hat converged on the known Gaussian target and samples stayed sane.
    samples = np.asarray(samples)
    assert np.all(np.isfinite(samples))
    assert np.max(np.abs(samples[:, :, :N_PARAMS])) < 10


# ---------------------------------------------------------------------------
# hessian-init-vmap
# ---------------------------------------------------------------------------


def test_hessian_mass_matrix_matches_sequential():
    from gholax.sampler import NUTS

    dim = 20
    curv = jnp.arange(1.0, dim + 1)
    jlp = jax.jit(lambda p: -0.5 * jnp.sum(curv * p**2))
    position = 0.1 * jnp.ones(dim)
    sampler = _nuts({})

    t0 = time.perf_counter()
    imm = sampler._hessian_mass_matrix(jlp, position)
    imm.block_until_ready()
    t_batched = time.perf_counter() - t0

    # Sequential reference (the previous implementation).
    jnlp = lambda p: -jlp(p)
    grad_fn = jax.grad(jnlp)
    eps = 1e-3
    g0 = grad_fn(position)
    t0 = time.perf_counter()
    diag_H = jnp.array([
        (grad_fn(position.at[i].set(position[i] + eps))[i] - g0[i]) / eps
        for i in range(dim)
    ])
    imm_seq = 1.0 / jnp.clip(diag_H, 1e-6, 1e6)
    imm_seq.block_until_ready()
    t_seq = time.perf_counter() - t0

    print(f"hessian imm: batched {t_batched:.3f}s vs sequential {t_seq:.3f}s")
    assert np.allclose(np.asarray(imm), np.asarray(imm_seq), rtol=1e-5, atol=1e-7)
    assert np.allclose(np.asarray(imm), 1.0 / np.asarray(curv), rtol=1e-2)


# ---------------------------------------------------------------------------
# rhat-latter-half
# ---------------------------------------------------------------------------


@needs_chains
def test_rhat_latter_half_logged(tmp_path, capsys):
    from gholax.sampler import NUTS

    cfg = {
        "sampler": {
            "NUTS": {
                "n_steps_warmup": 100,
                "n_steps_min": 20,
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "minimize_and_sample": False,
            }
        }
    }
    samples, _ = NUTS(cfg).run(ToyModel(), str(tmp_path / "nuts"))
    out = capsys.readouterr().out
    assert "rhat - 1 (latter half)" in out
    # The run stopped: retained-half R-hat reached the target on this toy
    # Gaussian at or after n_steps_min.
    assert np.asarray(samples).shape[1] >= 20


# ---------------------------------------------------------------------------
# vmap-chains-per-device
# ---------------------------------------------------------------------------


@needs_chains
def test_map_chains_flat_batching():
    """K>1 _map_chains keeps a flat chain axis with row order preserved."""
    sampler = _nuts({"chains_per_device": 3})
    n_total = 3 * N_DEVICES
    x = jnp.arange(n_total * 2, dtype=jnp.float32).reshape(n_total, 2)
    out = sampler._map_chains(lambda p: p * 2.0)(x)
    assert out.shape == (n_total, 2)
    assert np.allclose(np.asarray(out), np.asarray(x) * 2.0)


@needs_chains
def test_nuts_chains_per_device_shapes_and_rhat(tmp_path, capsys):
    from gholax.sampler import NUTS

    cfg = {
        "sampler": {
            "NUTS": {
                "n_steps_warmup": 100,
                "n_steps_min": 20,
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "minimize_and_sample": False,
                "chains_per_device": 2,
            }
        }
    }
    n_total = 2 * N_DEVICES
    prefix = str(tmp_path / "nuts_k2")
    samples, param_names = NUTS(cfg).run(ToyModel(), prefix)

    samples = np.asarray(samples)
    assert samples.shape[0] == n_total
    assert samples.shape[2] == N_PARAMS + 1
    assert np.all(np.isfinite(samples))

    chk = np.load(f"{prefix}.samples_chk.npy")
    lp = np.load(f"{prefix}.logposterior_chk.npy")
    assert chk.shape == (n_total, samples.shape[1], N_PARAMS)
    assert lp.shape == (n_total, samples.shape[1])

    with open(f"{prefix}.nuts_warmup_parameters.json") as fp:
        warmup = json.load(fp)
    starts = np.array(warmup["initial_state"])
    assert starts.shape[0] == n_total
    assert len(np.unique(starts, axis=0)) == n_total

    # All n_total chains are independent trajectories entering the R-hat.
    assert len(np.unique(chk[:, -1, :], axis=0)) == n_total
    assert "rhat - 1 (latter half)" in capsys.readouterr().out


@needs_chains
def test_nuts_k2_matches_k1_posterior_moments(tmp_path):
    from gholax.sampler import NUTS

    base = {
        "n_steps_warmup": 200,
        "n_steps_min": 400,
        "n_steps_incr": 100,
        "target_r_minus_one": 0.1,
        "minimize_and_sample": False,
    }

    def run(k, name):
        scfg = dict(base, chains_per_device=k)
        samples, _ = NUTS({"sampler": {"NUTS": scfg}}).run(
            ToyModel(), str(tmp_path / name)
        )
        s = np.asarray(samples)[:, :, :N_PARAMS]
        s = s[:, s.shape[1] // 2 :, :].reshape(-1, N_PARAMS)
        return s.mean(axis=0), s.std(axis=0)

    m1, s1 = run(1, "k1")
    m2, s2 = run(2, "k2")

    # Unit Gaussian target: both runs recover the moments and agree.
    assert np.all(np.abs(m1) < 0.15) and np.all(np.abs(m2) < 0.15)
    assert np.all(np.abs(s1 - 1.0) < 0.15) and np.all(np.abs(s2 - 1.0) < 0.15)
    assert np.all(np.abs(m2 - m1) < 0.2)
    assert np.all(np.abs(s2 - s1) < 0.2)


@needs_chains
def test_mclmc_chains_per_device_restart_path(tmp_path):
    """MCLMC K>1 via the restart path (skips the blackjax-version-dependent
    adaptation): total chains, checkpoint shapes, and chain independence."""
    from gholax.sampler import MCLMC

    n_total = 2 * N_DEVICES
    prefix = str(tmp_path / "mclmc_k2")
    warmup = {
        "L": 3.0,
        "step_size": 0.3,
        "inverse_mass_matrix": [1.0] * N_PARAMS,
        "initial_state": (0.1 * np.arange(n_total * N_PARAMS))
        .reshape(n_total, N_PARAMS)
        .tolist(),
        "adjusted": False,
    }
    with open(f"{prefix}.mclmc_warmup_parameters.json", "w") as fp:
        json.dump(warmup, fp)

    cfg = {
        "restart": True,
        "n_steps_min": 20,
        "n_steps_incr": 10,
        "target_r_minus_one": 0.5,
        "chains_per_device": 2,
        "minimize_and_sample": False,
    }
    samples, _ = MCLMC({"sampler": {"MCLMC": cfg}}).run(ToyModel(), prefix)
    samples = np.asarray(samples)
    assert samples.shape[0] == n_total
    assert np.all(np.isfinite(samples))
    chk = np.load(f"{prefix}.samples_chk.npy")
    assert chk.shape[0] == n_total
    assert len(np.unique(chk[:, -1, :], axis=0)) == n_total


@needs_chains
def test_nuts_mesh_chains_per_device(tmp_path):
    """Mesh mode with chains_per_device=2: K chains vmapped inside each
    chain group's shard_map body."""
    from gholax.sampler import NUTS

    n_groups = N_DEVICES // 2
    cfg = {
        "sampler": {
            "NUTS": {
                "n_chains": n_groups,
                "model_shards": 2,
                "chains_per_device": 2,
                "n_steps_warmup": 50,
                "n_steps_min": 20,
                "n_steps_incr": 10,
                "target_r_minus_one": 0.5,
                "minimize_and_sample": False,
            }
        }
    }
    n_total = n_groups * 2
    prefix = str(tmp_path / "mesh_k2")
    samples, _ = NUTS(cfg).run(MeshToyModel(), prefix)
    samples = np.asarray(samples)
    assert samples.shape[0] == n_total
    assert np.all(np.isfinite(samples))
    chk = np.load(f"{prefix}.samples_chk.npy")
    assert chk.shape[0] == n_total
    assert len(np.unique(chk[:, -1, :], axis=0)) == n_total


# ---------------------------------------------------------------------------
# warmup-warm-start
# ---------------------------------------------------------------------------


def test_nuts_warm_start_fewer_stages(tmp_path, capsys):
    jlp = _toy_jlp()
    pos0 = jnp.zeros(N_PARAMS)

    # Loose rtols: per-stage mass estimates on this toy target fluctuate at
    # the tens-of-percent level, and the mechanism under test is the stage
    # count, not the adaptation accuracy.
    cfg = {
        **ADAPTIVE_CFG,
        "adaptive_warmup_stage_steps": 200,
        "adaptive_warmup_min_steps": 400,
        "adaptive_warmup_max_steps": 1600,
        "adaptive_warmup_rtol_mass": 0.75,
        "adaptive_warmup_rtol_step": 0.75,
    }

    cold = _nuts(cfg)
    _, params_c = cold._adaptive_window_warmup(jlp, jax.random.key(0), pos0)
    out = capsys.readouterr().out
    m = re.search(r"Warmup converged after (\d+) steps", out)
    assert m, out
    cold_steps = int(m.group(1))
    assert cold_steps >= 400

    # Warm start from the cold run's parameters: min_steps drops to one stage
    # and the convergence check is seeded with the provided parameters.
    warm = _nuts({**cfg, "adaptive_warmup_min_steps": 200})
    _, params_w = warm._adaptive_window_warmup(
        jlp,
        jax.random.key(1),
        pos0,
        initial_inverse_mass_matrix=jnp.asarray(params_c["inverse_mass_matrix"]),
        initial_step_size=float(params_c["step_size"]),
    )
    out = capsys.readouterr().out
    m = re.search(r"Warmup converged after (\d+) steps", out)
    assert m, out
    warm_steps = int(m.group(1))
    assert warm_steps < cold_steps

    # Equivalent parameters: convergence was tested against the warm-start
    # values, so the result stays within the configured rtols of them.
    mass_c = np.asarray(params_c["inverse_mass_matrix"])
    mass_w = np.asarray(params_w["inverse_mass_matrix"])
    assert np.all(np.abs(mass_w - mass_c) / np.abs(mass_c) < 1.0)
    assert 0.25 < float(params_w["step_size"]) / float(params_c["step_size"]) < 4.0


@needs_chains
def test_nuts_warm_start_file_wiring(tmp_path, capsys):
    from gholax.sampler import NUTS

    scfg = {
        **ADAPTIVE_CFG,
        "adaptive_warmup_min_steps": 50,
        "n_steps_min": 20,
        "n_steps_incr": 10,
        "target_r_minus_one": 0.5,
    }
    prefix = str(tmp_path / "cold")
    NUTS({"sampler": {"NUTS": dict(scfg)}}).run(ToyModel(), prefix)
    warm_file = f"{prefix}.nuts_warmup_parameters.json"
    assert os.path.exists(warm_file)

    scfg["warmup_init_file"] = warm_file
    prefix_w = str(tmp_path / "warm")
    samples, _ = NUTS({"sampler": {"NUTS": dict(scfg)}}).run(ToyModel(), prefix_w)
    out = capsys.readouterr().out
    assert f"Warm-starting adaptation from {warm_file}" in out
    assert np.all(np.isfinite(np.asarray(samples)))


def test_mclmc_warm_start_fewer_rounds(tmp_path, monkeypatch, capsys):
    import blackjax
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    from gholax.sampler import MCLMC

    target = MCLMCAdaptationState(
        L=jnp.asarray(2.0),
        step_size=jnp.asarray(0.1),
        inverse_mass_matrix=jnp.array([4.0, 9.0]),
    )
    monkeypatch.setattr(MCLMC, "_adapt_unadjusted", _halfway_adapt_stub(target))

    jlp = _toy_jlp()
    state0 = blackjax.mcmc.mclmc.init(
        position=jnp.zeros(2), logdensity_fn=jlp, rng_key=jax.random.key(0)
    )
    cfg = {"warmup_tolerance": 0.05, "max_warmup_rounds": 10}

    cold = _mclmc(cfg)
    params0 = MCLMCAdaptationState(
        L=jnp.asarray(1.0),
        step_size=jnp.asarray(0.01),
        inverse_mass_matrix=jnp.ones(2),
    )
    cold._adapt_with_convergence(
        jlp, state0, jax.random.key(1), params0, output_file=str(tmp_path / "cold")
    )
    out = capsys.readouterr().out
    m = re.search(r"converged after (\d+) rounds", out)
    assert m, out
    cold_rounds = int(m.group(1))
    assert cold_rounds > 1

    # Warm start at the fixed point: converges on the first round.
    warm = _mclmc(cfg)
    warm._adapt_with_convergence(
        jlp, state0, jax.random.key(2), target,
        output_file=str(tmp_path / "warm"), warm_start=True,
    )
    out = capsys.readouterr().out
    m = re.search(r"converged after (\d+) rounds", out)
    assert m, out
    assert int(m.group(1)) < cold_rounds
