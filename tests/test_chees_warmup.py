"""Tests for the pooled cross-chain ChEES warmup (warmup_algorithm='chees').

Uses toy Gaussian targets only; CPU-runnable. Run standalone to force
4 host devices for the chain-parallel tests.
"""

import json
import os
import sys

if "jax" not in sys.modules:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N_DEVICES = jax.local_device_count()
DIM = 24

needs_chains = pytest.mark.skipif(
    N_DEVICES < 2,
    reason="R-hat convergence loop requires >= 2 chains (JAX devices); "
    "run this file standalone to force 4 host devices",
)

# Anisotropic correlated Gaussian: AR(1) correlation, variances in [0.5, 2].
_SIG = np.sqrt(np.linspace(0.5, 2.0, DIM))
_COV = 0.3 ** np.abs(np.subtract.outer(np.arange(DIM), np.arange(DIM)))
_COV = _SIG[:, None] * _COV * _SIG[None, :]
_PREC = jnp.asarray(np.linalg.inv(_COV), dtype=jnp.float32)
TRUE_VAR = np.diag(_COV)


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
        return -0.5 * p @ _PREC @ p


def _nuts(scfg):
    from gholax.sampler import NUTS

    return NUTS({"sampler": {"NUTS": dict(scfg)}})


CHEES_CFG = {
    "warmup_algorithm": "chees",
    "chees_warmup_steps": 150,
    "n_steps_min": 400,
    "n_steps_incr": 100,
    "target_r_minus_one": 0.1,
    "minimize_and_sample": False,
    "chains_per_device": 2,
}


def test_chees_warmup_algorithm_validated():
    assert _nuts({"warmup_algorithm": "chees"}).warmup_algorithm == "chees"
    with pytest.raises(ValueError, match="warmup_algorithm"):
        _nuts({"warmup_algorithm": "cheese"})


def test_chees_mesh_mode_raises(tmp_path):
    cfg = {
        "warmup_algorithm": "chees",
        "n_chains": N_DEVICES,
        "model_shards": 1,
        "minimize_and_sample": False,
    }
    with pytest.raises(ValueError, match="chees.*not.*supported in mesh mode"):
        _nuts(cfg).run(ToyModel(), str(tmp_path / "mesh_chees"))


def test_chees_kernel_matches_blackjax_parameters():
    """The serializable-scalar kernel rebuild must reproduce the kernel
    blackjax's chees_adaptation returns, step for step."""
    import blackjax
    import optax

    jlp = jax.jit(lambda p: -0.5 * p @ _PREC @ p)
    n_chains, num_steps, max_sampling_steps = 4, 32, 2**15
    positions = jax.random.normal(jax.random.key(0), (n_chains, DIM)) * 0.1

    warmup = blackjax.chees_adaptation(
        jlp, num_chains=n_chains, target_acceptance_rate=0.65
    )
    (last_states, parameters), info = warmup.run(
        jax.random.key(1), positions, 0.05, optax.adam(0.25), num_steps,
        max_sampling_steps=max_sampling_steps,
    )
    ad = info.adaptation_state
    tla = float(
        jnp.exp(
            ad.log_trajectory_length_moving_average[-1]
            - ad.log_step_size_moving_average[-1]
        )
    )
    bits = int(np.ceil(np.log2(num_steps + max_sampling_steps)))

    ref = blackjax.dynamic_hmc(jlp, **parameters)
    mine = _nuts({})._chees_kernel(
        jlp, parameters["step_size"], parameters["inverse_mass_matrix"],
        tla, bits,
    )

    rga = jnp.asarray(num_steps, dtype=jnp.int32)
    s_ref = ref.init(positions[0], rga)
    s_mine = mine.init(positions[0], rga)
    for i in range(5):
        k = jax.random.key(100 + i)
        s_ref, _ = ref.step(k, s_ref)
        s_mine, _ = mine.step(k, s_mine)
        assert np.array_equal(np.asarray(s_ref.position), np.asarray(s_mine.position))
        assert int(s_ref.random_generator_arg) == int(s_mine.random_generator_arg)


@needs_chains
def test_chees_end_to_end_convergence_and_moments(tmp_path, capsys):
    """8 chains, 150 sequential ChEES warmup steps (vs the 500-step window
    default): downstream sampling must converge with sane moments."""
    n_total = 2 * N_DEVICES
    prefix = str(tmp_path / "chees")
    samples, param_names = _nuts(CHEES_CFG).run(ToyModel(), prefix)
    out = capsys.readouterr().out
    assert "Running ChEES warmup (150 steps, 8 chains)" in out
    assert "rhat - 1 (latter half)" in out

    with open(f"{prefix}.nuts_warmup_parameters.json") as fp:
        warmup = json.load(fp)
    assert {"trajectory_length_adjusted", "halton_max_bits",
            "random_generator_arg"} <= set(warmup)
    starts = np.array(warmup["initial_state"])
    assert starts.shape == (n_total, DIM)
    assert len(np.unique(starts, axis=0)) == n_total

    samples = np.asarray(samples)
    assert samples.shape[0] == n_total
    assert samples.shape[2] == DIM + 1
    assert np.all(np.isfinite(samples))

    # Latter half against the known target moments.
    half = samples[:, samples.shape[1] // 2 :, :DIM].reshape(-1, DIM)
    assert np.max(np.abs(half.mean(axis=0)) / np.sqrt(TRUE_VAR)) < 0.35
    assert np.max(np.abs(half.var(axis=0) - TRUE_VAR) / TRUE_VAR) < 0.5


@needs_chains
def test_chees_restart_from_warmup_parameters(tmp_path, capsys):
    """Restart must rebuild the tuned dynamic-HMC kernel from the JSON
    without re-running adaptation."""
    prefix = str(tmp_path / "chees_restart")
    cfg = {**CHEES_CFG, "n_steps_min": 100}
    _nuts(cfg).run(ToyModel(), prefix)
    capsys.readouterr()

    # Exercise the warmup-parameters restart path (no samples checkpoint).
    os.remove(f"{prefix}.samples_chk.npy")
    os.remove(f"{prefix}.logposterior_chk.npy")
    samples, _ = _nuts({**cfg, "restart": True}).run(ToyModel(), prefix)
    out = capsys.readouterr().out
    assert "Running ChEES warmup" not in out
    samples = np.asarray(samples)
    assert samples.shape[0] == 2 * N_DEVICES
    assert np.all(np.isfinite(samples))
