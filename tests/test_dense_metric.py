"""Dense (full-Hessian) inverse mass matrix: estimation and frozen use in
pooled-window warmup. CPU-runnable toy correlated Gaussians."""

import os
import sys

if "jax" not in sys.modules:
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N_DEVICES = jax.local_device_count()
DIM = 8

needs_chains = pytest.mark.skipif(N_DEVICES < 2, reason="needs >= 2 devices")


def _cov():
    rho, sig = 0.9, np.geomspace(0.5, 2.0, DIM)
    R = np.eye(DIM)
    for i in range(0, DIM - 1, 2):
        R[i, i + 1] = R[i + 1, i] = rho
    return sig[:, None] * R * sig[None, :]


def make_model():
    prec = jnp.asarray(np.linalg.inv(_cov()), dtype=jnp.float32)

    class ToyPrior:
        def __init__(self):
            self.params = [f"x{i}" for i in range(DIM)]

        def get_prior_sigmas(self):
            return np.ones(DIM)

        def get_reference_values(self):
            return np.zeros(DIM)

        def initial_position(self, random_start=True, key=None, normalize=True):
            vals = (jax.random.normal(key, (DIM,)) * 0.1 + 0.5
                    if random_start and key is not None else jnp.full((DIM,), 0.5))
            return {f"x{i}": vals[i] for i in range(DIM)}

    class ToyModel:
        def __init__(self):
            self.prior = ToyPrior()

        def log_posterior_scaled_params(self, p):
            return -0.5 * p @ prec @ p

    return ToyModel()


def _nuts(extra=None):
    from gholax.sampler import NUTS

    cfg = {"warmup_algorithm": "pooled_window", "minimize_and_sample": False,
           "chains_per_device": 2, "pooled_window_steps": 5,
           "pooled_window_max_steps": 40, "pooled_window_max_window": 20,
           "pooled_window_allow_unconverged": True}
    cfg.update(extra or {})
    return NUTS({"sampler": {"NUTS": cfg}})


def test_dense_hessian_recovers_covariance():
    model = make_model()
    nuts = _nuts()
    setup = nuts._init_chains(model)
    imm = np.asarray(
        nuts._hessian_mass_matrix_dense(setup.jlp, jnp.zeros(DIM))
    )
    assert imm.shape == (DIM, DIM)
    assert np.allclose(imm, imm.T, atol=1e-5)
    assert np.all(np.linalg.eigvalsh(imm) > 0)
    # For a Gaussian, inverse Hessian == covariance.
    assert np.allclose(imm, _cov(), rtol=0.05, atol=1e-3)


def test_dense_eigen_guard_indefinite():
    """A saddle (one negative curvature direction) must still yield SPD."""
    from gholax.sampler import NUTS

    nuts = _nuts()
    sign = jnp.asarray([-1.0] + [1.0] * (DIM - 1))
    jlp = jax.jit(lambda p: -0.5 * jnp.sum(sign * p**2))
    imm = np.asarray(nuts._hessian_mass_matrix_dense(jlp, jnp.zeros(DIM)))
    assert np.all(np.linalg.eigvalsh(imm) > 0)


@needs_chains
def test_pooled_warmup_frozen_dense_metric(capsys):
    """Warmup with a dense initial imm and dense updates off keeps the
    metric frozen (2D, unchanged) and returns a finite tuned step size."""
    model = make_model()
    nuts = _nuts({"pooled_window_dense_update": False})
    setup = nuts._init_chains(model)
    dense = nuts._hessian_mass_matrix_dense(setup.jlp, jnp.zeros(DIM))
    _, params = nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=dense, initial_step_size=0.5,
    )
    imm = np.asarray(params["inverse_mass_matrix"])
    assert imm.ndim == 2
    assert np.allclose(imm, np.asarray(dense))
    assert np.isfinite(float(np.asarray(params["step_size"])))


@needs_chains
def test_dense_covariance_update_on_convergence(capsys):
    """On convergence a dense metric is swapped for the pooled tail
    covariance (still SPD, close to the true covariance for a Gaussian)
    and eps is re-seeded."""
    model = make_model()
    nuts = _nuts({"pooled_window_min_tail_steps": 16,
                  "pooled_window_mixing_rhat": 2.0,
                  "adaptive_warmup_rtol_mass": 1e9,
                  "adaptive_warmup_rtol_step": 1e9,
                  "pooled_window_consecutive_windows": 1})
    setup = nuts._init_chains(model)
    dense = nuts._hessian_mass_matrix_dense(setup.jlp, jnp.zeros(DIM))
    _, params = nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=dense, initial_step_size=0.5,
    )
    out = capsys.readouterr().out
    assert "Dense metric updated from pooled tail covariance" in out
    assert "Step size re-seeded" in out
    imm = np.asarray(params["inverse_mass_matrix"])
    assert imm.ndim == 2
    assert not np.allclose(imm, np.asarray(dense))
    assert np.all(np.linalg.eigvalsh(imm) > 0)
    # Pooled covariance of a mixed Gaussian ~ true covariance (loose: short
    # tails and shrinkage).
    scale = np.sqrt(np.diag(_cov()))
    corr_true = _cov() / np.outer(scale, scale)
    s = np.sqrt(np.diag(imm))
    corr_est = imm / np.outer(s, s)
    assert np.max(np.abs(corr_est - corr_true)) < 0.35
    assert np.isfinite(float(np.asarray(params["step_size"])))


def test_dense_update_every_window(capsys):
    """With pooled_window_dense_update a dense metric is re-estimated from
    the tail at every boundary with a long-enough tail, not only at
    convergence, starting from a deliberately wrong (identity) metric."""
    model = make_model()
    nuts = _nuts({"pooled_window_dense_update": True,
                  "pooled_window_min_tail_steps": 16,
                  "pooled_window_mixing_rhat": 1e9,
                  "pooled_window_max_steps": 100,
                  "pooled_window_max_window": 20,
                  "pooled_window_consecutive_windows": 1e9})
    setup = nuts._init_chains(model)
    _, params = nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.eye(DIM), initial_step_size=0.5,
    )
    out = capsys.readouterr().out
    assert out.count("Dense metric updated from pooled tail covariance") >= 2
    imm = np.asarray(params["inverse_mass_matrix"])
    assert imm.ndim == 2 and not np.allclose(imm, np.eye(DIM))
    assert np.all(np.linalg.eigvalsh(imm) > 0)
    scale = np.sqrt(np.diag(_cov()))
    corr_true = _cov() / np.outer(scale, scale)
    s = np.sqrt(np.diag(imm))
    assert np.max(np.abs(imm / np.outer(s, s) - corr_true)) < 0.35


def test_dense_update_gate_converges(capsys):
    """Per-window dense updates must not block the rtol_mass stability gate
    (relative Frobenius change, not element-wise)."""
    model = make_model()
    nuts = _nuts({"pooled_window_dense_update": True,
                  "pooled_window_min_tail_steps": 16,
                  "pooled_window_mixing_rhat": 2.0,
                  "pooled_window_max_steps": 200,
                  "pooled_window_max_window": 20,
                  # toy windows are 128 samples: variance noise ~0.3
                  "adaptive_warmup_rtol_mass": 1.0,
                  "pooled_window_consecutive_windows": 1})
    setup = nuts._init_chains(model)
    nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.eye(DIM), initial_step_size=0.5,
    )
    assert "Pooled warmup converged" in capsys.readouterr().out


@needs_chains
def test_dense_update_skips_underdetermined_tail(capsys):
    """Per-window dense updates are skipped (with a warning) while the pooled
    tail cannot overdetermine the covariance."""
    model = make_model()
    nuts = _nuts({"pooled_window_dense_update": True,
                  "chains_per_device": 1,
                  "pooled_window_steps": 2,
                  "pooled_window_min_tail_steps": 2,
                  "pooled_window_max_window": 8,
                  "pooled_window_max_steps": 14})
    setup = nuts._init_chains(model)
    nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.eye(DIM), initial_step_size=0.5,
    )
    out = capsys.readouterr().out
    # First 2-step boundary is underdetermined (4 chains x 2 <= 8 params);
    # grown windows then satisfy the gate and update.
    assert "Skipping dense metric update" in out
    assert "Dense metric updated" in out


@needs_chains
def test_dense_update_rejects_hopeless_config():
    """Fail fast when even the longest window cannot overdetermine the
    covariance."""
    model = make_model()
    nuts = _nuts({"pooled_window_dense_update": True,
                  "chains_per_device": 1,
                  "pooled_window_steps": 2,
                  "pooled_window_min_tail_steps": 2,
                  "pooled_window_max_window": 2,
                  "pooled_window_max_steps": 8})
    setup = nuts._init_chains(model)
    with pytest.raises(ValueError, match="can never run"):
        nuts._pooled_window_warmup(
            setup.jlp, jax.random.key(0), setup.initial_positions,
            initial_inverse_mass_matrix=jnp.eye(DIM), initial_step_size=0.5,
        )


def test_dense_update_low_rank_recovers_structure(capsys):
    """pooled_window_dense_rank: auto keeps only correlation directions above
    the Marchenko-Pastur edge; on a Gaussian target the recovered metric is
    still close to the true covariance and reports a small rank."""
    model = make_model()
    nuts = _nuts({"pooled_window_dense_update": True,
                  "pooled_window_dense_rank": "auto",
                  "pooled_window_min_tail_steps": 16,
                  "pooled_window_mixing_rhat": 1e9,
                  "pooled_window_max_steps": 100,
                  "pooled_window_max_window": 20,
                  "pooled_window_consecutive_windows": 1e9})
    setup = nuts._init_chains(model)
    _, params = nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.eye(DIM), initial_step_size=0.5,
    )
    out = capsys.readouterr().out
    assert "low-rank metric: kept" in out
    ranks = [int(l.split("kept ")[1].split()[0]) for l in out.splitlines()
             if "low-rank metric" in l]
    assert max(ranks) >= 1
    imm = np.asarray(params["inverse_mass_matrix"])
    assert imm.ndim == 2 and np.all(np.linalg.eigvalsh(imm) > 0)
    scale = np.sqrt(np.diag(_cov()))
    corr_true = _cov() / np.outer(scale, scale)
    s = np.sqrt(np.diag(imm))
    assert np.max(np.abs(imm / np.outer(s, s) - corr_true)) < 0.35
    with pytest.raises(ValueError):
        _nuts({"pooled_window_dense_rank": -1})


def _fisher_run(extra, capsys):
    model = make_model()
    nuts = _nuts({"pooled_window_dense_update": True,
                  "pooled_window_metric_estimator": "fisher",
                  "pooled_window_min_tail_steps": 16,
                  "pooled_window_mixing_rhat": 1e9,
                  "pooled_window_max_steps": 100,
                  "pooled_window_max_window": 20,
                  "pooled_window_consecutive_windows": 1e9, **extra})
    setup = nuts._init_chains(model)
    _, params = nuts._pooled_window_warmup(
        setup.jlp, jax.random.key(0), setup.initial_positions,
        initial_inverse_mass_matrix=jnp.eye(DIM), initial_step_size=0.5,
    )
    imm = np.asarray(params["inverse_mass_matrix"])
    assert imm.ndim == 2 and np.all(np.linalg.eigvalsh(imm) > 0)
    return imm, capsys.readouterr().out


@pytest.mark.parametrize("rank", [None, "auto", DIM])
def test_fisher_metric_recovers_gaussian_covariance(rank, capsys):
    """The Fisher-divergence estimator (dense / low-rank+diag) recovers the
    covariance of a Gaussian target from draws and scores."""
    imm, out = _fisher_run({"pooled_window_dense_rank": rank}, capsys)
    if rank is not None:
        assert "fisher low-rank metric: kept" in out
    scale = np.sqrt(np.diag(_cov()))
    corr_true = _cov() / np.outer(scale, scale)
    s = np.sqrt(np.diag(imm))
    assert np.max(np.abs(imm / np.outer(s, s) - corr_true)) < 0.35
    assert np.max(np.abs(np.log(s / scale))) < 0.5


def test_fisher_metric_unit_exact_on_gaussian_samples():
    """With exact Gaussian draws and scores the geometric mean returns the
    true covariance (dense) and its diagonal/low-rank form nearly so."""
    from gholax.sampler.nuts import _fisher_metric
    rng = np.random.default_rng(1)
    S = _cov(); P = np.linalg.inv(S)
    x = rng.multivariate_normal(np.zeros(DIM), S, size=20000)
    g = -x @ P                                 # score of N(0, S)
    C = np.cov(x.T); G = np.cov(g.T)
    dense = np.asarray(_fisher_metric(jnp.asarray(C), jnp.asarray(G), None, 1.5, 1e-8))
    assert np.allclose(dense, S, rtol=0.1, atol=0.05 * S.max())
    lr = np.asarray(_fisher_metric(jnp.asarray(C), jnp.asarray(G), "auto", 1.2, 1e-8))
    assert np.allclose(lr, S, rtol=0.2, atol=0.1 * S.max())


def test_fisher_metric_rank_deficient_seeds_is_spd():
    """n < dim draws (a Pathfinder seed cloud) must still give an SPD metric
    whose identified directions match the target and whose unidentified
    ones fall back to the diagonal scale."""
    from gholax.sampler.nuts import _fisher_metric
    rng = np.random.default_rng(2)
    d, n = 40, 16
    sd = np.exp(rng.normal(0, 1, d)); S = np.diag(sd**2); P = np.diag(1 / sd**2)
    x = rng.normal(size=(n, d)) * sd; g = -x @ P
    C = np.cov(x.T); G = np.cov(g.T)          # rank 15 each
    for rank in [None, "auto"]:
        M = np.asarray(_fisher_metric(jnp.asarray(C), jnp.asarray(G), rank, 1.5, 1e-4, n=n))
        ev = np.linalg.eigvalsh(M)
        assert np.all(ev > 0) and np.isfinite(M).all()
        assert np.max(np.abs(np.log(np.diag(M) / sd**2))) < 1.0
