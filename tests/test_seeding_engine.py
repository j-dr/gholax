"""The seeding engine standing alone: no sampler, no model."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gholax.sampler.seeding import (
    FisherParams,
    Seeder,
    SeedingConfig,
    SeedingHost,
    SeedingRequest,
    proposal_covariance_from_metric,
)

DIM = 3
SCALES = jnp.array([4.0, 1.0, 0.25])


def _jlp():
    return jax.jit(lambda x: -0.5 * jnp.sum(x**2 / SCALES))


def _seeder(host=None, **cfg):
    return Seeder(
        SeedingConfig(**cfg),
        host or SeedingHost(),
        fisher_params=FisherParams(),
    )


def _starts(n=4, spread=1.0, seed=0):
    key = jax.random.key(seed)
    return jnp.array([2.0, -1.5, 0.5]) + spread * jax.random.normal(
        key, (n, DIM)
    )


def test_best_fit_position_finds_the_map():
    x = _seeder().best_fit_position(_jlp(), jnp.array([3.0, -2.0, 1.0]))
    assert np.allclose(np.asarray(x), 0.0, atol=1e-3)


def test_minimize_per_chain_keeps_every_chain(tmp_path):
    prefix = str(tmp_path / "min")
    seeder = _seeder()
    pos = seeder.minimize(_jlp(), _starts(), 4, prefix, mode="per_chain")
    assert pos.shape == (4, DIM)
    assert np.allclose(np.asarray(pos), 0.0, atol=1e-3)
    res = json.load(open(prefix + ".minimization_results.json"))
    assert set(res) == {"x_opt", "value"}  # no constrain on this host
    assert np.asarray(res["x_opt"]).shape == (4, DIM)


def test_minimize_writes_physical_positions_when_the_prior_constrains(tmp_path):
    prefix = str(tmp_path / "min")
    host = SeedingHost(constrain=lambda y: 2.0 * y)
    seeder = _seeder(host=host)
    seeder.minimize(_jlp(), _starts(), 4, prefix, mode="per_chain")
    res = json.load(open(prefix + ".minimization_results.json"))
    assert np.allclose(
        np.asarray(res["x_opt_physical"]), 2 * np.asarray(res["x_opt"])
    )


def test_minimize_auto_tiles_identical_starts(tmp_path):
    """The identical-starts branch is a value test, not a config flag."""
    seeder = _seeder()
    same = jnp.tile(jnp.array([2.0, -1.5, 0.5]), (4, 1))
    pos = np.asarray(
        seeder.minimize(_jlp(), same, 4, str(tmp_path / "a"), mode="auto")
    )
    assert np.array_equal(pos, np.tile(pos[0], (4, 1)))

    distinct = np.asarray(
        seeder.minimize(_jlp(), _starts(), 4, str(tmp_path / "b"), mode="auto")
    )
    assert distinct.shape == (4, DIM)


@pytest.mark.parametrize(
    "kind", ["ones", "hessian", "hessian_dense", "fisher_seeds", "mclmc"]
)
def test_initial_metric_ladder(kind, tmp_path):
    """Every metric the ladder can produce, on one anisotropic Gaussian."""
    jlp = _jlp()
    cfg = dict(mass_matrix_init=kind, metric_fallback="ones",
               minimize_and_sample=True)
    if kind == "fisher_seeds":
        cfg["pathfinder_init"] = True
    if kind == "mclmc":
        cfg.update(mclmc_mm_n_tune_steps=100, mclmc_mm_n_steps=200,
                   mclmc_mm_n_chains=4)
    seeder = _seeder(**cfg)
    key = jax.random.key(3)
    draws = jax.random.normal(key, (64, DIM)) * jnp.sqrt(SCALES)
    imm, info = seeder.initial_metric(jlp, jnp.zeros(DIM), draws, key)
    imm = np.asarray(imm)
    assert info["kind"] == kind
    if kind == "ones":
        assert np.allclose(imm, 1.0)
        return
    diag = np.diag(imm) if imm.ndim == 2 else imm
    assert np.all(diag > 0) and np.all(np.isfinite(diag))
    # the metric must order the parameters like the target's scales
    assert np.argmax(diag) == 0 and np.argmin(diag) == DIM - 1


def test_fisher_seeds_without_params_raises():
    seeder = Seeder(
        SeedingConfig(mass_matrix_init="fisher_seeds", pathfinder_init=True),
        SeedingHost(),
    )
    with pytest.raises(ValueError, match="fisher_params"):
        seeder.initial_metric(_jlp(), jnp.zeros(DIM), _starts(8), None)


def test_pathfinder_metric_requires_a_pathfinder_run():
    seeder = _seeder(mass_matrix_init="pathfinder", pathfinder_init=True)
    with pytest.raises(Exception):
        seeder.initial_metric(_jlp(), jnp.zeros(DIM), _starts(8), None)


def test_pathfinder_seeds_a_dispersed_cloud(tmp_path):
    seeder = _seeder(pathfinder_init=True, pathfinder_n_paths=3,
                     pathfinder_elbo_samples=10)
    pos = np.asarray(
        seeder.pathfinder_init(
            _jlp(), jnp.array([2.0, -1.5, 0.5]), 16, jax.random.key(2),
            str(tmp_path / "pf"),
        )
    )
    assert pos.shape == (16, DIM) and np.isfinite(pos).all()
    assert len(np.unique(pos, axis=0)) > 8
    imm = np.asarray(seeder.pathfinder_imm)
    assert imm.shape == (DIM, DIM)
    assert np.argmax(np.diag(imm)) == 0
    info = json.load(open(str(tmp_path / "pf") + ".pathfinder_init.json"))
    assert len(info["elbo"]) == 3 and info["resample"] is False


def test_run_pipeline_returns_positions_and_metric(tmp_path):
    seeder = _seeder(minimize_and_sample=True, mass_matrix_init="hessian_dense")
    res = seeder.run(
        SeedingRequest(
            _jlp(), _jlp(), _starts(), 4, rng_key=jax.random.key(0),
            output_file=str(tmp_path / "run"),
        )
    )
    assert res.positions.shape == (4, DIM)
    assert np.asarray(res.inverse_mass_matrix).shape == (DIM, DIM)
    assert np.allclose(np.asarray(res.map_position), 0.0, atol=1e-3)
    assert res.values is not None


def test_run_can_skip_both_stages(tmp_path):
    """A warmup checkpoint makes the MAP and the metric pure waste."""
    starts = _starts()
    res = _seeder(minimize_and_sample=True).run(
        SeedingRequest(_jlp(), _jlp(), starts, 4, rng_key=jax.random.key(0),
                       skip_minimize=True, skip_metric=True)
    )
    assert np.array_equal(np.asarray(res.positions), np.asarray(starts))
    assert res.inverse_mass_matrix is None


def test_proposal_covariance_caps_at_the_prior_width():
    """A metric wider than the prior is clipped; correlations survive.

    The cap acts on the conditional sigmas (1/sqrt(diag(precision))), not on
    the marginal ones, so the returned diagonal can still exceed 1 through
    the variance inflation factor of the correlation matrix - that is MH's
    existing semantics, kept verbatim.
    """
    imm = jnp.array([[9.0, 1.0], [1.0, 0.25]])  # SPD, wider than the prior
    cov = np.asarray(proposal_covariance_from_metric(imm))
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    # direction 0 is wider than the prior and gets clipped; direction 1 is
    # already narrow and is passed through
    assert cov[0, 0] < 9.0
    assert np.isclose(cov[1, 1], 0.25, rtol=1e-3)
    assert cov[0, 1] > 0  # correlation sign preserved
