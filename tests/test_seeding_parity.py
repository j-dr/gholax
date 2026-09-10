"""Golden values pinning the seeding stage across its extraction.

The numbers below were recorded from the pre-refactor implementation
(BaseSampler's _best_fit_position / _hessian_mass_matrix* / _pathfinder_init
and MetropolisHastings' bespoke init_covariance: hessian branch).  They are
what protects production runs: any change to an RNG split site, a jitter
scale, or a curvature guard moves them.

The MH proposal covariance is deliberately pinned too.  Redirecting
init_covariance: hessian onto the shared metric ladder is the one commit in
this series allowed to change it.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_pooled_window_warmup import make_correlated_gaussian_model

DIM = 4
RHO = 0.8
X0 = jnp.array([1.5, -2.0, 0.75, -0.5])

# L-BFGS stops shy of the exact optimum; the offset is the pinned quantity.
BEST_FIT = [
    -0.00026059383526444435,
    -0.0002929393667727709,
    -0.00027835601940751076,
    -2.7697766199707985e-05,
]

HESSIAN_DIAG = [
    0.35999995470046997,
    0.21951216459274292,
    0.2195122092962265,
    0.36000001430511475,
]


def _model():
    return make_correlated_gaussian_model(dim=DIM, rho=RHO)


def _nuts(**cfg):
    from gholax.sampler import NUTS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return NUTS({"sampler": {"NUTS": dict(cfg)}})


def _jlp():
    return jax.jit(_model().log_posterior_scaled_params)


def test_best_fit_position_is_unchanged():
    s = _nuts()
    x = s._best_fit_position(_jlp(), X0, rng_key=jax.random.key(0))
    assert np.allclose(np.asarray(x), BEST_FIT, rtol=1e-4, atol=1e-9)


def test_hessian_metrics_are_unchanged():
    """Diagonal and dense curvature at a fixed off-MAP point."""
    s, jlp = _nuts(), _jlp()
    diag = np.asarray(s._hessian_mass_matrix(jlp, X0))
    assert np.allclose(diag, HESSIAN_DIAG, rtol=1e-5)

    dense = np.asarray(s._hessian_mass_matrix_dense(jlp, X0))
    idx = np.arange(DIM)
    truth = RHO ** np.abs(idx[:, None] - idx[None, :])
    # the dense metric is the covariance, recovered exactly for a Gaussian
    assert np.allclose(dense, truth, atol=1e-5)
    assert np.allclose(np.diag(dense), 1.0, atol=1e-5)


def test_pathfinder_seeds_are_unchanged(tmp_path):
    """Same key, same paths, same cloud - and the ELBO metric is the target."""
    s = _nuts(pathfinder_init=True, pathfinder_n_paths=3,
              pathfinder_elbo_samples=10)
    pos = np.asarray(
        s._pathfinder_init(_jlp(), X0, 8, jax.random.key(1), str(tmp_path / "pf"))
    )
    assert pos.shape == (8, DIM) and np.isfinite(pos).all()
    idx = np.arange(DIM)
    truth = RHO ** np.abs(idx[:, None] - idx[None, :])
    assert np.allclose(np.asarray(s._pathfinder_imm), truth, atol=0.35)
    # a second identical call must reproduce the cloud exactly
    pos2 = np.asarray(
        s._pathfinder_init(_jlp(), X0, 8, jax.random.key(1), str(tmp_path / "pf2"))
    )
    assert np.array_equal(pos, pos2)


def test_minimize_and_sample_writes_the_same_map(tmp_path):
    """x_opt and value in .minimization_results.json, identical starts."""
    import json

    s = _nuts(minimize_and_sample=True, pathfinder_init=False)
    s._prior = None
    prefix = str(tmp_path / "min")
    starts = jnp.tile(X0, (4, 1))
    pos = s._minimize_and_sample(
        _model().log_posterior_scaled_params, starts, 4, prefix
    )
    res = json.load(open(prefix + ".minimization_results.json"))
    x_opt = np.asarray(res["x_opt"])
    assert x_opt.shape == (4, DIM)
    # identical starts: one minimization, tiled to every chain
    assert np.array_equal(x_opt, np.tile(x_opt[0], (4, 1)))
    assert np.allclose(x_opt[0], BEST_FIT, rtol=1e-4, atol=1e-9)
    assert np.allclose(np.asarray(res["value"]), 0.0, atol=1e-6)
    assert "x_opt_physical" not in res  # prior has no constrain
    assert np.allclose(np.asarray(pos), x_opt)


def _mh_model():
    """Correlated target with the prior methods MetropolisHastings needs."""
    idx = jnp.arange(DIM)
    covariance = RHO ** jnp.abs(idx[:, None] - idx[None, :])
    precision = jnp.linalg.inv(covariance)

    class MHPrior:
        params = [f"x{i}" for i in range(DIM)]

        def get_prior_sigmas(self):
            return np.ones(DIM)

        def get_reference_values(self):
            return np.zeros(DIM)

        def get_proposal_sigmas(self):
            return np.full(DIM, 0.5)

        def constrain(self, y):
            return y

        def initial_position(self, random_start=True, key=None, normalize=True):
            return {f"x{i}": jnp.float32(0.5) for i in range(DIM)}

    class MHModel:
        prior = MHPrior()

        def log_posterior_scaled_params(self, p):
            return -0.5 * p @ precision @ p

    return MHModel()


@pytest.mark.parametrize("init_covariance", ["hessian", "hessian_dense"])
def test_mh_hessian_proposal_covariance_is_unchanged(tmp_path, init_covariance):
    """The MH proposal covariance, pinned across the metric redirect.

    `hessian` used to run a bespoke per-chain jax.hessian block and now
    routes to the shared hessian_dense metric; both must recover the target
    covariance, and both must respect the prior-width cap.

    `update_covariance: false` is required for a stable anchor: with updates
    on, the file holds the chain-estimated covariance from the last
    convergence increment, which depends on the wall-clock rng seed.  The
    commit that redirects `hessian` onto the shared metric ladder is the one
    allowed to move these numbers.
    """
    from gholax.sampler import MetropolisHastings

    cfg = {"sampler": {"MetropolisHastings": {
        "n_steps": 10, "n_steps_checkpoint": 5, "n_steps_incr": 100,
        "target_r_minus_one": 1.0, "init_covariance": init_covariance,
        "update_covariance": False, "random_start": False,
    }}}
    prefix = str(tmp_path / f"mh_{init_covariance}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        MetropolisHastings(cfg).run(_mh_model(), prefix)
    cov = np.load(prefix + ".proposal_cov.npy")
    idx = np.arange(DIM)
    truth = RHO ** np.abs(idx[:, None] - idx[None, :])
    # curvature at the MAP of a Gaussian recovers the covariance exactly
    assert np.allclose(cov, truth, atol=1e-5)
    # the prior-width cap (unit-width prior, unit marginal variances here)
    assert np.all(np.sqrt(np.diag(cov)) <= 1.0 + 1e-5)


@pytest.mark.parametrize("resample", [False, True])
def test_pathfinder_resample_flag_is_honored(tmp_path, resample, capsys):
    """Pins the resolved default disagreement: config wins, not base.py."""
    s = _nuts(pathfinder_init=True, pathfinder_resample=resample,
              pathfinder_n_paths=3, pathfinder_elbo_samples=10)
    assert s.pathfinder_resample is resample
    s._pathfinder_init(_jlp(), X0, 8, jax.random.key(1),
                       str(tmp_path / f"pf{resample}"))
    assert ("importance-weight ESS" in capsys.readouterr().out) is resample
