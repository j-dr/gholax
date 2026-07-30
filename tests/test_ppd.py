"""Tests for the posterior predictive distribution machinery in gholax.util.ppd.

The statistical core takes plain arrays, so most of these run without emulators
or data files.  The end-to-end tests drive posterior_predictive_test() with
lightweight stand-ins for Model / Likelihood / DataVector that expose only the
attributes the PPD code actually touches.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from gholax.likelihood.gaussian_likelihood import GaussianLikelihood
from gholax.util.ppd import (
    DISJOINT_CONDITIONAL,
    DISJOINT_INDEPENDENT,
    GOODNESS_OF_FIT,
    NESTED_CONDITIONAL,
    collect_elements,
    conditional_moments,
    element_keys,
    posterior_predictive_test,
    ppd_pvalue,
    resolve_mode,
    sample_prior,
)


def _random_covariance(rng, n, scale=1.0):
    """A well-conditioned positive-definite matrix."""
    a = rng.standard_normal((n, n))

    return scale * (a @ a.T / n + np.eye(n))


# ----------------------------------------------------------------------
# Gaussian conditional
# ----------------------------------------------------------------------
def test_conditional_moments_matches_explicit_formula():
    rng = np.random.default_rng(0)
    n1, n2, n_samples = 5, 3, 7

    joint = _random_covariance(rng, n1 + n2)
    C11, C12, C22 = joint[:n1, :n1], joint[:n1, n1:], joint[n1:, n1:]
    mu1 = rng.standard_normal((n_samples, n1))
    mu2 = rng.standard_normal((n_samples, n2))
    d1 = rng.standard_normal(n1)

    mu_cond, cov_cond = conditional_moments(mu1, mu2, d1, C11, C12, C22)

    C21 = C12.T
    expected_cov = C22 - C21 @ np.linalg.inv(C11) @ C12
    assert np.allclose(cov_cond, expected_cov)
    assert np.allclose(cov_cond, cov_cond.T)

    for i in range(n_samples):
        expected = mu2[i] + C21 @ np.linalg.inv(C11) @ (d1 - mu1[i])
        assert np.allclose(mu_cond[i], expected)


def test_conditional_moments_reproduce_joint_samples():
    """Residuals of a joint Gaussian about the conditional mean have cov_cond."""
    rng = np.random.default_rng(1)
    n1, n2, n_draws = 4, 3, 200_000

    joint = _random_covariance(rng, n1 + n2)
    C11, C12, C22 = joint[:n1, :n1], joint[:n1, n1:], joint[n1:, n1:]

    draws = rng.multivariate_normal(np.zeros(n1 + n2), joint, size=n_draws)
    d1, d2 = draws[:, :n1], draws[:, n1:]

    # conditional_moments batches over the leading axis of mu1/mu2; feed it a
    # fixed zero mean and let d1 vary by treating each draw as its own "sample".
    mu_cond, cov_cond = conditional_moments(
        np.zeros((1, n1)), np.zeros((1, n2)), np.zeros(n1), C11, C12, C22
    )
    shift = np.linalg.solve(C11, C12)
    residual = d2 - d1 @ shift

    assert np.allclose(np.cov(residual, rowvar=False), cov_cond, atol=0.02)
    assert np.allclose(residual.mean(axis=0), 0.0, atol=0.02)
    assert mu_cond.shape == (1, n2)


def test_conditioning_shrinks_predictive_covariance():
    rng = np.random.default_rng(2)
    n1, n2 = 6, 4

    joint = _random_covariance(rng, n1 + n2)
    C11, C12, C22 = joint[:n1, :n1], joint[:n1, n1:], joint[n1:, n1:]

    _, cov_cond = conditional_moments(
        np.zeros((1, n1)), np.zeros((1, n2)), np.zeros(n1), C11, C12, C22
    )

    # Conditioning on correlated data can only remove variance.
    assert np.linalg.det(cov_cond) < np.linalg.det(C22)
    assert np.all(np.diag(cov_cond) <= np.diag(C22) + 1e-12)
    assert np.all(np.linalg.eigvalsh(cov_cond) > 0)


# ----------------------------------------------------------------------
# The p-value itself
# ----------------------------------------------------------------------
def _held_out_pvalue(rng, n_pred=8, n_samples=400, posterior_scatter=0.0):
    """One trial of predicting held-out data from a well-constrained posterior."""
    cov = _random_covariance(rng, n_pred)
    chol = np.linalg.cholesky(cov)

    mu_true = np.zeros(n_pred)
    d_obs = mu_true + chol @ rng.standard_normal(n_pred)
    mu = mu_true[None, :] + posterior_scatter * (
        rng.standard_normal((n_samples, n_pred)) @ chol.T
    )

    return ppd_pvalue(d_obs, mu, cov, rng)[0]


def test_ppd_pvalue_uniform_for_held_out_prediction():
    """With a tightly constrained posterior the held-out p-value is uniform.

    This is the strongest correctness check on the statistic: the replica and
    the real data are then exchangeable draws from the same predictive
    distribution, so P(T_rep > T_obs) must be uniform on [0, 1].
    """
    rng = np.random.default_rng(3)
    pvals = np.array([_held_out_pvalue(rng) for _ in range(400)])

    assert stats.kstest(pvals, "uniform").pvalue > 0.01
    assert abs(pvals.mean() - 0.5) < 0.05
    assert abs(pvals.std() - 1 / np.sqrt(12)) < 0.03


def test_ppd_pvalue_stays_uniform_with_modest_posterior_scatter():
    rng = np.random.default_rng(4)
    pvals = np.array(
        [_held_out_pvalue(rng, posterior_scatter=0.05) for _ in range(400)]
    )

    assert stats.kstest(pvals, "uniform").pvalue > 0.01


def test_ppd_pvalue_goodness_of_fit_is_underdispersed():
    """Fitting the data first makes the raw p-value conservative.

    The parameters were tuned to d_obs, so T(d_obs, Theta) is systematically
    pulled toward T(d_rep, Theta) and the p-value clusters around 0.5 rather
    than spreading uniformly. This is exactly the bias the paper's calibrated
    p-value exists to remove, and it is why we report the raw value as
    conservative rather than exact.
    """
    rng = np.random.default_rng(5)
    n_data, n_params, n_samples = 20, 8, 400

    design = rng.standard_normal((n_data, n_params))
    cov = np.eye(n_data)
    fisher_inv = np.linalg.inv(design.T @ design)
    post_chol = np.linalg.cholesky(fisher_inv)

    pvals = []
    for _ in range(400):
        d_obs = rng.standard_normal(n_data)
        theta_hat = fisher_inv @ design.T @ d_obs
        theta = theta_hat[None, :] + rng.standard_normal((n_samples, n_params)) @ post_chol.T
        pvals.append(ppd_pvalue(d_obs, theta @ design.T, cov, rng)[0])
    pvals = np.array(pvals)

    assert abs(pvals.mean() - 0.5) < 0.05
    assert pvals.std() < 0.8 / np.sqrt(12)
    assert stats.kstest(pvals, "uniform").pvalue < 0.01


def test_ppd_pvalue_rejects_shifted_data():
    rng = np.random.default_rng(6)
    n_pred, n_samples = 10, 500

    cov = _random_covariance(rng, n_pred)
    mu = np.zeros((n_samples, n_pred))
    sigma = np.sqrt(np.diag(cov))

    p_good, _, _, _ = ppd_pvalue(np.zeros(n_pred), mu, cov, rng)
    p_bad, _, _, _ = ppd_pvalue(10.0 * sigma, mu, cov, rng)

    assert p_good > 0.2
    assert p_bad == 0.0


def test_ppd_replica_chi2_follows_chi2_distribution():
    rng = np.random.default_rng(7)
    n_pred, n_samples = 12, 20_000

    cov = _random_covariance(rng, n_pred)
    mu = np.zeros((n_samples, n_pred))
    _, chi2_rep, _, d_rep = ppd_pvalue(np.zeros(n_pred), mu, cov, rng)

    assert abs(chi2_rep.mean() - n_pred) < 0.2
    assert abs(chi2_rep.var() - 2 * n_pred) < 2.0
    # replicas are actually drawn from N(mu_cond, cov_cond)
    assert np.allclose(np.cov(d_rep, rowvar=False), cov, atol=0.1)


def test_ppd_pvalue_raises_on_singular_covariance():
    rng = np.random.default_rng(8)
    cov = np.ones((4, 4))  # rank 1

    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        ppd_pvalue(np.zeros(4), np.zeros((3, 4)), cov, rng)


# ----------------------------------------------------------------------
# Mode resolution
# ----------------------------------------------------------------------
def test_resolve_mode_identical_is_goodness_of_fit():
    keys = [("l", "c_dd", 0, 0, 10.0), ("l", "c_dd", 0, 0, 20.0)]

    assert resolve_mode(keys, list(keys), False) == GOODNESS_OF_FIT
    assert resolve_mode(keys, list(keys), True) == GOODNESS_OF_FIT


def test_resolve_mode_disjoint_depends_on_cross_covariance():
    a = [("l", "c_dd", 0, 0, 10.0)]
    b = [("m", "p_gg_ell", 0, 0, 0.1)]

    assert resolve_mode(a, b, False) == DISJOINT_INDEPENDENT
    assert resolve_mode(a, b, True) == DISJOINT_CONDITIONAL


def test_resolve_mode_superset_is_nested():
    a = [("l", "c_dd", 0, 0, 10.0)]
    b = a + [("l", "c_dd", 0, 0, 20.0)]

    assert resolve_mode(a, b, False) == NESTED_CONDITIONAL


def test_resolve_mode_rejects_subset_prediction():
    a = [("l", "c_dd", 0, 0, 10.0), ("l", "c_dd", 0, 0, 20.0)]

    with pytest.raises(ValueError, match="strict subset"):
        resolve_mode(a, a[:1], False)


def test_resolve_mode_rejects_partial_overlap():
    a = [("l", "c_dd", 0, 0, 10.0), ("l", "c_dd", 0, 0, 20.0)]
    b = [("l", "c_dd", 0, 0, 20.0), ("l", "c_dd", 0, 0, 30.0)]

    with pytest.raises(ValueError, match="partially overlap"):
        resolve_mode(a, b, False)


# ----------------------------------------------------------------------
# Stand-ins for Model / Likelihood / DataVector
# ----------------------------------------------------------------------
_SPECTRA_DTYPE = np.dtype([
    ("spectrum_type", "S10"),
    ("zbin0", int),
    ("zbin1", int),
    ("separation", float),
    ("value", float),
])


class _StubDataVector:
    """Minimum surface of DataVector that the PPD code reads."""

    def __init__(self, spectrum_type, separations, values, cov, scale_mask=None):
        n = len(separations)
        spectra = np.zeros(n, dtype=_SPECTRA_DTYPE)
        spectra["spectrum_type"] = spectrum_type
        spectra["separation"] = separations
        spectra["value"] = values

        self.spectra = spectra
        self.n_dv = n
        self.measured_spectra = jnp.array(values)
        self.scale_mask = jnp.arange(n) if scale_mask is None else jnp.array(scale_mask)

        structured = np.zeros((n, n), dtype=[("value", float)])
        structured["value"] = cov
        self.cov = structured

    def _covariance_match_fields(self):
        return [
            ("spectrum_type1", "spectrum_type"),
            ("zbin10", "zbin0"),
            ("zbin11", "zbin1"),
            ("separation1", "separation"),
        ]


class _StubLikelihood:
    """A likelihood whose model is linear in the sampled parameters."""

    def __init__(self, dv, param_names, design):
        self.observed_data_vector = dv
        self.sampled_params = {p: {} for p in param_names}
        self.Nlin = 0
        self._param_names = list(param_names)
        self._design = jnp.array(design)

    def predict_model(self, params, params_am=None, **kwargs):
        theta = jnp.stack([params[p] for p in self._param_names])

        return self._design @ theta


class _StubModel:
    def __init__(self, likelihoods, param_names, prior=None):
        self.likelihoods = likelihoods
        self.param_names = list(param_names)
        self.prior = prior


def _make_stub(rng, separations, param_names, scale_mask=None, seed_design=None,
               values=None, cov=None):
    n = len(separations)
    design = seed_design if seed_design is not None else rng.standard_normal(
        (n if scale_mask is None else len(scale_mask), len(param_names))
    )
    cov = _random_covariance(rng, n, scale=0.01) if cov is None else cov
    values = rng.standard_normal(n) if values is None else values

    dv = _StubDataVector("c_dd", separations, values, cov, scale_mask=scale_mask)

    return _StubModel({"L": _StubLikelihood(dv, param_names, design)}, param_names)


# ----------------------------------------------------------------------
# Element bookkeeping against the stand-ins
# ----------------------------------------------------------------------
def test_element_keys_cover_masked_elements_only():
    rng = np.random.default_rng(10)
    seps = np.array([10.0, 20.0, 30.0, 40.0])
    dv = _StubDataVector("c_dd", seps, rng.standard_normal(4),
                         _random_covariance(rng, 4), scale_mask=[1, 2])

    keys = element_keys(dv, "L")

    assert keys == [("L", "c_dd", 0, 0, 20.0), ("L", "c_dd", 0, 0, 30.0)]


def test_collect_elements_orders_and_blocks():
    rng = np.random.default_rng(11)
    seps = np.array([10.0, 20.0, 30.0])
    values = rng.standard_normal(3)
    cov = _random_covariance(rng, 3)
    model = _make_stub(rng, seps, ["a", "b"], values=values, cov=cov)

    elements = collect_elements(model)

    assert len(elements.keys) == 3
    assert np.allclose(elements.d_obs, values)
    assert np.allclose(elements.cov, cov)
    assert elements.slices["L"] == slice(0, 3)


def test_collect_elements_requires_a_covariance():
    rng = np.random.default_rng(12)
    model = _make_stub(rng, np.array([10.0, 20.0]), ["a"])
    model.likelihoods["L"].observed_data_vector.cov = None

    with pytest.raises(ValueError, match="no covariance loaded"):
        collect_elements(model)


# ----------------------------------------------------------------------
# Prior sampling
# ----------------------------------------------------------------------
def _stub_prior(prior_info, joint=None, linear=None, reference=None):
    return SimpleNamespace(
        prior_info=prior_info,
        joint_prior_groups=joint or {},
        linear_constraint_groups=linear or {},
        get_reference_point=lambda: reference or {},
    )


def test_sample_prior_recovers_prior_moments():
    rng = np.random.default_rng(13)
    prior = _stub_prior({
        "u": {"dist": "uniform", "min": -2.0, "max": 4.0},
        "g": {"dist": "norm", "loc": 1.5, "scale": 0.5},
    })

    draws = sample_prior(prior, ["u", "g"], 200_000, rng)

    assert abs(draws["u"].mean() - 1.0) < 0.02
    assert abs(draws["u"].std() - 6.0 / np.sqrt(12)) < 0.02
    assert draws["u"].min() >= -2.0 and draws["u"].max() <= 4.0
    assert abs(draws["g"].mean() - 1.5) < 0.01
    assert abs(draws["g"].std() - 0.5) < 0.01


def test_sample_prior_pins_jointly_constrained_params():
    rng = np.random.default_rng(14)
    prior = _stub_prior(
        {"a": {"dist": "norm", "loc": 0.0, "scale": 1.0}},
        joint={"grp": {"params": ["a"]}},
        reference={"a": 0.75},
    )

    with pytest.warns(UserWarning, match="joint or linear-constraint prior"):
        draws = sample_prior(prior, ["a"], 10, rng)

    assert np.all(draws["a"] == 0.75)


def test_sample_prior_rejects_unknown_prior_form():
    rng = np.random.default_rng(15)
    prior = _stub_prior({"a": {"dist": "cauchy"}})

    with pytest.raises(NotImplementedError, match="cauchy"):
        sample_prior(prior, ["a"], 5, rng)


# ----------------------------------------------------------------------
# End-to-end through posterior_predictive_test
# ----------------------------------------------------------------------
def test_end_to_end_goodness_of_fit_matches_the_data():
    """A data vector generated from the model at the sampled params passes."""
    rng = np.random.default_rng(20)
    n, n_params, n_samples = 6, 2, 500

    seps = np.arange(1.0, n + 1.0)
    design = rng.standard_normal((n, n_params))
    cov = _random_covariance(rng, n, scale=0.01)
    theta_true = rng.standard_normal(n_params)
    values = design @ theta_true + np.linalg.cholesky(cov) @ rng.standard_normal(n)

    model = _make_stub(rng, seps, ["a", "b"], seed_design=design, values=values,
                       cov=cov)
    samples = theta_true[None, :] + 0.01 * rng.standard_normal((n_samples, n_params))

    result = posterior_predictive_test((model, samples, ["a", "b"]), thin=1)

    assert result.mode == GOODNESS_OF_FIT
    assert result.n_pred == n
    assert result.n_samples == n_samples
    assert np.allclose(result.cov_cond, cov)
    assert 0.02 < result.p_value < 0.98


def test_end_to_end_goodness_of_fit_flags_a_bad_data_vector():
    rng = np.random.default_rng(21)
    n, n_params, n_samples = 6, 2, 300

    seps = np.arange(1.0, n + 1.0)
    design = rng.standard_normal((n, n_params))
    cov = _random_covariance(rng, n, scale=0.01)
    theta_true = rng.standard_normal(n_params)
    values = design @ theta_true + 25.0 * np.sqrt(np.diag(cov))

    model = _make_stub(rng, seps, ["a", "b"], seed_design=design, values=values,
                       cov=cov)
    samples = theta_true[None, :] + 0.01 * rng.standard_normal((n_samples, n_params))

    result = posterior_predictive_test((model, samples, ["a", "b"]))

    assert result.p_value == 0.0
    assert result.chi2_obs.mean() > result.chi2_rep.mean()


def test_end_to_end_nested_conditional_uses_the_cross_covariance():
    """Predicting held-out scales conditions on the fitted ones."""
    rng = np.random.default_rng(22)
    n_fit, n_extra, n_params, n_samples = 5, 4, 2, 400
    n_all = n_fit + n_extra

    seps_all = np.arange(1.0, n_all + 1.0)
    design_all = rng.standard_normal((n_all, n_params))
    cov_all = _random_covariance(rng, n_all, scale=0.01)
    theta_true = rng.standard_normal(n_params)
    values = design_all @ theta_true + np.linalg.cholesky(cov_all) @ rng.standard_normal(n_all)

    chain_model = _make_stub(
        rng, seps_all[:n_fit], ["a", "b"], seed_design=design_all[:n_fit],
        values=values[:n_fit], cov=cov_all[:n_fit, :n_fit],
    )
    pred_model = _make_stub(
        rng, seps_all, ["a", "b"], seed_design=design_all, values=values, cov=cov_all,
    )
    samples = theta_true[None, :] + 0.01 * rng.standard_normal((n_samples, n_params))

    result = posterior_predictive_test(
        (chain_model, samples, ["a", "b"]), prediction_config=pred_model
    )

    assert result.mode == NESTED_CONDITIONAL
    assert result.n_pred == n_extra
    assert np.allclose(result.d_obs, values[n_fit:])

    # cov_cond must be the Schur complement, strictly tighter than C22 alone.
    c11 = cov_all[:n_fit, :n_fit]
    c12 = cov_all[:n_fit, n_fit:]
    expected = cov_all[n_fit:, n_fit:] - c12.T @ np.linalg.solve(c11, c12)
    assert np.allclose(result.cov_cond, expected, atol=1e-10)
    assert np.linalg.det(result.cov_cond) < np.linalg.det(cov_all[n_fit:, n_fit:])


def test_end_to_end_disjoint_with_and_without_cross_covariance():
    rng = np.random.default_rng(23)
    n1, n2, n_params, n_samples = 5, 4, 2, 300

    joint = _random_covariance(rng, n1 + n2, scale=0.01)
    c11, c12, c22 = joint[:n1, :n1], joint[:n1, n1:], joint[n1:, n1:]
    theta_true = rng.standard_normal(n_params)

    design1 = rng.standard_normal((n1, n_params))
    design2 = rng.standard_normal((n2, n_params))
    draw = np.linalg.cholesky(joint) @ rng.standard_normal(n1 + n2)
    values1 = design1 @ theta_true + draw[:n1]
    values2 = design2 @ theta_true + draw[n1:]

    chain_model = _make_stub(rng, np.arange(1.0, n1 + 1.0), ["a", "b"],
                             seed_design=design1, values=values1, cov=c11)
    # Distinct separations so the two data vectors do not share any element.
    pred_model = _make_stub(rng, np.arange(100.0, 100.0 + n2), ["a", "b"],
                            seed_design=design2, values=values2, cov=c22)
    samples = theta_true[None, :] + 0.01 * rng.standard_normal((n_samples, n_params))

    independent = posterior_predictive_test(
        (chain_model, samples, ["a", "b"]), prediction_config=pred_model
    )
    conditional = posterior_predictive_test(
        (chain_model, samples, ["a", "b"]), prediction_config=pred_model,
        cross_covariance=c12,
    )

    assert independent.mode == DISJOINT_INDEPENDENT
    assert np.allclose(independent.cov_cond, c22)

    assert conditional.mode == DISJOINT_CONDITIONAL
    expected = c22 - c12.T @ np.linalg.solve(c11, c12)
    assert np.allclose(conditional.cov_cond, expected, atol=1e-10)
    assert np.linalg.det(conditional.cov_cond) < np.linalg.det(c22)
    # The conditional shift moves the predictive mean off the raw prediction.
    assert not np.allclose(conditional.mu_cond, independent.mu_cond)


def test_cross_covariance_shape_is_checked():
    rng = np.random.default_rng(24)
    chain_model = _make_stub(rng, np.arange(1.0, 4.0), ["a"])
    pred_model = _make_stub(rng, np.arange(100.0, 104.0), ["a"])
    samples = rng.standard_normal((10, 1))

    with pytest.raises(ValueError, match="expected the masked"):
        posterior_predictive_test(
            (chain_model, samples, ["a"]), prediction_config=pred_model,
            cross_covariance=np.zeros((7, 7)),
        )


def test_result_is_reproducible_and_seed_sensitive():
    rng = np.random.default_rng(25)
    model = _make_stub(rng, np.arange(1.0, 7.0), ["a", "b"])
    samples = rng.standard_normal((200, 2))

    a = posterior_predictive_test((model, samples, ["a", "b"]), seed=1)
    b = posterior_predictive_test((model, samples, ["a", "b"]), seed=1)
    c = posterior_predictive_test((model, samples, ["a", "b"]), seed=2)

    assert a.p_value == b.p_value
    assert np.allclose(a.d_rep, b.d_rep)
    assert not np.allclose(a.d_rep, c.d_rep)


def test_missing_parameters_are_drawn_from_the_prior():
    """A prediction model param absent from the chain is filled from its prior."""
    rng = np.random.default_rng(26)
    n, n_samples = 5, 200

    model = _make_stub(rng, np.arange(1.0, n + 1.0), ["a", "b"])
    model.prior = _stub_prior({"b": {"dist": "norm", "loc": 0.0, "scale": 1.0}})
    samples = rng.standard_normal((n_samples, 1))

    with pytest.warns(UserWarning, match="drawn from their prior"):
        result = posterior_predictive_test((model, samples, ["a"]))

    assert result.n_samples == n_samples
    # 'b' varying over its prior widens the spread of predictions.
    assert result.mu_cond.std(axis=0).min() > 0.0


# ----------------------------------------------------------------------
# Analytic marginalization moments
# ----------------------------------------------------------------------
class _LinearAMLikelihood(GaussianLikelihood):
    """A GaussianLikelihood whose model is exactly linear in the AM params.

    Bypasses the config-driven __init__ and supplies only what _am_solve reads,
    so the real Va/Lab code path is exercised without any data files.
    """

    def __init__(self, d_obs, cinv, m0, templates, means, stds,
                 include_am_priors=True, include_am_determinant=True):
        self.linear_params_means = dict(means)
        self.linear_params_names = np.array(list(means.keys()))
        self.linear_params_stds = jnp.array(stds)
        self.Nlin = len(means)
        self.include_am_priors = include_am_priors
        self.include_am_determinant = include_am_determinant
        self.observed_data_vector = SimpleNamespace(
            cinv=jnp.array(cinv),
            measured_spectra=jnp.array(d_obs),
            scale_mask=jnp.arange(len(d_obs)),
        )
        self._m0 = jnp.array(m0)
        self._templates = jnp.array(templates)

    def predict_model(self, params, params_am=None, **kwargs):
        delta = jnp.stack([
            params_am[k] - self.linear_params_means[k]
            for k in self.linear_params_names
        ])

        return self._m0 + delta @ self._templates


def _make_am_likelihood(rng, n_data=8, n_lin=3, **kwargs):
    cov = _random_covariance(rng, n_data)
    cinv = np.linalg.inv(cov)
    m0 = rng.standard_normal(n_data)
    templates = rng.standard_normal((n_lin, n_data))
    means = {f"a{i}": float(v) for i, v in enumerate(rng.standard_normal(n_lin))}
    stds = np.abs(rng.standard_normal(n_lin)) + 0.5
    d_obs = m0 + rng.standard_normal(n_data)

    like = _LinearAMLikelihood(d_obs, cinv, m0, templates, means, stds, **kwargs)

    return like, dict(d_obs=d_obs, cinv=cinv, m0=m0, templates=templates,
                      means=means, stds=stds)


def test_am_conditional_moments_match_augmented_least_squares():
    """p(a | Theta, d) recovered independently as a regularized least squares fit."""
    rng = np.random.default_rng(30)
    like, ref = _make_am_likelihood(rng)

    mean, cov = like.am_conditional_moments({})
    mean, cov = np.asarray(mean, dtype=float), np.asarray(cov, dtype=float)

    # Independent derivation: stack the whitened data residual on top of the
    # whitened prior and solve the resulting linear least squares problem.
    whiten = np.linalg.cholesky(ref["cinv"]).T
    design = np.vstack([whiten @ ref["templates"].T, np.diag(1.0 / ref["stds"])])
    target = np.concatenate([whiten @ (ref["d_obs"] - ref["m0"]),
                             np.zeros(len(ref["stds"]))])

    delta = np.linalg.lstsq(design, target, rcond=None)[0]
    expected_cov = np.linalg.inv(design.T @ design)
    expected_mean = np.array(list(ref["means"].values())) + delta

    assert np.allclose(mean, expected_mean, rtol=1e-4, atol=1e-5)
    assert np.allclose(cov, expected_cov, rtol=1e-4, atol=1e-6)


def test_am_conditional_moments_without_priors():
    rng = np.random.default_rng(31)
    like, ref = _make_am_likelihood(rng, include_am_priors=False)

    mean, cov = like.am_conditional_moments({})

    whiten = np.linalg.cholesky(ref["cinv"]).T
    design = whiten @ ref["templates"].T
    delta = np.linalg.lstsq(design, whiten @ (ref["d_obs"] - ref["m0"]), rcond=None)[0]

    assert np.allclose(np.asarray(mean, dtype=float),
                       np.array(list(ref["means"].values())) + delta,
                       rtol=1e-4, atol=1e-5)
    assert np.allclose(np.asarray(cov, dtype=float),
                       np.linalg.inv(design.T @ design), rtol=1e-4, atol=1e-6)


def test_am_conditional_moments_empty_when_no_linear_params():
    rng = np.random.default_rng(32)
    like, _ = _make_am_likelihood(rng, n_lin=1)
    like.Nlin = 0

    mean, cov = like.am_conditional_moments({})

    assert mean.shape == (0,)
    assert cov.shape == (0, 0)


def test_compute_am_equals_numerical_marginalization():
    """The refactored compute_am still integrates the linear param out correctly."""
    rng = np.random.default_rng(33)
    like, ref = _make_am_likelihood(rng, n_data=5, n_lin=1)

    lnL = float(like.compute_am({})[0])

    a0 = np.array(list(ref["means"].values()))[0]
    std = ref["stds"][0]
    grid = np.linspace(a0 - 40 * std, a0 + 40 * std, 400_001)
    resid = (ref["d_obs"] - ref["m0"])[None, :] - (grid - a0)[:, None] * ref["templates"][0][None, :]
    chi2 = np.einsum("ij,jk,ik->i", resid, ref["cinv"], resid)
    integrand = np.exp(-0.5 * chi2 - 0.5 * ((grid - a0) / std) ** 2)

    assert np.isclose(lnL, np.log(np.trapezoid(integrand, grid)), rtol=1e-4, atol=1e-4)


def test_compute_am_returns_templates_of_the_right_shape():
    rng = np.random.default_rng(34)
    like, ref = _make_am_likelihood(rng)

    _, _, templates = like.compute_am({})

    assert np.allclose(np.asarray(templates, dtype=float), ref["templates"],
                       rtol=1e-4, atol=1e-5)
