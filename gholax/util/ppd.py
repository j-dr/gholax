"""Posterior predictive distribution (PPD) internal-consistency tests.

Implements the PPD test of Doux et al. 2021 (arXiv:2011.03410), which asks how
often a replica data vector drawn from the fitted model fits worse than the data
actually observed:

    P(d_rep | d_obs) = INT dTheta  P(d_rep | d_obs, Theta) P(Theta | d_obs)
    T(d, Theta)      = (d - mu(Theta))^T C^-1 (d - mu(Theta))
    p                = P( T(d_rep, Theta) > T(d_obs, Theta) | d_obs )

When the data splits into a block d1 that the chain conditioned on and a block
d2 being predicted, and the two are correlated, the replica is drawn from the
Gaussian conditional

    P(d2 | d1, Theta) = N( mu2 + C21 C11^-1 (d1 - mu1),  C22 - C21 C11^-1 C12 ).

The entry point is :func:`posterior_predictive_test`, which takes a chain (as a
config file for a run that has already completed) and optionally a second config
describing the data to predict, and resolves which of the above cases applies by
comparing the two data vectors element by element.  See :func:`resolve_mode`.

Only the raw PPD p-value is computed.  It is conservative -- the parameters were
tuned to d_obs -- but directly interpretable.  The calibrated p-value of the
paper (simulated data vectors reweighted onto a fiducial chain) would slot in
around :func:`ppd_pvalue`; see the note there.
"""

import argparse
import os
import warnings
from collections import namedtuple
from dataclasses import dataclass, field

import h5py as h5
import jax
import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from gholax.util.model import Model
from gholax.util.postprocess_chain import load_model_samples

# Test modes, resolved from how the predicted elements relate to the
# conditioned-on elements.  See resolve_mode().
GOODNESS_OF_FIT = "goodness_of_fit"
DISJOINT_INDEPENDENT = "disjoint_independent"
DISJOINT_CONDITIONAL = "disjoint_conditional"
NESTED_CONDITIONAL = "nested_conditional"


@dataclass
class PPDResult:
    """Outcome of a posterior predictive test.

    Attributes:
        p_value: Fraction of posterior samples whose replica fits worse than the
            data, P(T(d_rep, Theta) > T(d_obs, Theta) | d_obs).
        p_value_error: Binomial Monte-Carlo error on p_value.
        mode: Which of the four test modes was resolved.
        chi2_obs: (n_samples,) T(d_obs, Theta_i).
        chi2_rep: (n_samples,) T(d_rep_i, Theta_i).
        d_rep: (n_samples, n_pred) replica data vectors.
        d_obs: (n_pred,) observed values of the predicted elements.
        mu_cond: (n_samples, n_pred) conditional predictive means.
        cov_cond: (n_pred, n_pred) conditional predictive covariance.
        element_keys: Identifying tuples for the predicted elements.
        key_fields: Dict mapping likelihood name to the field names of that
            likelihood's element keys, so the plotting routines can pull out the
            spectrum type, tomographic bins and separation of each element
            without assuming a fixed key layout.
        n_samples: Number of posterior samples used.
        n_pred: Number of predicted data vector elements.
        n_nonfinite: Number of posterior samples dropped because the model
            prediction was non-finite.
        n_outlier: Number of posterior samples dropped because the prediction
            was finite but implausibly far from the data; see
            `max_residual_sigma`.
        params: Dict mapping every parameter the prediction used to an
            (n_samples,) array, aligned with chi2_obs. Includes the parameters
            drawn from their prior and any redraws that replaced a non-finite
            prediction, so this is what was actually fed to the model rather
            than what the chain stored.
        drawn_params: Names within `params` that came from the prior because
            the chain did not sample them.
        sample_index: (n_samples,) indices back into the post-burn-in, post-thin
            chain samples. Not simply arange(n_samples) when non-finite
            predictions were dropped.
    """

    p_value: float
    p_value_error: float
    mode: str
    chi2_obs: np.ndarray
    chi2_rep: np.ndarray
    d_rep: np.ndarray
    d_obs: np.ndarray
    mu_cond: np.ndarray
    cov_cond: np.ndarray
    element_keys: list = field(repr=False)
    key_fields: dict = field(default_factory=dict, repr=False)
    n_samples: int = 0
    n_pred: int = 0
    n_nonfinite: int = 0
    n_outlier: int = 0
    params: dict = field(default_factory=dict, repr=False)
    drawn_params: list = field(default_factory=list, repr=False)
    sample_index: np.ndarray = field(default=None, repr=False)

    def worst_samples(self, k=10):
        """The k samples with the largest chi2_obs, with the parameters behind them.

        A quick way to find out which parameters drive an implausibly large
        discrepancy -- typically prior-drawn parameters wandering outside the
        region where the theory pipeline is well behaved.

        Args:
            k: Number of samples to report.

        Returns:
            Dict with 'index' (into the chain), 'chi2_obs', and one entry per
            parameter, each ordered by decreasing chi2_obs.
        """
        order = np.argsort(self.chi2_obs)[::-1][:k]

        out = {"chi2_obs": self.chi2_obs[order]}
        if self.sample_index is not None:
            out["index"] = np.asarray(self.sample_index)[order]
        out.update({p: np.asarray(v)[order] for p, v in self.params.items()})

        return out

    def __str__(self):
        dropped = ""
        if self.n_nonfinite or self.n_outlier:
            dropped = (
                f"  dropped    = {self.n_nonfinite + self.n_outlier} "
                f"({self.n_nonfinite} non-finite, {self.n_outlier} outlying)\n"
            )

        return (
            f"PPD test [{self.mode}]\n"
            f"  n_pred     = {self.n_pred}\n"
            f"  n_samples  = {self.n_samples}\n"
            f"{dropped}"
            f"  <chi2_obs> = {np.mean(self.chi2_obs):.2f}\n"
            f"  <chi2_rep> = {np.mean(self.chi2_rep):.2f}\n"
            f"  p-value    = {self.p_value:.4f} +/- {self.p_value_error:.4f}"
        )


# ----------------------------------------------------------------------
# Data vector element bookkeeping
# ----------------------------------------------------------------------
def _round_sig(x, sig=12):
    """Round to `sig` significant digits so floats hash reliably across files."""
    x = float(x)
    if x == 0.0 or not np.isfinite(x):
        return x

    return float(round(x, sig - int(np.floor(np.log10(abs(x)))) - 1))


def element_key_fields(dv):
    """Names of the fields making up each element key of a data vector.

    The leading ``'likelihood'`` matches the likelihood name that
    :func:`element_keys` puts first; the rest are the spectra fields
    ``dv._covariance_match_fields()`` matches on.

    Args:
        dv: A DataVector instance.

    Returns:
        List of field names, aligned with the entries of an element key.
    """
    return ["likelihood"] + [
        spectra_field for _, spectra_field in dv._covariance_match_fields()
    ]


def element_keys(dv, likelihood_name):
    """Tuple keys identifying each *masked* element of a data vector.

    The key fields are taken from ``dv._covariance_match_fields()`` -- the same
    fields ``DataVector.load_covariance_matrix`` uses to match data vector
    elements onto covariance rows -- so this automatically picks up the
    multipole order for redshift-space multipoles.

    Args:
        dv: A DataVector instance with load_data() already called.
        likelihood_name: Name of the owning likelihood, used to keep elements of
            different likelihoods distinct.

    Returns:
        List of hashable tuples, one per element surviving the scale cuts, in
        data vector order.
    """
    fields = element_key_fields(dv)[1:]
    rows = np.asarray(dv.spectra)[np.asarray(dv.scale_mask)]

    keys = []
    for row in rows:
        key = [likelihood_name]
        for f in fields:
            v = row[f]
            if isinstance(v, (bytes, np.bytes_)):
                key.append(v.decode())
            elif isinstance(v, (float, np.floating)):
                key.append(_round_sig(v))
            else:
                key.append(int(v))
        keys.append(tuple(key))

    return keys


@dataclass
class ModelElements:
    """Flattened, scale-cut masked view of every data vector in a Model.

    The element ordering matches the concatenation of each likelihood's
    ``predict_model(apply_scale_mask=True)`` output, in ``model.likelihoods``
    order, so predictions and observations line up index for index.
    """

    model: object
    keys: list
    d_obs: np.ndarray
    cov: np.ndarray
    slices: dict
    key_fields: dict = field(default_factory=dict)


def collect_elements(model):
    """Build the flattened masked element view of every likelihood in a Model.

    Args:
        model: A Model instance.

    Returns:
        ModelElements with keys, observed values, block-diagonal covariance and
        the per-likelihood slices into the concatenated vector.

    Raises:
        ValueError: If a likelihood has no covariance loaded (``dummy_cov``), or
            if two elements collide on the same key.
    """
    keys, d_obs, blocks, slices, key_fields = [], [], [], {}, {}
    start = 0

    for lname, like in model.likelihoods.items():
        dv = like.observed_data_vector
        mask = np.asarray(dv.scale_mask)

        cov = getattr(dv, "cov", None)
        if cov is None:
            raise ValueError(
                f"Likelihood {lname!r} has no covariance loaded (dummy_cov=True?); "
                "the PPD test needs one to draw replica data vectors."
            )

        k = element_keys(dv, lname)
        keys.extend(k)
        d_obs.append(np.asarray(dv.measured_spectra, dtype=float)[mask])
        blocks.append(np.asarray(cov["value"], dtype=float)[np.ix_(mask, mask)])
        slices[lname] = slice(start, start + len(k))
        key_fields[lname] = element_key_fields(dv)
        start += len(k)

    if len(set(keys)) != len(keys):
        raise ValueError(
            "Data vector elements are not uniquely identified by "
            "(likelihood, spectrum_type, zbin0, zbin1, separation[, ell]); "
            "cannot match elements between configs."
        )

    return ModelElements(
        model=model,
        keys=keys,
        d_obs=np.concatenate(d_obs) if d_obs else np.zeros(0),
        cov=block_diag(*blocks) if blocks else np.zeros((0, 0)),
        slices=slices,
        key_fields=key_fields,
    )


def resolve_mode(keys_a, keys_b, has_cross_cov):
    """Resolve the test mode from how predicted elements relate to conditioned ones.

    Args:
        keys_a: Element keys the chain conditioned on.
        keys_b: Element keys the prediction config predicts.
        has_cross_cov: Whether an external cross-covariance was supplied.

    Returns:
        One of GOODNESS_OF_FIT, DISJOINT_INDEPENDENT, DISJOINT_CONDITIONAL,
        NESTED_CONDITIONAL.

    Raises:
        ValueError: For a partial overlap, which is neither a clean
            goodness-of-fit test nor a clean conditional prediction.
    """
    set_a, set_b = set(keys_a), set(keys_b)

    if set_a == set_b:
        return GOODNESS_OF_FIT
    if not (set_a & set_b):
        return DISJOINT_CONDITIONAL if has_cross_cov else DISJOINT_INDEPENDENT
    if set_a < set_b:
        return NESTED_CONDITIONAL

    if set_b < set_a:
        raise ValueError(
            "The prediction config predicts a strict subset of the data the "
            "chain conditioned on. Conditioning on data that includes everything "
            "being predicted leaves a degenerate (zero) predictive covariance. "
            "Widen the prediction config, or narrow the chain config."
        )
    raise ValueError(
        f"Prediction and conditioning data vectors partially overlap "
        f"({len(set_a & set_b)} shared, {len(set_a - set_b)} conditioned-only, "
        f"{len(set_b - set_a)} predicted-only elements). Supported cases are "
        "identical, disjoint, or prediction-is-a-superset."
    )


# ----------------------------------------------------------------------
# Gaussian conditional and the test statistic
# ----------------------------------------------------------------------
def conditional_moments(mu1, mu2, d1_obs, C11, C12, C22):
    """Gaussian conditional mean and covariance of d2 given d1.

        mu_{2|1} = mu2 + C21 C11^-1 (d1 - mu1)
        C_{2|1}  = C22 - C21 C11^-1 C12

    The covariance does not depend on the parameters, so it is returned
    unbatched while the mean is batched over posterior samples.

    Args:
        mu1: (n_samples, n1) model prediction for the conditioned-on block.
        mu2: (n_samples, n2) model prediction for the predicted block.
        d1_obs: (n1,) observed values of the conditioned-on block.
        C11: (n1, n1) covariance of the conditioned-on block.
        C12: (n1, n2) cross-covariance.
        C22: (n2, n2) covariance of the predicted block.

    Returns:
        Tuple of (mu_cond, cov_cond) with shapes (n_samples, n2) and (n2, n2).
    """
    solve = np.linalg.solve(C11, C12)  # C11^-1 C12, shape (n1, n2)
    cov_cond = C22 - C12.T @ solve
    mu_cond = mu2 + (d1_obs[None, :] - mu1) @ solve

    return mu_cond, 0.5 * (cov_cond + cov_cond.T)


def ppd_pvalue(d2_obs, mu_cond, cov_cond, rng):
    """Draw replicas and compute the posterior predictive p-value.

    For each posterior sample i a replica is drawn from N(mu_cond_i, cov_cond)
    and the chi2-like discrepancy T(d, Theta) = (d - mu)^T C^-1 (d - mu) is
    evaluated for both the replica and the real data.

    Note:
        The returned p-value is the raw (uncalibrated) one.  The calibrated
        p-value of Doux et al. would wrap this: simulate many data vectors from
        a fiducial cosmology, importance-reweight the fiducial chain onto each,
        recompute p for each, and report the fraction falling below the observed
        p.  That needs a separate fiducial chain and is not implemented here.

    Args:
        d2_obs: (n_pred,) observed values of the predicted elements.
        mu_cond: (n_samples, n_pred) conditional predictive means.
        cov_cond: (n_pred, n_pred) conditional predictive covariance.
        rng: A numpy Generator.

    Returns:
        Tuple of (p_value, chi2_rep, chi2_obs, d_rep).

    Raises:
        ValueError: If any input is non-finite.  Neither ``np.linalg.cholesky``
            nor the replica draw complains about a NaN, and ``np.mean(chi2_rep >
            chi2_obs)`` counts a NaN comparison as False, so an unguarded NaN
            would silently bias the p-value downwards.
        np.linalg.LinAlgError: If cov_cond is not positive definite.
    """
    n_samples, n_pred = mu_cond.shape

    if not np.isfinite(cov_cond).all():
        raise ValueError(
            f"The conditional predictive covariance has "
            f"{np.count_nonzero(~np.isfinite(cov_cond))} non-finite entries. This "
            "comes from the covariance matrices themselves rather than the model "
            "predictions; check the covariance each likelihood loaded."
        )
    if not np.isfinite(d2_obs).all():
        raise ValueError(
            f"The observed data vector has "
            f"{np.count_nonzero(~np.isfinite(d2_obs))} non-finite entries at indices "
            f"{np.flatnonzero(~np.isfinite(d2_obs))[:10]}..."
        )
    if not np.isfinite(mu_cond).all():
        bad = ~np.isfinite(mu_cond).all(axis=1)
        raise ValueError(
            f"{bad.sum()} of {n_samples} conditional predictive means are non-finite. "
            "posterior_predictive_test() filters these out; call it rather than "
            "ppd_pvalue() directly, or drop the offending samples yourself."
        )

    try:
        chol = np.linalg.cholesky(cov_cond)
    except np.linalg.LinAlgError as exc:
        w = np.linalg.eigvalsh(cov_cond)
        raise np.linalg.LinAlgError(
            "The conditional predictive covariance is not positive definite "
            f"(smallest eigenvalue {w.min():.4e}, largest {w.max():.4e}). This "
            "usually means the conditioned-on and predicted blocks are nearly "
            "degenerate, or the supplied cross-covariance is inconsistent with "
            "the two auto-covariances."
        ) from exc

    # T(d_rep, Theta) is |z|^2 by construction, which avoids a second solve.
    z = rng.standard_normal((n_samples, n_pred))
    d_rep = mu_cond + z @ chol.T
    chi2_rep = np.einsum("ij,ij->i", z, z)

    cho = cho_factor(cov_cond)
    resid = d2_obs[None, :] - mu_cond
    chi2_obs = np.einsum("ij,ij->i", resid, cho_solve(cho, resid.T).T)

    p_value = float(np.mean(chi2_rep > chi2_obs))

    return p_value, chi2_rep, chi2_obs, d_rep


# ----------------------------------------------------------------------
# Chain, parameter and prediction plumbing
# ----------------------------------------------------------------------
def _load_chain(chain, burn_in_frac):
    """Return (model, samples, names) from a config path or an explicit tuple."""
    if isinstance(chain, (str, os.PathLike)):
        model, gds, _ = load_model_samples(
            chain, compute_sigma8=False, burn_in_frac=burn_in_frac
        )
        if gds is None:
            raise FileNotFoundError(
                f"No chain samples found for config {chain!r}. Run the chain first, "
                "or pass an explicit (model, samples, names) tuple."
            )
        names = list(model.param_names)
        samples = np.asarray(gds.samples, dtype=float)[:, : len(names)]
        return model, samples, names

    if isinstance(chain, tuple) and len(chain) == 3:
        model, samples, names = chain
        return model, np.asarray(samples, dtype=float), list(names)

    raise TypeError(
        "chain must be a path to the YAML config of a completed run, or a "
        f"(model, samples, param_names) tuple; got {type(chain).__name__}."
    )


def _resolve_model(config, default):
    """Turn a config path / Model / None into a Model."""
    if config is None:
        return default
    if hasattr(config, "likelihoods"):
        return config

    return Model(config)


def sample_prior(prior, param_names, n, rng):
    """Draw independent samples from the priors of the named parameters.

    Deliberately does not use ``Prior.initial_position``, which collapses to a
    narrow ``proposal`` ball whenever a ``ref`` value is present and so is not a
    prior draw.

    Args:
        prior: A Prior instance.
        param_names: Names of the parameters to draw.
        n: Number of samples.
        rng: A numpy Generator.

    Returns:
        Dict mapping parameter name to an (n,) array of draws.
    """
    constrained = set()
    for group in getattr(prior, "joint_prior_groups", {}).values():
        constrained.update(group["params"])
    for group in getattr(prior, "linear_constraint_groups", {}).values():
        constrained.update(group["params"])

    out = {}
    reference = None
    for p in param_names:
        pi = prior.prior_info[p]
        if p in constrained:
            if reference is None:
                reference = prior.get_reference_point()
            warnings.warn(
                f"Parameter {p!r} belongs to a joint or linear-constraint prior, "
                "which cannot be sampled independently; fixing it to its "
                "reference value for the PPD.",
                UserWarning,
            )
            out[p] = np.full(n, float(reference[p]))
        elif pi["dist"] == "uniform":
            out[p] = rng.uniform(pi["min"], pi["max"], size=n)
        elif pi["dist"] == "norm":
            out[p] = rng.normal(pi["loc"], pi["scale"], size=n)
        else:
            raise NotImplementedError(
                f"Cannot draw parameter {p!r} from prior form {pi['dist']!r}."
            )

    return out


def _batched_map(fn, args, n, batch_size, use_vmap=True, label=""):
    """Apply `fn` over the leading axis of the pytree `args`, in chunks.

    Tries ``jax.vmap`` first and falls back to a Python loop, since not every
    pipeline module is guaranteed to batch cleanly.

    Args:
        fn: Callable taking one unbatched pytree and returning a pytree.
        args: Pytree whose leaves all have the same leading axis of length n.
        n: Number of samples.
        batch_size: Chunk size for the vmapped path.
        use_vmap: If False, go straight to the loop.
        label: Name used in the fallback warning.

    Returns:
        The output pytree with numpy leaves stacked along a leading axis.
    """
    tree_map = jax.tree_util.tree_map

    if use_vmap:
        try:
            vfn = jax.jit(jax.vmap(fn))
            chunks = []
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                chunks.append(vfn(tree_map(lambda x: x[start:stop], args)))
            return tree_map(
                lambda *xs: np.concatenate([np.asarray(x, dtype=float) for x in xs]),
                *chunks,
            )
        except Exception as exc:  # noqa: BLE001 - fall back on any tracing failure
            warnings.warn(
                f"Batched evaluation of {label or 'the model'} failed "
                f"({type(exc).__name__}: {exc}); falling back to a sample-by-sample "
                "loop, which is slower.",
                UserWarning,
            )

    outs = [fn(tree_map(lambda x: x[i], args)) for i in range(n)]

    return tree_map(
        lambda *xs: np.stack([np.asarray(x, dtype=float) for x in xs]), *outs
    )


def _am_params(pred_like, chain_like, params_chain, n, rng, sample_am_params,
               batch_size, use_vmap, lname):
    """Values for a likelihood's analytically marginalized parameters.

    The linear nuisance parameters are marginalized out of the likelihood, so
    they are absent from the chain and have to be reinstated to form a
    prediction.  When the same likelihood also exists in the chain's model, its
    conditional posterior p(a | Theta, d1) is exact and cheap -- it reuses the
    Va/Lab already built by GaussianLikelihood.  Crucially the moments come from
    the *chain's* likelihood, whose data vector is d1 alone, so no information
    about the data being predicted leaks into the prediction.

    Otherwise the parameters are unconstrained by d1 and are drawn from their
    prior.

    Args:
        pred_like: The likelihood whose predictions are being formed.
        chain_like: Same-named likelihood in the chain's model, or None.
        params_chain: Dict of (n,) arrays covering chain_like's sampled params.
        n: Number of samples.
        rng: A numpy Generator.
        sample_am_params: If False, use the prior means (debugging only).
        batch_size: Chunk size for batched evaluation.
        use_vmap: Whether to attempt vmap.
        lname: Likelihood name, for messages.

    Returns:
        Dict mapping AM parameter name to an (n,) array, empty if the likelihood
        has no analytically marginalized parameters.
    """
    if pred_like.Nlin == 0:
        return {}

    names = [str(k) for k in pred_like.linear_params_names]
    means = np.array([pred_like.linear_params_means[k] for k in names], dtype=float)
    stds = np.asarray(pred_like.linear_params_stds, dtype=float)

    if not sample_am_params:
        return {p: np.full(n, means[i]) for i, p in enumerate(names)}

    usable_chain_like = (
        chain_like is not None
        and chain_like.Nlin == pred_like.Nlin
        and [str(k) for k in chain_like.linear_params_names] == names
    )

    if not usable_chain_like:
        if chain_like is not None:
            warnings.warn(
                f"Likelihood {lname!r} has analytically marginalized parameters that "
                "do not match those of the chain's likelihood of the same name; "
                "drawing them from their prior instead of their conditional posterior.",
                UserWarning,
            )
        draws = means[None, :] + rng.standard_normal((n, len(names))) * stds[None, :]
        return {p: draws[:, i] for i, p in enumerate(names)}

    mean, cov = _batched_map(
        chain_like.am_conditional_moments,
        {k: params_chain[k] for k in chain_like.sampled_params},
        n,
        batch_size,
        use_vmap=use_vmap,
        label=f"analytic-marginalization moments for {lname!r}",
    )
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    # The conditional covariance depends on Theta, so factorize per sample.
    normal = rng.standard_normal(mean.shape)
    draws = np.empty_like(mean)
    for i in range(n):
        draws[i] = mean[i] + np.linalg.cholesky(0.5 * (cov[i] + cov[i].T)) @ normal[i]

    return {p: draws[:, i] for i, p in enumerate(names)}


def _predict(elements, params_all, chain_model, n, rng, sample_am_params,
             batch_size, use_vmap):
    """Model predictions for every element of a ModelElements view.

    Args:
        elements: ModelElements for the model doing the predicting.
        params_all: Dict mapping every sampled parameter of that model to an
            (n,) array of values.
        chain_model: The chain's Model, used to source analytically marginalized
            parameters from their conditional posterior given d1.
        n: Number of samples.
        rng: A numpy Generator.
        sample_am_params: Whether to sample AM parameters or use prior means.
        batch_size: Chunk size for batched evaluation.
        use_vmap: Whether to attempt vmap.

    Returns:
        (n, n_elements) array of predictions in element order.
    """
    out = np.empty((n, len(elements.keys)), dtype=float)

    for lname, like in elements.model.likelihoods.items():
        chain_like = chain_model.likelihoods.get(lname) if chain_model else None
        params_like = {k: params_all[k] for k in like.sampled_params}

        params_am = _am_params(
            like, chain_like, params_all, n, rng, sample_am_params,
            batch_size, use_vmap, lname,
        )

        def predict(args, _like=like):
            params, params_am_i = args
            return _like.predict_model(params, params_am_i if params_am_i else {})

        preds = _batched_map(
            predict,
            (params_like, params_am),
            n,
            batch_size,
            use_vmap=use_vmap,
            label=f"predictions for {lname!r}",
        )
        out[:, elements.slices[lname]] = np.asarray(preds, dtype=float)

    return out


def _build_params(model, samples, names, n, rng):
    """Assemble every sampled parameter of `model`, drawing what the chain lacks.

    Returns:
        Tuple of (params, drawn), where `drawn` lists the parameters that came
        from the prior rather than the chain.  Those are the ones
        :func:`_predict_finite` is allowed to redraw.
    """
    params = {p: samples[:, names.index(p)] for p in model.param_names if p in names}

    missing = [p for p in model.param_names if p not in names]
    if missing:
        warnings.warn(
            f"{len(missing)} parameter(s) of the prediction model are not sampled in "
            f"the chain and will be drawn from their prior: {', '.join(sorted(missing))}.",
            UserWarning,
        )
        params.update(sample_prior(model.prior, missing, n, rng))

    return params, missing


def _predict_finite(elements, params, drawn, chain_model, n, rng, sample_am_params,
                    batch_size, use_vmap, retries):
    """Predict, redrawing prior-drawn parameters until the prediction is finite.

    Parameters the chain did not sample are drawn from their prior, which is
    free to wander outside the region where the theory pipeline evaluates (an
    emulator's training bounds, say).  Redrawing just those parameters for the
    offending samples amounts to using the prior truncated to the model's
    support, which is what is meant anyway, and it keeps the posterior sample
    set intact -- unlike dropping the sample, which throws away a perfectly good
    draw of the parameters that *were* constrained by the chain.

    Retries only ever touch the failing samples, so the common case of no NaN at
    all costs one extra ``isfinite`` pass.  Samples still non-finite after
    `retries` attempts are left that way for the caller to drop.

    Args:
        elements: ModelElements for the model doing the predicting.
        params: Dict of (n,) parameter arrays; the `drawn` entries are updated
            in place with whatever redraw succeeded.
        drawn: Names of the prior-drawn parameters, from :func:`_build_params`.
        chain_model: The chain's Model, passed through to :func:`_predict`.
        n: Number of samples.
        rng: A numpy Generator.
        sample_am_params: Whether to sample AM parameters or use prior means.
        batch_size: Chunk size for batched evaluation.
        use_vmap: Whether to attempt vmap.
        retries: Maximum number of redraw rounds.

    Returns:
        (n, n_elements) array of predictions.
    """
    mu = _predict(elements, params, chain_model, n, rng, sample_am_params,
                  batch_size, use_vmap)

    if not drawn:
        return mu

    for name in drawn:
        params[name] = np.array(params[name], dtype=float)

    for _ in range(retries):
        idx = np.flatnonzero(~np.isfinite(mu).all(axis=1))
        if idx.size == 0:
            break

        trial = {k: np.asarray(v)[idx] for k, v in params.items()}
        trial.update(sample_prior(elements.model.prior, drawn, idx.size, rng))

        mu_trial = _predict(elements, trial, chain_model, idx.size, rng,
                            sample_am_params, batch_size, use_vmap)

        ok = np.isfinite(mu_trial).all(axis=1)
        if not ok.any():
            continue

        mu[idx[ok]] = mu_trial[ok]
        for name in drawn:
            params[name][idx[ok]] = trial[name][ok]

    return mu


def _usable_sample_mask(d2_obs, mu_cond, cov_cond, max_dropped_frac,
                        max_residual_sigma):
    """Samples whose predictive mean is finite and on the scale of the data.

    A theory pipeline pushed outside its region of validity fails in two ways,
    and only one of them announces itself.  A NaN is caught by any finite check;
    a prediction that merely blows up to some enormous *finite* value passes
    every such check and then dominates the test statistic, dragging the p-value
    down as surely as a genuine misfit.  Both are filtered here, on the
    conditional mean rather than the raw predictions so that the outlier
    criterion can be expressed in units of the predictive sigma and needs no
    knowledge of the data vector's absolute scale.

    Dropping samples is only harmless while there are few of them.  Once the
    fraction is appreciable the failures are correlated with a region of
    parameter space, and the p-value silently becomes conditional on wherever
    the pipeline happens to work -- so past `max_dropped_frac` this raises
    rather than quietly returning a biased answer.

    Args:
        d2_obs: (n_pred,) observed values of the predicted elements.
        mu_cond: (n_samples, n_pred) conditional predictive means.
        cov_cond: (n_pred, n_pred) conditional predictive covariance.
        max_dropped_frac: Largest tolerable fraction of dropped samples.
        max_residual_sigma: Drop a sample if any element's predictive residual
            exceeds this many sigma. None disables the outlier test, leaving
            only the finite check.

    Returns:
        Tuple of (keep, n_nonfinite, n_outlier) with keep an (n_samples,)
        boolean mask.

    Raises:
        ValueError: If more than `max_dropped_frac` of samples are dropped, or
            if no samples survive.
    """
    n = len(mu_cond)
    finite = np.isfinite(mu_cond).all(axis=1)
    inrange = np.ones(n, dtype=bool)

    sigma = np.sqrt(np.diag(cov_cond))
    if max_residual_sigma is not None and np.isfinite(sigma).all() and (sigma > 0).all():
        z = np.abs(d2_obs[None, :] - mu_cond[finite]) / sigma[None, :]
        inrange[finite] = (z <= max_residual_sigma).all(axis=1)
    # A non-finite sigma is a covariance problem rather than a prediction one;
    # leave it for ppd_pvalue to report, which says so in as many words.

    keep = finite & inrange
    n_nonfinite = int((~finite).sum())
    n_outlier = int((finite & ~inrange).sum())
    n_bad = n_nonfinite + n_outlier
    if n_bad == 0:
        return keep, 0, 0

    what = f"{n_nonfinite} non-finite, {n_outlier} beyond {max_residual_sigma:g} sigma"
    frac = n_bad / n
    if n_bad == n:
        raise ValueError(
            f"Every posterior sample predicts unusably ({what}). The prediction "
            "pipeline is broken over the whole posterior, not just at its edges."
        )
    if frac > max_dropped_frac:
        raise ValueError(
            f"{n_bad} of {n} posterior samples ({100 * frac:.2f}%) predict unusably "
            f"({what}), above max_dropped_frac={max_dropped_frac:g}. Dropping this "
            "many would condition the p-value on the region where the pipeline "
            "happens to work. Fix the predictions, tighten the priors on the "
            "parameters the chain does not constrain, or raise max_dropped_frac "
            "deliberately."
        )

    warnings.warn(
        f"{n_bad} of {n} posterior samples ({100 * frac:.3f}%) predict unusably "
        f"({what}) and are dropped from the PPD.",
        UserWarning,
    )

    return keep, n_nonfinite, n_outlier


def _concat_masks(model):
    """Scale-cut indices of every likelihood, offset into the concatenated vector."""
    idx, offset = [], 0
    for like in model.likelihoods.values():
        dv = like.observed_data_vector
        idx.append(np.asarray(dv.scale_mask) + offset)
        offset += dv.n_dv

    return np.concatenate(idx) if idx else np.zeros(0, dtype=int)


def _load_cross_covariance(cross_covariance, elements_a, elements_b):
    """Validate and shape-check an externally supplied cross-covariance."""
    if cross_covariance is None:
        return None

    if isinstance(cross_covariance, (str, os.PathLike)):
        cross_covariance = np.load(cross_covariance)
    cross_covariance = np.asarray(cross_covariance, dtype=float)

    n_a, n_b = len(elements_a.keys), len(elements_b.keys)
    if cross_covariance.shape == (n_a, n_b):
        return cross_covariance
    if cross_covariance.shape == (n_b, n_a) and n_a != n_b:
        raise ValueError(
            f"cross_covariance has shape {cross_covariance.shape}, the transpose of "
            f"the expected (n_conditioned, n_predicted) = ({n_a}, {n_b}). Pass it "
            "with the conditioned-on data along the first axis."
        )

    unmasked_a = sum(
        like.observed_data_vector.n_dv for like in elements_a.model.likelihoods.values()
    )
    unmasked_b = sum(
        like.observed_data_vector.n_dv for like in elements_b.model.likelihoods.values()
    )
    if cross_covariance.shape == (unmasked_a, unmasked_b):
        return cross_covariance[
            np.ix_(_concat_masks(elements_a.model), _concat_masks(elements_b.model))
        ]

    raise ValueError(
        f"cross_covariance has shape {cross_covariance.shape}; expected the masked "
        f"({n_a}, {n_b}) or the unmasked ({unmasked_a}, {unmasked_b})."
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def posterior_predictive_test(
    chain,
    prediction_config=None,
    cross_covariance=None,
    burn_in_frac=0.3,
    thin=1,
    n_samples=None,
    seed=0,
    sample_am_params=True,
    batch_size=64,
    use_vmap=True,
    max_dropped_frac=0.01,
    max_residual_sigma=1e3,
    nonfinite_retries=10,
):
    """Run a posterior predictive distribution test on a completed chain.

    With no ``prediction_config`` this is a goodness-of-fit test: replica data
    vectors are drawn for exactly the data the chain was fit to.  With one, the
    test mode is resolved by comparing the two data vectors element by element
    (see :func:`resolve_mode`):

    * disjoint data, no ``cross_covariance`` -- the predicted block is treated
      as independent of the conditioned-on block;
    * disjoint data, with ``cross_covariance`` -- the conditional Gaussian shift
      is applied using the supplied cross-covariance;
    * the prediction config predicts the conditioned-on data *and* more -- the
      new elements are predicted conditional on the shared ones, with all three
      covariance blocks read out of the prediction config's own covariance.

    Args:
        chain: Path to the YAML config of a completed run, or an explicit
            ``(model, samples, param_names)`` tuple.
        prediction_config: Optional path to a second YAML config, or a Model,
            describing the data to predict. Defaults to the chain's own model.
        cross_covariance: Optional (n_conditioned, n_predicted) cross-covariance,
            as an array or a path to a ``.npy`` file. Only used when the two data
            vectors are disjoint; ignored otherwise, since a superset prediction
            config already carries the cross terms.
        burn_in_frac: Fraction of each chain discarded as burn-in.
        thin: Keep every ``thin``-th post-burn-in sample.
        n_samples: If set, randomly subsample down to this many samples.
        seed: Seed for the replica draws, prior draws and subsampling.
        sample_am_params: Draw analytically marginalized parameters from their
            conditional posterior. Set False to pin them to their prior means.
        batch_size: Number of posterior samples per batched model evaluation.
        use_vmap: Attempt ``jax.vmap`` batching before falling back to a loop.
        max_dropped_frac: Largest fraction of samples that may be dropped for
            predicting unusably before this raises instead.
        max_residual_sigma: Drop a sample if its predictive residual exceeds this
            many sigma on any element. Guards against a theory pipeline that
            blows up to a large *finite* value outside its region of validity,
            which no finiteness check catches and which would otherwise dominate
            the test statistic. The default is deliberately far beyond anything a
            real misfit produces, so it only removes the unambiguous; None
            disables it.
        nonfinite_retries: How many times to redraw the prior-drawn parameters of
            a sample whose prediction is non-finite before giving up on it.

    Returns:
        A PPDResult.

    Raises:
        ValueError: If more than `max_dropped_frac` of the samples predict
            unusably; see :func:`_usable_sample_mask`.
    """
    rng = np.random.default_rng(seed)

    chain_model, samples, names = _load_chain(chain, burn_in_frac)
    samples = samples[::thin]
    if n_samples is not None and n_samples < len(samples):
        samples = samples[rng.choice(len(samples), size=n_samples, replace=False)]
    n = len(samples)
    if n == 0:
        raise ValueError("No posterior samples left after burn-in and thinning.")

    pred_model = _resolve_model(prediction_config, chain_model)
    elements_a = collect_elements(chain_model)
    elements_b = (
        elements_a if pred_model is chain_model else collect_elements(pred_model)
    )

    cross_cov = _load_cross_covariance(cross_covariance, elements_a, elements_b)
    mode = resolve_mode(elements_a.keys, elements_b.keys, cross_cov is not None)

    if mode == NESTED_CONDITIONAL and cross_cov is not None:
        warnings.warn(
            "cross_covariance was supplied but the prediction config already "
            "predicts the conditioned-on data, so its own covariance carries the "
            "cross terms; the supplied cross-covariance is ignored.",
            UserWarning,
        )

    if mode == GOODNESS_OF_FIT:
        params, drawn = _build_params(chain_model, samples, names, n, rng)
        mu_cond = _predict_finite(elements_a, params, drawn, chain_model, n, rng,
                                  sample_am_params, batch_size, use_vmap,
                                  nonfinite_retries)
        cov_cond = 0.5 * (elements_a.cov + elements_a.cov.T)
        d2_obs, pred_keys = elements_a.d_obs, elements_a.keys
        pred_key_fields = elements_a.key_fields
        params_used, drawn_used = params, drawn

    elif mode == NESTED_CONDITIONAL:
        shared = set(elements_a.keys)
        position = {k: i for i, k in enumerate(elements_b.keys)}
        idx1 = np.array([position[k] for k in elements_a.keys])
        idx2 = np.array([i for i, k in enumerate(elements_b.keys) if k not in shared])

        if not np.allclose(elements_b.d_obs[idx1], elements_a.d_obs, rtol=1e-6,
                           atol=0.0, equal_nan=True):
            warnings.warn(
                "The shared elements have different observed values in the chain and "
                "prediction configs; using the prediction config's values.",
                UserWarning,
            )

        params, drawn = _build_params(pred_model, samples, names, n, rng)
        mu = _predict_finite(elements_b, params, drawn, chain_model, n, rng,
                             sample_am_params, batch_size, use_vmap,
                             nonfinite_retries)

        cov = elements_b.cov
        mu_cond, cov_cond = conditional_moments(
            mu[:, idx1], mu[:, idx2], elements_b.d_obs[idx1],
            cov[np.ix_(idx1, idx1)], cov[np.ix_(idx1, idx2)], cov[np.ix_(idx2, idx2)],
        )
        d2_obs = elements_b.d_obs[idx2]
        pred_keys = [elements_b.keys[i] for i in idx2]
        pred_key_fields = elements_b.key_fields
        params_used, drawn_used = params, drawn

    else:  # disjoint, with or without a cross-covariance
        params_b, drawn_b = _build_params(pred_model, samples, names, n, rng)
        mu2 = _predict_finite(elements_b, params_b, drawn_b, chain_model, n, rng,
                              sample_am_params, batch_size, use_vmap,
                              nonfinite_retries)

        if mode == DISJOINT_INDEPENDENT:
            mu_cond = mu2
            cov_cond = 0.5 * (elements_b.cov + elements_b.cov.T)
            params_used, drawn_used = params_b, drawn_b
        else:
            params_a, drawn_a = _build_params(chain_model, samples, names, n, rng)
            mu1 = _predict_finite(elements_a, params_a, drawn_a, chain_model, n, rng,
                                  sample_am_params, batch_size, use_vmap,
                                  nonfinite_retries)
            mu_cond, cov_cond = conditional_moments(
                mu1, mu2, elements_a.d_obs, elements_a.cov, cross_cov,
                elements_b.cov,
            )
            # The two models draw their unsampled parameters independently; where
            # they share a name the prediction model's draw is the one reported.
            params_used = {**params_a, **params_b}
            drawn_used = sorted(set(drawn_a) | set(drawn_b))

        d2_obs, pred_keys = elements_b.d_obs, elements_b.keys
        pred_key_fields = elements_b.key_fields

    # A single filtering step, after the conditional so that an unusable
    # prediction in either block has already propagated into mu_cond.
    keep, n_nonfinite, n_outlier = _usable_sample_mask(
        d2_obs, mu_cond, cov_cond, max_dropped_frac, max_residual_sigma
    )
    mu_cond = mu_cond[keep]

    p_value, chi2_rep, chi2_obs, d_rep = ppd_pvalue(d2_obs, mu_cond, cov_cond, rng)
    n_kept = len(mu_cond)
    params_kept = {
        p: np.asarray(v, dtype=float)[keep] for p, v in params_used.items()
    }

    return PPDResult(
        p_value=p_value,
        p_value_error=float(np.sqrt(p_value * (1.0 - p_value) / n_kept)),
        mode=mode,
        chi2_obs=chi2_obs,
        chi2_rep=chi2_rep,
        d_rep=d_rep,
        d_obs=d2_obs,
        mu_cond=mu_cond,
        cov_cond=cov_cond,
        element_keys=pred_keys,
        key_fields=pred_key_fields,
        n_samples=n_kept,
        n_pred=len(d2_obs),
        n_nonfinite=n_nonfinite,
        n_outlier=n_outlier,
        params=params_kept,
        drawn_params=list(drawn_used),
        sample_index=np.flatnonzero(keep),
    )


# ----------------------------------------------------------------------
# Diagnostics and CLI
# ----------------------------------------------------------------------
# Element key layout assumed when a PPDResult carries no key_fields, e.g. one
# built by hand.  It matches DataVector._covariance_match_fields() plus the
# leading likelihood name, with the one optional trailing field being the
# multipole order of redshift-space multipoles.
_BASE_KEY_FIELDS = ("likelihood", "spectrum_type", "zbin0", "zbin1", "separation")


def _key_field_names(key, key_fields):
    """Field names labelling the entries of one element key."""
    names = key_fields.get(key[0]) if key_fields else None
    if names is not None:
        return list(names)

    names = list(_BASE_KEY_FIELDS)
    extra = len(key) - len(names)
    if extra == 1:
        names.append("ell")
    elif extra > 1:
        names += [f"field{i}" for i in range(extra)]

    return names[: len(key)]


def _element_records(result):
    """Decode every predicted element into the fields the panels are built from.

    Anything the keys do not carry falls back to a value that still plots: a
    missing separation becomes the element's position in the data vector, so a
    result whose keys follow no known layout degrades to a single panel against
    the element index rather than failing.

    Args:
        result: A PPDResult.

    Returns:
        List of dicts with 'index', 'likelihood', 'spectrum_type', 'zbin0',
        'zbin1', 'ell' and 'separation', in data vector order.
    """
    key_fields = getattr(result, "key_fields", None) or {}

    records = []
    for i, key in enumerate(result.element_keys):
        fields = dict(zip(_key_field_names(key, key_fields), key))
        records.append(
            {
                "index": i,
                "likelihood": str(fields.get("likelihood", "")),
                "spectrum_type": str(fields.get("spectrum_type", "")),
                "zbin0": fields.get("zbin0", 0),
                "zbin1": fields.get("zbin1", 0),
                "ell": fields.get("ell"),
                "separation": float(fields.get("separation", i)),
            }
        )

    return records


def _group_panels(records):
    """Group elements into one figure per statistic and one panel per bin pair.

    Returns:
        Dict keyed by (likelihood, spectrum_type) -- one figure each -- whose
        values are dicts keyed by (zbin0, zbin1) -- one panel each -- whose
        values are dicts keyed by multipole order (None when the statistic has
        none) holding the element indices, ordered by separation, and the
        separations themselves.
    """
    groups = {}
    for r in records:
        panel = groups.setdefault((r["likelihood"], r["spectrum_type"]), {})
        series = panel.setdefault((r["zbin0"], r["zbin1"]), {})
        series.setdefault(r["ell"], []).append(r)

    out = {}
    for gkey, panels in groups.items():
        out[gkey] = {}
        for pkey, series in panels.items():
            out[gkey][pkey] = {}
            for ell, recs in series.items():
                x = np.array([r["separation"] for r in recs], dtype=float)
                order = np.argsort(x, kind="stable")
                out[gkey][pkey][ell] = (
                    np.array([r["index"] for r in recs])[order],
                    x[order],
                )

    return out


MarginalPValue = namedtuple(
    "MarginalPValue", "p_value p_value_error n_elements"
)


def marginal_pvalue(result, idx):
    """The PPD p-value restricted to a subset of the predicted elements.

    A Gaussian marginalizes by taking a sub-block, so the replicas already drawn
    for the full vector are, on the selected elements, exactly draws from the
    predictive distribution of those elements.  The subset p-value therefore
    reuses ``result.d_rep`` rather than drawing again, so the same posterior
    samples and the same replicas stand behind every number and a per-bin
    p-value cannot disagree with the full-vector one through Monte-Carlo noise
    alone.

    Note:
        These are marginal, not conditional, p-values: each asks how the model
        does on its own elements while ignoring the others, rather than given
        them.  They inherit the conservatism of the full-vector p-value -- the
        parameters were fit to all of the data, these elements included -- and
        they are correlated with each other wherever the covariance is, so the
        smallest of many is smaller than its face value suggests.

    Args:
        result: A PPDResult.
        idx: Indices of the elements to keep, into the predicted vector.

    Returns:
        A MarginalPValue. The p-value is NaN when the sub-block of the
        predictive covariance is not positive definite, which no subset of a
        well-formed covariance is, but a hand-built one can be.
    """
    idx = np.asarray(idx, dtype=int)
    n_samples = len(result.mu_cond)

    mu = result.mu_cond[:, idx]
    resid_obs = result.d_obs[idx][None, :] - mu
    resid_rep = result.d_rep[:, idx] - mu

    try:
        cho = cho_factor(result.cov_cond[np.ix_(idx, idx)])
    except (np.linalg.LinAlgError, ValueError):
        return MarginalPValue(float("nan"), float("nan"), idx.size)

    chi2_obs = np.einsum("ij,ij->i", resid_obs, cho_solve(cho, resid_obs.T).T)
    chi2_rep = np.einsum("ij,ij->i", resid_rep, cho_solve(cho, resid_rep.T).T)

    p = float(np.mean(chi2_rep > chi2_obs))

    return MarginalPValue(p, float(np.sqrt(p * (1.0 - p) / n_samples)), idx.size)


def _panel_indices(series):
    """Every element index of one panel, pooled across its multipole orders."""
    return np.concatenate([idx for idx, _ in series.values()])


def statistic_pvalues(result):
    """Marginal p-value of each summary statistic as a whole.

    Args:
        result: A PPDResult.

    Returns:
        Dict mapping (likelihood, spectrum_type) to a MarginalPValue. See
        :func:`marginal_pvalue` for how to read one.
    """
    groups = _group_panels(_element_records(result))

    return {
        key: marginal_pvalue(
            result,
            np.concatenate([_panel_indices(series) for series in panels.values()]),
        )
        for key, panels in groups.items()
    }


def panel_pvalues(result):
    """Marginal p-value of each tomographic bin pair of each statistic.

    The multipole orders of a bin pair are pooled into its one p-value, matching
    the panels of :func:`plot_ppd_panels`, which overlays them.

    Args:
        result: A PPDResult.

    Returns:
        Dict mapping (likelihood, spectrum_type, zbin0, zbin1) to a
        MarginalPValue. See :func:`marginal_pvalue` for how to read one.
    """
    groups = _group_panels(_element_records(result))

    return {
        (lname, stype, b0, b1): marginal_pvalue(result, _panel_indices(series))
        for (lname, stype), panels in groups.items()
        for (b0, b1), series in panels.items()
    }


def format_marginal_pvalues(result):
    """A text summary of the per-statistic and per-bin marginal p-values.

    Args:
        result: A PPDResult.

    Returns:
        Multi-line string, one line per statistic and one per bin pair beneath
        it, in data vector order.
    """
    panels = panel_pvalues(result)

    lines = [
        "Marginal p-values (subsets of the same posterior samples and replicas;",
        "conservative, and correlated with one another):",
    ]
    for (lname, stype), pv in statistic_pvalues(result).items():
        lines.append(
            f"  {lname} / {stype}: p = {pv.p_value:.4f} +/- {pv.p_value_error:.4f}"
            f"  (n = {pv.n_elements})"
        )
        for (pl, ps, b0, b1), q in panels.items():
            if (pl, ps) == (lname, stype):
                lines.append(
                    f"    bins ({b0}, {b1}): p = {q.p_value:.4f} "
                    f"+/- {q.p_value_error:.4f}  (n = {q.n_elements})"
                )

    return "\n".join(lines)


def _grid_positions(panel_keys, max_cols=4):
    """Place each (zbin0, zbin1) panel in the grid of one figure.

    Statistics with cross-bin pairs get the layout the data vector plotting
    routines use -- zbin0 along the columns, zbin1 along the rows -- so the
    auto-spectra sit on the diagonal and each row and column is one tomographic
    bin.  A statistic with only auto-spectra would leave everything but that
    diagonal empty, so those wrap into a compact grid instead.

    Args:
        panel_keys: Iterable of (zbin0, zbin1) pairs.
        max_cols: Columns to wrap at in the auto-spectrum-only layout.

    Returns:
        Tuple of (n_rows, n_cols, positions), positions mapping each key to its
        (row, col).
    """
    panel_keys = list(panel_keys)

    if all(b0 == b1 for b0, b1 in panel_keys):
        ordered = sorted(panel_keys)
        n_cols = min(max_cols, len(ordered))
        n_rows = int(np.ceil(len(ordered) / n_cols))

        return n_rows, n_cols, {k: divmod(i, n_cols) for i, k in enumerate(ordered)}

    b0s = sorted({b0 for b0, _ in panel_keys})
    b1s = sorted({b1 for _, b1 in panel_keys})
    positions = {(b0, b1): (b1s.index(b1), b0s.index(b0)) for b0, b1 in panel_keys}

    return len(b1s), len(b0s), positions


def _share_scope(mode, row, col):
    """What a matplotlib-style share mode groups a panel with.

    Args:
        mode: True/'all', 'row', 'col' or False/'none', as in ``plt.subplots``.
        row: Grid row of the panel.
        col: Grid column of the panel.

    Returns:
        A key identifying the group the panel shares with, or None when it
        shares with nothing.

    Raises:
        ValueError: On an unrecognized mode.
    """
    if mode is True or mode == "all":
        return "all"
    if mode == "row":
        return row
    if mode == "col":
        return col
    if mode is False or mode == "none":
        return None

    raise ValueError(
        f"Unknown axis sharing mode {mode!r}; expected True, False, 'all', "
        "'row', 'col' or 'none'."
    )


def _panel_axes(fig, n_rows, n_cols, predictions, occupied, sharex, sharey):
    """Lay out the grid of panels, one or two axes per cell.

    ``plt.subplots`` cannot express "a residual panel glued underneath its
    prediction panel, with room between the cells", so the grid is nested by
    hand -- which also means doing by hand the axis sharing and tick label
    hiding subplots would otherwise handle.  Every axes starts with its tick
    labels hidden; the caller switches them back on for the panels on the edge
    of the *occupied* region, which in a triangular or wrapped layout is not the
    edge of the grid.

    Two panels of the same cell always share their separation axis whatever
    ``sharex`` says: a residual that did not line up with the prediction above
    it would be actively misleading.  ``sharey`` never mixes the two kinds,
    which are in different units.

    Args:
        fig: The figure to populate.
        n_rows: Number of grid rows (tomographic bins, or wrapped panels).
        n_cols: Number of grid columns.
        predictions: Whether each cell carries a prediction panel as well.
        occupied: Set of (row, col) cells that hold data. The rest are left
            empty rather than created and blanked, which would otherwise drag
            the shared axis limits out to their default (0, 1) range.
        sharex: Share mode for the separation axis, as in ``plt.subplots``.
        sharey: Share mode for the value and residual axes, applied to each
            kind separately.

    Returns:
        (n_rows * (2 if predictions else 1), n_cols) object array of axes, None
        in the unoccupied cells, with the prediction panel of a cell
        immediately above its residual panel.
    """
    per_cell = 2 if predictions else 1
    axes = np.empty((per_cell * n_rows, n_cols), dtype=object)
    # Columns can sit almost flush while they share a value axis, since only the
    # leftmost carries tick labels; once they do not, every panel needs room for
    # its own.
    wspace = 0.05 if sharey in (True, "all", "row") else 0.28
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.25, wspace=wspace)

    refs = {}
    for row in range(n_rows):
        for col in range(n_cols):
            if (row, col) not in occupied:
                continue

            if predictions:
                inner = outer[row, col].subgridspec(
                    2, 1, height_ratios=[3, 1], hspace=0.04
                )
                specs = [inner[0], inner[1]]
            else:
                specs = [outer[row, col]]

            cell_main = None
            for i, spec in enumerate(specs):
                is_resid = i == len(specs) - 1
                kind = "resid" if is_resid else "main"

                x_key = ("x", _share_scope(sharex, row, col))
                y_key = (kind, _share_scope(sharey, row, col))
                if cell_main is not None:
                    share_x = cell_main
                else:
                    share_x = None if x_key[1] is None else refs.get(x_key)
                share_y = None if y_key[1] is None else refs.get(y_key)

                ax = fig.add_subplot(spec, sharex=share_x, sharey=share_y)
                # which='both': a log axis spanning under a decade labels its
                # minor ticks too, and those ignore the major-tick setting.
                ax.tick_params(which="both", labelbottom=False, labelleft=False)

                if x_key[1] is not None:
                    refs.setdefault(x_key, ax)
                if y_key[1] is not None:
                    refs.setdefault(y_key, ax)
                if not is_resid:
                    cell_main = ax
                axes[per_cell * row + i, col] = ax

    return axes


def _sorted_ells(panels):
    """Multipole orders present in a figure, with the None placeholder last."""
    ells = {ell for series in panels.values() for ell in series}

    return sorted(ells, key=lambda e: (e is None, e))


def _auto_xscale(x):
    """Log separations only when they are positive and span a decent range."""
    x = x[np.isfinite(x)]
    if x.size == 0 or x.min() <= 0.0:
        return "linear"

    return "log" if x.max() / x.min() > 20.0 else "linear"


def _x_limits(x, scale, pad=0.05):
    """Padded separation limits, computed rather than left to autoscale.

    The zero line of a residual panel is an ``axhline``, whose endpoints enter
    the data limits as x = 0 and x = 1: on a log axis autoscaling would then
    stretch every panel of the figure down to x = 1, far below any separation
    actually measured.

    Args:
        x: Every separation plotted in the figure.
        scale: 'log' or 'linear'.
        pad: Fraction of the range, or of the number of decades, to add.

    Returns:
        (low, high), or None if there is nothing plottable to bound.
    """
    x = x[np.isfinite(x)]
    if scale == "log":
        x = x[x > 0.0]
    if x.size == 0:
        return None

    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        delta = abs(lo) * 0.1 or 1.0
        return (lo / 1.1, hi * 1.1) if scale == "log" else (lo - delta, hi + delta)
    if scale == "log":
        factor = (hi / lo) ** pad
        return lo / factor, hi * factor

    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def _y_label(spectrum_type, x_power):
    """Default label for a prediction panel, noting the separation weighting."""
    if not x_power:
        return spectrum_type
    power = "" if x_power == 1 else f"^{{{x_power:g}}}"

    return rf"$x{power}\,\times\,$" + spectrum_type


def _resolve_label(label, lname, stype, tag, default):
    """Pick one figure's axis label out of what the caller supplied.

    One string labels every figure, which is what a single statistic wants; a
    dict labels them one at a time, which is what a data vector mixing angular
    separations with wavenumbers needs.  Dict lookups try the figure tag, then
    (likelihood, spectrum_type), then the spectrum type alone, so the common
    case of keying on the spectrum type stays terse.

    Args:
        label: None, a string, or a dict keyed by any of the three forms.
        lname: Likelihood name of the figure.
        stype: Spectrum type of the figure.
        tag: The figure's tag, as it appears in the returned dict's key.
        default: Label to use when nothing matches.

    Returns:
        The label string.
    """
    if label is None:
        return default
    if isinstance(label, str):
        return label

    for key in (tag, (lname, stype), stype):
        if key in label:
            return label[key]

    return default


def plot_chi2(result):
    """Replica versus observed discrepancy, the picture behind the p-value.

    Args:
        result: A PPDResult.

    Returns:
        A matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.histogram_bin_edges(
        np.concatenate([result.chi2_rep, result.chi2_obs]), bins=50
    )
    ax.hist(result.chi2_rep, bins=bins, histtype="step", density=True,
            label=r"$T(d_{\rm rep}, \Theta)$")
    ax.hist(result.chi2_obs, bins=bins, histtype="step", density=True,
            label=r"$T(d_{\rm obs}, \Theta)$")
    ax.set_xlabel(r"$T(d, \Theta)$")
    ax.set_ylabel("density")
    ax.set_title(
        f"{result.mode}: $p = {result.p_value:.3f} \\pm {result.p_value_error:.3f}$"
        f"  ($N_{{\\rm pred}} = {result.n_pred}$)"
    )
    ax.legend()

    return fig


def plot_ppd_panels(result, predictions=False, x_power=1.0, xscale=None,
                    yscale="linear", panel_size=None, resid_ylim=None,
                    xlabel=None, ylabel=None, resid_ylabel=None,
                    sharex=True, sharey="row"):
    """Residuals -- and optionally the predictions themselves -- panel by panel.

    One figure is produced per summary statistic, laid out like
    ``DataVector.plot_spectra_vs_model``: a grid of panels with ``zbin0`` along
    the columns and ``zbin1`` along the rows, and multipole orders overlaid
    within each panel.  A single flat plot of every element in data vector order
    hides which bin pair a run of outlying points belongs to, which is usually
    the first thing worth knowing.

    Each residual panel shows the observed residual in units of the predictive
    sigma against the 68% and 95% intervals of the replica residuals; the
    observed points should scatter inside the band as often as the band's
    coverage says.  Every panel is labelled inside itself with its bin pair and
    its own marginal p-value, and each figure titled with that of the whole
    statistic, so a bin pair that drives the full-vector p-value can be read
    straight off the plot; see :func:`marginal_pvalue` for what those numbers do
    and do not say.

    With ``predictions=True`` every panel gains the data and the model itself
    above it -- observed points with predictive error bars, the median model
    prediction and its 68% posterior interval -- with the residual panel
    underneath.

    Args:
        result: A PPDResult.
        predictions: Add the model-versus-data panel above each residual panel.
        x_power: Multiply plotted values by ``separation ** x_power`` in the
            prediction panels, following the data vector plotting convention of
            showing ``k P(k)`` and ``l C_l``. Residuals are unaffected.
        xscale: Separation axis scale; None picks 'log' or 'linear' per figure
            from the separations themselves.
        yscale: Value axis scale of the prediction panels.
        panel_size: (width, height) in inches of one grid cell. None uses a cell
            tall enough for whichever panels the figure has.
        resid_ylim: (low, high) limits of the residual panels. None widens the
            default +/-5 sigma just enough to keep every observed residual on
            the page, up to a limit of 50 sigma.
        xlabel: Separation axis label, defaulting to 'separation'. A string
            labels every figure; a dict keyed by figure tag, by (likelihood,
            spectrum_type) or by spectrum type labels them individually, which
            is what a data vector mixing wavenumbers with angular separations
            needs.
        ylabel: Value axis label of the prediction panels, in the same forms,
            defaulting to the spectrum type and the separation weighting. Only
            has an effect with ``predictions``, which is what creates that axis.
        resid_ylabel: Residual axis label, in the same forms, defaulting to
            (d - <mu>) / sigma.
        sharex: Separation axis sharing, as in ``plt.subplots``: True/'all',
            'row', 'col' or False/'none'. The two panels of one cell always
            share theirs regardless, since a residual that did not line up with
            the prediction above it would mislead. Panels that no longer share
            an axis keep their own tick labels.
        sharey: Value and residual axis sharing, in the same forms and applied
            to each kind separately -- they are in different units and never
            share with each other. With 'none', residual panels also autoscale
            individually instead of taking the common ``resid_ylim``.

    Returns:
        Dict of matplotlib figures, keyed 'residuals.<statistic>' or, with
        ``predictions``, 'predictions.<statistic>'. The statistic is the
        spectrum type, prefixed with the likelihood name when the result covers
        more than one likelihood.
    """
    import matplotlib.pyplot as plt

    if panel_size is None:
        panel_size = (4.0, 5.0) if predictions else (4.0, 3.0)

    sigma = np.sqrt(np.diag(result.cov_cond))
    mu_med = np.median(result.mu_cond, axis=0)
    mu_lo, mu_hi = np.percentile(result.mu_cond, [16.0, 84.0], axis=0)

    z_obs = (result.d_obs - result.mu_cond.mean(axis=0)) / sigma
    z_rep = (result.d_rep - result.mu_cond) / sigma[None, :]
    z_lo95, z_hi95 = np.percentile(z_rep, [2.5, 97.5], axis=0)
    z_lo68, z_hi68 = np.percentile(z_rep, [16.0, 84.0], axis=0)

    # An explicit resid_ylim is honoured whatever the sharing; an automatic one
    # is a common scale, which is exactly what 'none' asks not to have.
    if resid_ylim is None and _share_scope(sharey, 0, 0) is not None:
        largest = np.max(np.abs(z_obs)) if z_obs.size else 0.0
        lim = float(np.clip(1.05 * largest, 5.0, 50.0))
        resid_ylim = (-lim, lim)

    groups = _group_panels(_element_records(result))
    prefix = "predictions" if predictions else "residuals"
    named_by_likelihood = len({lname for lname, _ in groups}) > 1

    figs = {}
    for (lname, stype), panels in groups.items():
        ells = _sorted_ells(panels)
        n_rows, n_cols, positions = _grid_positions(panels)

        tag = f"{lname}_{stype}" if named_by_likelihood else stype
        x_text = _resolve_label(xlabel, lname, stype, tag, "separation")
        y_text = _resolve_label(ylabel, lname, stype, tag,
                                _y_label(stype, x_power))
        resid_text = _resolve_label(resid_ylabel, lname, stype, tag,
                                    r"$(d - \langle \mu \rangle) / \sigma$")

        occupied = set(positions.values())
        fig = plt.figure()
        fig.set_size_inches(panel_size[0] * n_cols, panel_size[1] * n_rows)
        axes = _panel_axes(fig, n_rows, n_cols, predictions, occupied,
                           sharex, sharey)

        all_x = np.concatenate([x for s in panels.values() for _, x in s.values()])
        scale = _auto_xscale(all_x) if xscale is None else xscale

        for (b0, b1), series in panels.items():
            row, col = positions[(b0, b1)]
            ax_main = axes[2 * row, col] if predictions else None
            ax_res = axes[2 * row + 1, col] if predictions else axes[row, col]

            for ell in ells:
                if ell not in series:
                    continue
                idx, x = series[ell]
                color = f"C{ells.index(ell)}"
                label = None if ell is None else rf"$\ell={ell}$"

                ax_res.fill_between(x, z_lo95[idx], z_hi95[idx], color=color,
                                    alpha=0.15, linewidth=0)
                ax_res.fill_between(x, z_lo68[idx], z_hi68[idx], color=color,
                                    alpha=0.3, linewidth=0)
                ax_res.plot(x, z_obs[idx], color=color, ls="", marker="o", ms=3,
                            label=label)

                if predictions:
                    w = x ** x_power if x_power else np.ones_like(x)
                    ax_main.errorbar(x, w * result.d_obs[idx], w * sigma[idx],
                                     color=color, ls="", marker="o", ms=3,
                                     capsize=3, label=label)
                    ax_main.plot(x, w * mu_med[idx], color=color)
                    ax_main.fill_between(x, w * mu_lo[idx], w * mu_hi[idx],
                                         color=color, alpha=0.3, linewidth=0)

            ax_res.axhline(0.0, color="k", lw=0.8)
            if resid_ylim is not None:
                ax_res.set_ylim(*resid_ylim)
            ax_res.set_xscale(scale)

            if predictions:
                ax_main.set_xscale(scale)
                ax_main.set_yscale(yscale)

            pv = marginal_pvalue(result, _panel_indices(series))
            label = f"({b0}, {b1})"
            if np.isfinite(pv.p_value):
                label += f"   $p = {pv.p_value:.3f}$"

            # Inside the panel rather than above it: a title would sit in the
            # gap between cells, which on a prediction figure is shared with the
            # residual panel of the cell above.  The box keeps it readable where
            # the data runs through the corner.
            ax_label = ax_main if predictions else ax_res
            ax_label.text(
                0.04, 0.95, label, transform=ax_label.transAxes,
                ha="left", va="top", fontsize=9,
                bbox={"facecolor": "w", "alpha": 0.7, "edgecolor": "none",
                      "pad": 1.5},
            )

        # Tick labels are hidden only where another panel shows them for this
        # one.  A column shares its separation ticks only when sharex spans the
        # column, and a row its value ticks only when sharey spans the row;
        # otherwise every panel is on its own scale and needs its own numbers.
        x_shared_down_column = sharex in (True, "all", "col")
        y_shared_along_row = sharey in (True, "all", "row")

        # Label the panels on the edge of the *occupied* region rather than of
        # the grid: in a triangular or wrapped layout the two differ, and a
        # column's lowest panel would otherwise be left without tick labels
        # because the cells below it are blank.
        for col in range(n_cols):
            rows = sorted(r for r, c in occupied if c == col)
            if not rows:
                continue
            for row in rows:
                ax = axes[2 * row + 1, col] if predictions else axes[row, col]
                if row == rows[-1]:
                    ax.set_xlabel(x_text)
                if row == rows[-1] or not x_shared_down_column:
                    ax.tick_params(which="both", labelbottom=True)

        first = None
        for row in range(n_rows):
            cols = sorted(c for r, c in occupied if r == row)
            if not cols:
                continue
            first = first if first is not None else (row, cols[0])
            for col in cols:
                ax_res = axes[2 * row + 1 if predictions else row, col]
                ax_main = axes[2 * row, col] if predictions else None
                if col == cols[0]:
                    ax_res.set_ylabel(resid_text)
                    if predictions:
                        ax_main.set_ylabel(y_text)
                if col == cols[0] or not y_shared_along_row:
                    ax_res.tick_params(which="both", labelleft=True)
                    if predictions:
                        ax_main.tick_params(which="both", labelleft=True)

        if first is not None and ells != [None]:
            row, col = first
            axes[2 * row if predictions else row, col].legend(fontsize=7)

        # Panels that share a separation axis must be bounded together, since a
        # limit set on one of them sets the whole group.
        spans = {}
        for pkey, series in panels.items():
            row, col = positions[pkey]
            scope = _share_scope(sharex, row, col)
            spans.setdefault(scope if scope is not None else pkey, []).extend(
                x for _, x in series.values()
            )

        for pkey, series in panels.items():
            row, col = positions[pkey]
            scope = _share_scope(sharex, row, col)
            xlim = _x_limits(np.concatenate(
                spans[scope if scope is not None else pkey]), scale
            )
            if xlim is None:
                continue
            for i in ((2 * row, 2 * row + 1) if predictions else (row,)):
                axes[i, col].set_xlim(*xlim)

        stat_pv = marginal_pvalue(
            result,
            np.concatenate([_panel_indices(series) for series in panels.values()]),
        )
        stat = "" if not np.isfinite(stat_pv.p_value) else (
            f"$p = {stat_pv.p_value:.3f} \\pm {stat_pv.p_value_error:.3f}$, "
        )
        fig.suptitle(
            f"{tag}  [{result.mode}: {stat}full vector "
            f"$p = {result.p_value:.3f} \\pm {result.p_value_error:.3f}$]",
            y=1.01,
        )
        figs[f"{prefix}.{tag}"] = fig

    return figs


def plot_ppd(result, predictions=False, **kwargs):
    """Graphical checks accompanying the numerical p-value.

    Args:
        result: A PPDResult.
        predictions: Plot the model predictions against the data, with the
            residuals in a subpanel underneath, rather than the residuals alone.
        **kwargs: Passed to :func:`plot_ppd_panels`.

    Returns:
        Dict of matplotlib figures: 'chi2' plus one panel figure per summary
        statistic, keyed as described in :func:`plot_ppd_panels`.
    """
    figs = {"chi2": plot_chi2(result)}
    figs.update(plot_ppd_panels(result, predictions=predictions, **kwargs))

    return figs


def save_ppd(result, filename):
    """Write a PPDResult to HDF5."""
    with h5.File(filename, "w") as f:
        f.attrs["mode"] = result.mode
        f.attrs["p_value"] = result.p_value
        f.attrs["p_value_error"] = result.p_value_error
        f.attrs["n_samples"] = result.n_samples
        f.attrs["n_pred"] = result.n_pred
        f.attrs["n_nonfinite"] = result.n_nonfinite
        f.attrs["n_outlier"] = result.n_outlier
        for name in ("chi2_obs", "chi2_rep", "d_rep", "d_obs", "mu_cond", "cov_cond"):
            f.create_dataset(name, data=getattr(result, name))
        f.create_dataset(
            "element_keys",
            data=np.array(
                ["|".join(str(p) for p in k) for k in result.element_keys],
                dtype=h5.string_dtype(),
            ),
        )

        if result.key_fields:
            group = f.create_group("key_fields")
            for lname, fields in result.key_fields.items():
                group.create_dataset(
                    lname, data=np.array(fields, dtype=h5.string_dtype())
                )

        if result.sample_index is not None:
            f.create_dataset("sample_index", data=np.asarray(result.sample_index))
        if result.drawn_params:
            f.create_dataset(
                "drawn_params",
                data=np.array(result.drawn_params, dtype=h5.string_dtype()),
            )
        group = f.create_group("params")
        for name, values in result.params.items():
            group.create_dataset(name, data=np.asarray(values))


def ppd_test_cli():
    """CLI entry point for the ``ppd-test`` command."""
    parser = argparse.ArgumentParser(
        prog="ppd-test",
        description="Posterior predictive distribution test (arXiv:2011.03410).",
    )
    parser.add_argument("config", help="YAML config of the completed chain")
    parser.add_argument("--predict-config", default=None,
                        help="YAML config describing the data to predict "
                             "(default: goodness-of-fit on the chain's own data)")
    parser.add_argument("--cross-cov", default=None,
                        help=".npy cross-covariance between the conditioned-on and "
                             "predicted data, shape (n_conditioned, n_predicted)")
    parser.add_argument("--burn-in-frac", type=float, default=0.3)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-dropped-frac", type=float, default=0.01,
                        help="Largest fraction of samples that may be dropped for "
                             "predicting unusably before this is an error")
    parser.add_argument("--max-residual-sigma", type=float, default=1e3,
                        help="Drop samples whose predictive residual exceeds this "
                             "many sigma on any element (0 disables)")
    parser.add_argument("--nonfinite-retries", type=int, default=10,
                        help="Redraw rounds for the prior-drawn parameters of a "
                             "sample whose prediction is non-finite")
    parser.add_argument("--no-vmap", action="store_true",
                        help="Skip vmap batching and evaluate sample by sample")
    parser.add_argument("--no-sample-am", action="store_true",
                        help="Pin analytically marginalized params to their prior "
                             "means instead of sampling their conditional posterior")
    parser.add_argument("--plot", action="store_true", help="Save diagnostic PDFs")
    parser.add_argument("--plot-predictions", action="store_true",
                        help="Plot the model predictions against the data with "
                             "the residuals underneath, rather than the "
                             "residuals alone; implies --plot")
    parser.add_argument("--x-power", type=float, default=1.0,
                        help="Weight the plotted predictions by separation to "
                             "this power (1 gives k P(k) and l C_l)")
    parser.add_argument("--xlabel", default=None,
                        help="Separation axis label of the panel plots "
                             "(default: 'separation')")
    parser.add_argument("--ylabel", default=None,
                        help="Value axis label of the prediction panels "
                             "(default: the spectrum type and its weighting)")
    parser.add_argument("--sharex", default="all",
                        choices=["all", "row", "col", "none"],
                        help="How far the panels share their separation axis")
    parser.add_argument("--sharey", default="row",
                        choices=["all", "row", "col", "none"],
                        help="How far the panels share their value axis")
    parser.add_argument("--output", default=None,
                        help="Output base name (default: derived from the config)")
    args = parser.parse_args()

    result = posterior_predictive_test(
        args.config,
        prediction_config=args.predict_config,
        cross_covariance=args.cross_cov,
        burn_in_frac=args.burn_in_frac,
        thin=args.thin,
        n_samples=args.n_samples,
        seed=args.seed,
        sample_am_params=not args.no_sample_am,
        batch_size=args.batch_size,
        use_vmap=not args.no_vmap,
        max_dropped_frac=args.max_dropped_frac,
        max_residual_sigma=args.max_residual_sigma or None,
        nonfinite_retries=args.nonfinite_retries,
    )
    print(result, flush=True)
    print(format_marginal_pvalues(result), flush=True)

    output = args.output
    if output is None:
        import yaml

        with open(args.config) as fp:
            cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        output = f"{cfg['output_file'].replace('.txt', '')}.ppd"

    save_ppd(result, f"{output}.h5")
    print(f"Saved PPD results to {output}.h5", flush=True)

    if args.plot or args.plot_predictions:
        figs = plot_ppd(
            result, predictions=args.plot_predictions, x_power=args.x_power,
            xlabel=args.xlabel, ylabel=args.ylabel,
            sharex=args.sharex, sharey=args.sharey,
        )
        for name, fig in figs.items():
            fig.savefig(f"{output}.{name}.pdf", bbox_inches="tight")
            print(f"Saved {output}.{name}.pdf", flush=True)
