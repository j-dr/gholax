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
        n_samples: Number of posterior samples used.
        n_pred: Number of predicted data vector elements.
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
    n_samples: int = 0
    n_pred: int = 0

    def __str__(self):
        return (
            f"PPD test [{self.mode}]\n"
            f"  n_pred     = {self.n_pred}\n"
            f"  n_samples  = {self.n_samples}\n"
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
    fields = [spectra_field for _, spectra_field in dv._covariance_match_fields()]
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
    keys, d_obs, blocks, slices = [], [], [], {}
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
    """
    n_samples, n_pred = mu_cond.shape

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
            return _like.predict_model(params, params_am_i if params_am_i else None)

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
    """Assemble every sampled parameter of `model`, drawing what the chain lacks."""
    params = {p: samples[:, names.index(p)] for p in model.param_names if p in names}

    missing = [p for p in model.param_names if p not in names]
    if missing:
        warnings.warn(
            f"{len(missing)} parameter(s) of the prediction model are not sampled in "
            f"the chain and will be drawn from their prior: {', '.join(sorted(missing))}.",
            UserWarning,
        )
        params.update(sample_prior(model.prior, missing, n, rng))

    return params


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

    Returns:
        A PPDResult.
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
        params = _build_params(chain_model, samples, names, n, rng)
        mu_cond = _predict(elements_a, params, chain_model, n, rng, sample_am_params,
                           batch_size, use_vmap)
        cov_cond = 0.5 * (elements_a.cov + elements_a.cov.T)
        d2_obs, pred_keys = elements_a.d_obs, elements_a.keys

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

        params = _build_params(pred_model, samples, names, n, rng)
        mu = _predict(elements_b, params, chain_model, n, rng, sample_am_params,
                      batch_size, use_vmap)

        cov = elements_b.cov
        mu_cond, cov_cond = conditional_moments(
            mu[:, idx1], mu[:, idx2], elements_b.d_obs[idx1],
            cov[np.ix_(idx1, idx1)], cov[np.ix_(idx1, idx2)], cov[np.ix_(idx2, idx2)],
        )
        d2_obs = elements_b.d_obs[idx2]
        pred_keys = [elements_b.keys[i] for i in idx2]

    else:  # disjoint, with or without a cross-covariance
        params_b = _build_params(pred_model, samples, names, n, rng)
        mu2 = _predict(elements_b, params_b, chain_model, n, rng, sample_am_params,
                       batch_size, use_vmap)

        if mode == DISJOINT_INDEPENDENT:
            mu_cond = mu2
            cov_cond = 0.5 * (elements_b.cov + elements_b.cov.T)
        else:
            params_a = _build_params(chain_model, samples, names, n, rng)
            mu1 = _predict(elements_a, params_a, chain_model, n, rng,
                           sample_am_params, batch_size, use_vmap)
            mu_cond, cov_cond = conditional_moments(
                mu1, mu2, elements_a.d_obs, elements_a.cov, cross_cov, elements_b.cov
            )

        d2_obs, pred_keys = elements_b.d_obs, elements_b.keys

    p_value, chi2_rep, chi2_obs, d_rep = ppd_pvalue(d2_obs, mu_cond, cov_cond, rng)

    return PPDResult(
        p_value=p_value,
        p_value_error=float(np.sqrt(p_value * (1.0 - p_value) / n)),
        mode=mode,
        chi2_obs=chi2_obs,
        chi2_rep=chi2_rep,
        d_rep=d_rep,
        d_obs=d2_obs,
        mu_cond=mu_cond,
        cov_cond=cov_cond,
        element_keys=pred_keys,
        n_samples=n,
        n_pred=len(d2_obs),
    )


# ----------------------------------------------------------------------
# Diagnostics and CLI
# ----------------------------------------------------------------------
def plot_ppd(result):
    """Graphical checks accompanying the numerical p-value.

    Args:
        result: A PPDResult.

    Returns:
        Dict of matplotlib figures keyed by 'chi2' and 'residuals'.
    """
    import matplotlib.pyplot as plt

    figs = {}

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
    figs["chi2"] = fig

    fig, ax = plt.subplots(figsize=(10, 4))
    sigma = np.sqrt(np.diag(result.cov_cond))
    z_obs = (result.d_obs - result.mu_cond.mean(axis=0)) / sigma
    z_rep = (result.d_rep - result.mu_cond) / sigma[None, :]
    lo, hi = np.percentile(z_rep, [2.5, 97.5], axis=0)
    x = np.arange(result.n_pred)
    ax.fill_between(x, lo, hi, alpha=0.3, label="PPD 95%")
    ax.plot(x, z_obs, ".", color="k", label="observed")
    ax.axhline(0.0, lw=0.5, color="grey")
    ax.set_xlabel("data vector element")
    ax.set_ylabel(r"$(d - \langle \mu_{2|1} \rangle) / \sigma_{2|1}$")
    ax.legend()
    figs["residuals"] = fig

    return figs


def save_ppd(result, filename):
    """Write a PPDResult to HDF5."""
    with h5.File(filename, "w") as f:
        f.attrs["mode"] = result.mode
        f.attrs["p_value"] = result.p_value
        f.attrs["p_value_error"] = result.p_value_error
        f.attrs["n_samples"] = result.n_samples
        f.attrs["n_pred"] = result.n_pred
        for name in ("chi2_obs", "chi2_rep", "d_rep", "d_obs", "mu_cond", "cov_cond"):
            f.create_dataset(name, data=getattr(result, name))
        f.create_dataset(
            "element_keys",
            data=np.array(
                ["|".join(str(p) for p in k) for k in result.element_keys],
                dtype=h5.string_dtype(),
            ),
        )


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
    parser.add_argument("--no-vmap", action="store_true",
                        help="Skip vmap batching and evaluate sample by sample")
    parser.add_argument("--no-sample-am", action="store_true",
                        help="Pin analytically marginalized params to their prior "
                             "means instead of sampling their conditional posterior")
    parser.add_argument("--plot", action="store_true", help="Save diagnostic PDFs")
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
    )
    print(result, flush=True)

    output = args.output
    if output is None:
        import yaml

        with open(args.config) as fp:
            cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        output = f"{cfg['output_file'].replace('.txt', '')}.ppd"

    save_ppd(result, f"{output}.h5")
    print(f"Saved PPD results to {output}.h5", flush=True)

    if args.plot:
        for name, fig in plot_ppd(result).items():
            fig.savefig(f"{output}.{name}.pdf", bbox_inches="tight")
            print(f"Saved {output}.{name}.pdf", flush=True)
