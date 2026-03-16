import argparse
import jax
import jax.numpy as jnp
import numpy as np
import yaml
import sys
import h5py as h5

from gholax import likelihood
from gholax.sampler.priors import Prior

jax.config.update("jax_default_matmul_precision", "float32")


class Model:
    """Central orchestrator for Bayesian cosmological inference.

    Instantiated from a YAML config file, holds named GaussianLikelihood
    instances and exposes JAX-differentiable log-posterior evaluation.

    Attributes:
        likelihoods: Dict mapping likelihood names to GaussianLikelihood instances.
        param_names: List of sampled parameter names.
        prior: Prior instance managing parameter priors and reference values.
        likelihood_param_index: Dict mapping likelihood names to index arrays
            for slicing the global parameter vector.
    """

    def __init__(self, config_file):
        """Initialize the Model from a YAML config file or config dict.

        Parses the config, instantiates all likelihoods, collects sampled/fixed/derived
        parameters, builds the Prior, and creates parameter index mappings.

        Args:
            config_file: Path to a YAML config file, or a pre-loaded config dict.
        """
        try:
            with open(config_file, "r") as fp:
                cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        except TypeError:
            cfg = config_file

        self.likelihoods = {}
        sampled_params = []
        fixed_params = []
        derived_params = []
        param_idx = {}
        prior_info = {}

        for lname in cfg["likelihood"]:
            if lname == 'params':
                continue
            like = getattr(likelihood, lname)(cfg)
            self.likelihoods[lname] = like

            sampled_params.extend(list(like.sampled_params.keys()))
            derived_params.extend(list(like.derived_params.keys()))
            fixed_params.extend(list(like.fixed_params.keys()))
            prior_info.update(like.sampled_params)

        sampled_params = np.unique(sampled_params).tolist()
        prior_info = {p: prior_info[p] for p in sampled_params}

        for lname in self.likelihoods:
            param_idx[lname] = jnp.array(
                [
                    sampled_params.index(p)
                    for p in self.likelihoods[lname].sampled_params.keys()
                ]
            )

        self.param_names = list(prior_info.keys())
        joint_priors = cfg.get("joint_priors", {})
        self.prior = Prior(
            prior_info, derived_params=derived_params, fixed_params=fixed_params,
            joint_priors=joint_priors
        )
        self.likelihood_param_index = param_idx

    def log_posterior(self, param_values):
        """Compute the log-posterior for a parameter vector.

        Evaluates the prior and all likelihoods. Parameters are in natural
        (unscaled) units.

        Args:
            param_values: Array of parameter values in the same order as param_names.

        Returns:
            Scalar log-posterior value (NaN mapped to -inf).
        """
        param_dict_all = dict(zip(self.param_names, param_values))
        logp = self.prior.log_prior(param_dict_all)

        for lname in self.likelihoods:
            param_dict_like = dict(
                zip(
                    self.likelihoods[lname].sampled_params,
                    param_values[self.likelihood_param_index[lname]],
                )
            )
            logp += self.likelihoods[lname].compute(param_dict_like)

        return jnp.nan_to_num(logp, nan=-jnp.inf)

    def log_posterior_scaled_params(self, param_values):
        """Compute the log-posterior from normalized parameter values.

        Parameters are rescaled from normalized space (zero-mean, unit-prior-sigma)
        to natural units before evaluation.

        Args:
            param_values: Array of normalized parameter values.

        Returns:
            Scalar log-posterior value (NaN mapped to -inf).
        """
        param_values = (
            param_values * self.prior.prior_sigmas + self.prior.reference_values
        )  # rescale
        param_dict_all = dict(zip(self.param_names, param_values))
        logp = self.prior.log_prior(param_dict_all)

        for lname in self.likelihoods:
            param_dict_like = dict(
                zip(
                    self.likelihoods[lname].sampled_params,
                    param_values[self.likelihood_param_index[lname]],
                )
            )
            logp += self.likelihoods[lname].compute(param_dict_like)

        return jnp.nan_to_num(logp, nan=-jnp.inf)

    def log_likelihood(self, param_values):
        """Compute the log-likelihood (without the prior contribution).

        Args:
            param_values: Array of parameter values in natural units.

        Returns:
            Scalar log-likelihood value (NaN mapped to -inf).
        """
        logp = 0
        for lname in self.likelihoods:
            param_dict = dict(
                zip(
                    self.likelihoods[lname].sampled_params,
                    param_values[self.likelihood_param_index[lname]],
                )
            )
            logp += self.likelihoods[lname].compute(param_dict)

        return jnp.nan_to_num(logp, nan=-jnp.inf)

    def predict_model(
        self,
        likelihood_name,
        params,
        params_am=None,
        return_state=False,
        apply_scale_mask=True,
    ):
        """Generate the model prediction for a named likelihood.

        Args:
            likelihood_name: Key into self.likelihoods.
            params: Dict of parameter values (must include all sampled params).
            params_am: Optional analytically marginalized parameter values.
                Defaults to the likelihood's linear_params_means.
            return_state: If True, also return the full pipeline state dict.
            apply_scale_mask: If True, apply scale cuts to the prediction.

        Returns:
            Model prediction vector, or (prediction, state) if return_state is True.
        """
        like = self.likelihoods[likelihood_name]
        if params_am is None:
            params_am = like.linear_params_means
        params_like = {k: params[k] for k in self.likelihoods[likelihood_name].sampled_params}

        return like.predict_model(
            params_like,
            params_am,
            return_state=return_state,
            apply_scale_mask=apply_scale_mask,
        )

    def setup_training_config(self, training_requirements):
        """Configure the model for training data generation.

        Determines which pipeline modules and parameters are needed to
        produce the requested training outputs.

        Args:
            training_requirements: Dict mapping likelihood names to the
                observables required for emulator training.

        Returns:
            List of unique parameter names required for training.
        """
        all_required_params = []
        for lname in training_requirements:
            like = self.likelihoods[lname]
            required_modules, required_params = like.setup_training_requirements(
                training_requirements[lname]
            )
            like.training_modules = required_modules
            all_required_params.extend(required_params)

        self.required_params = np.unique(all_required_params).tolist()
        for p in self.required_params:
            if p in self.prior.fixed_params:
                self.required_params.remove(p)

        return self.required_params

    def generate_training_data(self, likelihood_name, params):
        """Run the likelihood pipeline to generate emulator training data.

        Args:
            likelihood_name: Key into self.likelihoods.
            params: Dict of parameter values for this evaluation.

        Returns:
            State dict containing the computed training observables.
        """
        params_like = {
            k: params[k] if k in params else 0
            for k in self.likelihoods[likelihood_name].sampled_params
        }
        like = self.likelihoods[likelihood_name]
        return like.generate_training_data(params_like)


def _minimize(model, output_base):
    """Find the maximum-a-posteriori parameters using L-BFGS optimization.

    Args:
        model: Model instance.
        output_base: Base path for output files (unused, kept for API compat).

    Returns:
        Dict mapping parameter names to best-fit values in natural units.
    """
    import jaxopt

    prior = model.prior
    sigmas = prior.get_prior_sigmas()
    reference = prior.get_reference_values()
    log_posterior = model.log_posterior_scaled_params

    jnlp = jax.jit(lambda p: -log_posterior(p))
    vgrad = jax.value_and_grad(jnlp)
    solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

    x0 = jnp.zeros(len(model.param_names))

    # --- NaN diagnostics ---
    val0 = jnlp(x0)
    grad0 = jax.grad(jnlp)(x0)
    for lname, like in model.likelihoods.items():
        dv = like.observed_data_vector
        if hasattr(dv, 'cinv') and dv.cinv is not None:
            cinv_nan = bool(jnp.any(jnp.isnan(dv.cinv)))
            cinv_inf = bool(jnp.any(jnp.isinf(dv.cinv)))
            print(f"  [{lname}] cinv has NaN: {cinv_nan}, has Inf: {cinv_inf}", flush=True)
    print(f"  -logp at x0: {float(val0):.4f}", flush=True)
    print(f"  grad at x0 has NaN: {bool(jnp.any(jnp.isnan(grad0)))}", flush=True)
    print(f"  grad at x0 has Inf: {bool(jnp.any(jnp.isinf(grad0)))}", flush=True)
    if bool(jnp.any(jnp.isnan(grad0))):
        nan_params = [model.param_names[i] for i in range(len(model.param_names)) if jnp.isnan(grad0[i])]
        print(f"  NaN gradient params: {nan_params}", flush=True)
    # --- end diagnostics ---

    print("Running minimization", flush=True)
    res = solver.run(x0)

    x_opt_scaled = res.params
    x_opt = x_opt_scaled * sigmas + reference
    params = dict(zip(model.param_names, x_opt))

    print(f"Best-fit -logp = {float(res.state.value):.4f}", flush=True)
    #for p in params:
    #    print(f"  {p} = {float(params[p]):.6g}", flush=True)

    return params


def _apply_gaussian_covariance(like, pred, model_spectra=None):
    """Replace the likelihood's covariance with a Gaussian covariance.

    If *model_spectra* is provided, use it directly (e.g. from a cross-bin
    model). Otherwise, build the spectra dict from the prediction vector *pred*.
    """
    from gholax.data_vector.two_point_spectrum import TwoPointSpectrum

    dv = like.observed_data_vector

    if isinstance(dv, TwoPointSpectrum):
        if model_spectra is None:
            model_spectra = dv.spectra_dict_from_vector(np.asarray(pred))
        cov = dv.gaussian_covariance(model_spectra=model_spectra)
    else:
        cov = dv.gaussian_covariance()

    dv.cov = cov
    cov_mask_i, cov_mask_j = np.meshgrid(dv.scale_mask, dv.scale_mask, indexing="ij")
    dv.cinv = jnp.linalg.inv(
        dv.cov["value"][cov_mask_i, cov_mask_j].reshape(dv.n_dv_masked, dv.n_dv_masked)
    )
    print("  Applied model-predicted Gaussian covariance.", flush=True)


def _needs_cross_c_dd_model(model):
    """Return True if any likelihood has auto-only (use_cross=False) c_dd spectra."""
    from gholax.data_vector.two_point_spectrum import TwoPointSpectrum

    for like in model.likelihoods.values():
        dv = like.observed_data_vector
        if isinstance(dv, TwoPointSpectrum):
            si = dv.spectrum_info
            if "c_dd" in si and not si["c_dd"]["use_cross"]:
                return True
    return False


def _set_c_dd_use_cross(cfg, value):
    """Set use_cross for c_dd spectrum in a config dict (in-place)."""
    def _walk(d):
        if isinstance(d, dict):
            if "c_dd" in d and isinstance(d["c_dd"], dict) and "use_cross" in d["c_dd"]:
                d["c_dd"]["use_cross"] = value
                return
            for v in d.values():
                _walk(v)
        elif isinstance(d, list):
            for item in d:
                _walk(item)
    _walk(cfg)


def _remove_scale_cuts(cfg):
    """Remove all scale_cuts entries from a config dict (in-place)."""
    def _walk(d):
        if isinstance(d, dict):
            d.pop("scale_cuts", None)
            for v in d.values():
                _walk(v)
        elif isinstance(d, list):
            for item in d:
                _walk(item)
    _walk(cfg)


def _set_dummy_cov(cfg, value):
    """Set dummy_cov in every data_vector config block (in-place).

    Targets dicts that contain 'data_vector_info_filename', which is the
    distinguishing key of a data_vector config block.
    """
    def _walk(d):
        if isinstance(d, dict):
            if "data_vector_info_filename" in d:
                d["dummy_cov"] = value
            else:
                for v in d.values():
                    _walk(v)
        elif isinstance(d, list):
            for item in d:
                _walk(item)
    _walk(cfg)


def save_model_pred():
    """CLI entry point for the ``save-model`` command.

    Saves model predictions at a given parameter point, with options to
    minimize, apply Gaussian covariance, load parameters from file, and
    generate comparison plots.
    """
    parser = argparse.ArgumentParser(
        prog='save-model',
        description='Save model predictions at a given parameter point.',
    )
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output file base name (default: derived from config)')
    parser.add_argument('--no-window', action='store_true',
                        help='Save pre-window (unconvolved) theory predictions')
    parser.add_argument('--minimize', action='store_true',
                        help='Run LBFGS minimizer to find best-fit parameters')
    parser.add_argument('--params', metavar='FILE',
                        help='Load best-fit parameters from an HDF5 file')
    parser.add_argument('--gauss-cov', action='store_true',
                        help='Replace covariance with a Gaussian covariance evaluated '
                             'at the reference parameter point, then run best-fit minimization')
    parser.add_argument('--plot', action='store_true',
                        help='Save PDF plots of the best-fit model vs data for each likelihood')
    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if args.output is not None:
        output_base = args.output
    else:
        output_base = cfg['output_file'].replace('.txt', '.h5')

    model = Model(cfg)

    if args.gauss_cov:
        import copy
        ref_params = model.prior.get_reference_point()

        needs_cross_model = _needs_cross_c_dd_model(model)
        if needs_cross_model:
            cross_cfg = copy.deepcopy(cfg)
            _set_c_dd_use_cross(cross_cfg, True)
            _remove_scale_cuts(cross_cfg)
            _set_dummy_cov(cross_cfg, True)
            cross_model = Model(cross_cfg)
        else:
            cross_model = model

        print("Computing model prediction at reference parameters for Gaussian covariance...",
              flush=True)
        for lname in model.likelihoods:
            like = model.likelihoods[lname]
            cross_like = cross_model.likelihoods[lname]

            params_like = {k: ref_params[k] for k in cross_like.sampled_params}
            cross_pred = cross_like.predict_model(
                params_like, cross_like.linear_params_means,
                return_state=False, apply_scale_mask=False
            )
            cross_spectra = cross_like.observed_data_vector.spectra_dict_from_vector(
                np.asarray(cross_pred)
            )
            _apply_gaussian_covariance(like, None, model_spectra=cross_spectra)

        params = _minimize(model, output_base)
    elif args.params is not None:
        params = _load_params_from_h5(args.params)
        ref = model.prior.get_reference_point()
        for p in ref:
            if p not in params:
                params[p] = ref[p]
        print(f"Loaded parameters from {args.params}")
    elif args.minimize:
        params = _minimize(model, output_base)
    else:
        params = model.prior.get_reference_point()

    for lname in model.likelihoods:
        like = model.likelihoods[lname]
        params_am = like.linear_params_means
        params_like = {k: params[k] for k in model.likelihoods[lname].sampled_params}

        if args.no_window:
            pred, state = like.predict_model(
                params_like,
                params_am,
                return_state=True,
                apply_scale_mask=False,
                apply_window=False,
            )
            output_file = f"{output_base}.{lname}.no_window"
            _save_no_window(like, pred, output_file)
            print(f"Saved no-window model prediction for {lname} to {output_file}")
        else:
            pred = like.predict_model(
                params_like,
                params_am,
                return_state=False,
                apply_scale_mask=False,
            )
            output_file = f"{output_base}.{lname}"
            like.observed_data_vector.save_data_vector(output_file, pred)
            print(f"Saved model prediction for {lname} to {output_file}")

            if args.plot:
                figs = like.observed_data_vector.plot_spectra_vs_model(pred)
                for spec_type, fig in figs.items():
                    plot_file = f"{output_file}.{spec_type}.pdf"
                    fig.savefig(plot_file, bbox_inches='tight')
                    print(f"Saved plot for {spec_type} to {plot_file}")

        if args.minimize or args.params is not None or args.gauss_cov:
            _save_params_to_h5(output_file, params)
            print(f"Saved best-fit parameters to {output_file}")


def _save_params_to_h5(filename, params):
    """Save best-fit parameters as HDF5 group attributes.

    Args:
        filename: Path to the HDF5 file (appended to).
        params: Dict mapping parameter names to scalar values.
    """
    with h5.File(filename, "a") as f:
        grp = f.create_group("best_fit_params")
        for name, val in params.items():
            grp.attrs[name] = float(val)


def _load_params_from_h5(filename):
    """Load best-fit parameters from an HDF5 file.

    Args:
        filename: Path to the HDF5 file containing a 'best_fit_params' group.

    Returns:
        Dict mapping parameter names to float values.
    """
    with h5.File(filename, "r") as f:
        return {name: float(val) for name, val in f["best_fit_params"].attrs.items()}


def _save_no_window(like, pred, filename):
    """Save pre-window (unconvolved) theory predictions to an HDF5 file.

    Writes a structured array with spectrum type, bin indices, separation
    values, and prediction values.

    Args:
        like: GaussianLikelihood instance (Nx2PTAngularPowerSpectrum or RSDPK).
        pred: Model prediction vector (before window convolution).
        filename: Output HDF5 file path.
    """
    from gholax.likelihood.nx2pt_aps import Nx2PTAngularPowerSpectrum
    from gholax.likelihood.rsd_pk import RSDPK

    dv = like.observed_data_vector
    spectrum_info = dv.spectrum_info

    if isinstance(like, Nx2PTAngularPowerSpectrum):
        sep_grid = np.array(like.ell_no_window)
        sep_name = "separation"
    elif isinstance(like, RSDPK):
        sep_grid = np.array(like.k_no_window)
        sep_name = "separation"
    else:
        raise NotImplementedError(
            f"No-window saving not implemented for {like.__class__.__name__}"
        )

    rows = []
    offset = 0
    for t in dv.spectrum_types:
        bin_pairs = []
        for ii, i in enumerate(spectrum_info[t]["bins0"]):
            if spectrum_info[t]["use_cross"]:
                from gholax.data_vector.two_point_spectrum import field_types as ft_2pt
                from gholax.data_vector.redshift_space_multipoles import field_types as ft_rsd
                ft = ft_rsd if isinstance(like, RSDPK) else ft_2pt
                if ft[t][0] == ft[t][1]:
                    bins1 = spectrum_info[t]["bins1"][ii:]
                else:
                    bins1 = spectrum_info[t]["bins1"][:]
                for j in bins1:
                    bin_pairs.append((i, j))
            else:
                bin_pairs.append((i, i))

        if isinstance(like, RSDPK):
            n_ell_multipoles = like.likelihood_pipeline[-1].n_ell
            n_sep = len(sep_grid)
            for (b0, b1) in bin_pairs:
                for ell_idx in range(n_ell_multipoles):
                    for si, sep_val in enumerate(sep_grid):
                        rows.append((t, b0, b1, ell_idx * n_ell_multipoles, sep_val,
                                     float(pred[offset])))
                        offset += 1
        else:
            n_sep = len(sep_grid)
            for (b0, b1) in bin_pairs:
                for si, sep_val in enumerate(sep_grid):
                    rows.append((t, b0, b1, sep_val,
                                 float(pred[offset])))
                    offset += 1

    if isinstance(like, RSDPK):
        dt = np.dtype([
            ("spectrum_type", "S10"),
            ("zbin0", int),
            ("zbin1", int),
            ("ell", int),
            (sep_name, float),
            ("value", float),
        ])
    else:
        dt = np.dtype([
            ("spectrum_type", "S10"),
            ("zbin0", int),
            ("zbin1", int),
            (sep_name, float),
            ("value", float),
        ])

    data = np.array(rows, dtype=dt)
    with h5.File(filename, "w") as f:
        f.create_dataset("spectra", data=data)
    
