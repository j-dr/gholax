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
    print("Running minimization", flush=True)
    res = solver.run(x0)

    x_opt_scaled = res.params
    x_opt = x_opt_scaled * sigmas + reference
    params = dict(zip(model.param_names, x_opt))

    print(f"Best-fit -logp = {float(res.state.value):.4f}", flush=True)
    #for p in params:
    #    print(f"  {p} = {float(params[p]):.6g}", flush=True)

    return params


def _apply_gaussian_covariance(like, pred):
    """Replace the likelihood's covariance with a Gaussian covariance computed
    from the full (unmasked) model prediction vector *pred*."""
    from gholax.data_vector.two_point_spectrum import TwoPointSpectrum

    dv = like.observed_data_vector
    pred_np = np.asarray(pred)

    if isinstance(dv, TwoPointSpectrum):
        model_spectra = dv.spectra_dict_from_vector(pred_np)
        cov = dv.gaussian_covariance(model_spectra=model_spectra)
    else:
        cov = dv.gaussian_covariance()

    dv.cov = cov
    cov_mask_i, cov_mask_j = np.meshgrid(dv.scale_mask, dv.scale_mask, indexing="ij")
    dv.cinv = jnp.linalg.inv(
        dv.cov["value"][cov_mask_i, cov_mask_j].reshape(dv.n_dv_masked, dv.n_dv_masked)
    )
    print("  Applied model-predicted Gaussian covariance.", flush=True)


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
        ref_params = model.prior.get_reference_point()
        print("Computing model prediction at reference parameters for Gaussian covariance...",
              flush=True)
        for lname in model.likelihoods:
            like = model.likelihoods[lname]
            params_am = like.linear_params_means
            params_like = {k: ref_params[k] for k in like.sampled_params}
            pred_ref = like.predict_model(
                params_like, params_am, return_state=False, apply_scale_mask=False
            )
            _apply_gaussian_covariance(like, pred_ref)
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
    
