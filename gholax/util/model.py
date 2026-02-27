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
    def __init__(self, config_file):
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
        self.prior = Prior(
            prior_info, derived_params=derived_params, fixed_params=fixed_params
        )
        self.likelihood_param_index = param_idx

    def log_posterior(self, param_values):
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
        params_like = {
            k: params[k] if k in params else 0
            for k in self.likelihoods[likelihood_name].sampled_params
        }
        like = self.likelihoods[likelihood_name]
        return like.generate_training_data(params_like)


def save_model_pred():

    args = sys.argv[1:]
    no_window = '--no-window' in args
    if no_window:
        args.remove('--no-window')

    with open(args[0], 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if len(args) > 1:
        output_base = args[1]
    else:
        output_base = cfg['output_file'].replace('.txt', '.h5')

    model = Model(cfg)
    params = model.prior.get_reference_point()

    for lname in model.likelihoods:
        like = model.likelihoods[lname]
        params_am = like.linear_params_means
        params_like = {k: params[k] for k in model.likelihoods[lname].sampled_params}

        if no_window:
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


def _save_no_window(like, pred, filename):
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
                        rows.append((t, b0, b1, ell_idx * 2, sep_val,
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
    
