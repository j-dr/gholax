import jax
import jax.numpy as jnp
import numpy as np
import yaml

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
