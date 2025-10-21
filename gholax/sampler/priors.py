import warnings
from datetime import datetime

import jax
import jax.numpy as jnp
import scipy.stats


class Prior(object):
    def __init__(self, config, derived_params=[], fixed_params=[]):
        self.prior_info = {}
        for p in config:
            self.prior_info[p] = config[p]["prior"]
            if "ref" in config[p]:
                self.prior_info[p]["ref"] = config[p]["ref"]
            if "proposal" in config[p]:
                self.prior_info[p]["proposal"] = config[p]["proposal"]

        self.params = list(config.keys())
        self.derived_params = derived_params 
        self.fixed_params = fixed_params
        self.config = config
        
        self.get_prior_sigmas()
        self.get_reference_values()

    def uniform(self, x, x_l, x_r, L=100):
        return jnp.log(0.5 * (jax.lax.erf(L * (x - x_l)) - jax.lax.erf(L * (x - x_r))))

    def normal(self, x, x_mean, sigma_x):
        return -((x - x_mean) ** 2) / (2 * sigma_x**2)

    def log_prior(self, params_values):
        logp = 0
        for p in params_values:
            pi = self.prior_info[p]
            if pi["dist"] == "uniform":
                logp += self.uniform(params_values[p], pi["min"], pi["max"])
            elif pi["dist"] == "norm":
                logp += self.normal(params_values[p], pi["loc"], pi["scale"])
            else:
                raise (
                    NotImplementedError(f"Prior form {pi['dist']} is not implemented.")
                )
        return logp

    def initial_position(self, random_start=True, key=None, normalize=False):
        init = {}
        if random_start & (key is None):
            key = jax.random.key(int(datetime.now().strftime("%Y%m%d%s")))

        if normalize:
            sigmas = self.get_prior_sigmas()
            ref = jnp.array(list(self.get_reference_point().values()))
        else:
            sigmas = jnp.ones(len(self.prior_info))
            ref = jnp.zeros(len(self.prior_info))

        if random_start:
            for i, p in enumerate(self.prior_info):
                pi = self.prior_info[p]
                if ("proposal" in pi) & ("ref" in pi):
                    init[p] = (
                        (jax.random.uniform(key) - 0.5) * pi["proposal"]
                        + pi["ref"]
                        - ref[i]
                    ) / sigmas[i]
                    _, key = jax.random.split(key)
                else:
                    if pi["dist"] == "uniform":
                        init[p] = (
                            jax.random.uniform(key) * (pi["max"] - pi["min"])
                            + pi["min"]
                            - ref[i]
                        ) / sigmas[i]
                        _, key = jax.random.split(key)
                    elif pi["dist"] == "norm":
                        init[p] = (
                            jax.random.normal(key) * pi["scale"] + pi["loc"] - ref[i]
                        ) / sigmas[i]
                        _, key = jax.random.split(key)
                    else:
                        raise (
                            NotImplementedError(
                                f"Prior form {pi['dist']} is not implemented."
                            )
                        )
        else:
            if normalize:
                init = dict(
                    zip(self.prior_info.keys(), jnp.zeros(len(self.prior_info)))
                )
            else:
                init = self.get_reference_point()

        return init

    def get_reference_point(self):
        ref = {}
        for p in self.prior_info:
            pi = self.prior_info[p]
            if "ref" in pi:
                ref[p] = pi["ref"]
            else:
                warnings.warn(
                    f"No reference value provided for {p}, using mean of prior."
                )
                if pi["dist"] == "uniform":
                    ref[p] = (pi["max"] + pi["min"]) / 2
                elif pi["dist"] == "norm":
                    ref[p] = pi["loc"]
                else:
                    raise (
                        NotImplementedError(
                            f"Prior form {pi['dist']} is not implemented"
                        )
                    )
        return ref

    def get_reference_values(self):
        self.reference_values = jnp.array(list(self.get_reference_point().values()))

        return self.reference_values

    def get_proposal_sigmas(self):
        proposal_std = []
        for p in self.prior_info:
            pi = self.prior_info[p]
            if "proposal" in pi:
                proposal_std.append(pi["proposal"])
            elif pi["dist"] == "uniform":
                proposal_std.append((pi["max"] - pi["min"]))
            elif pi["dist"] == "norm":
                proposal_std.append(pi["scale"])

        self.proposal_sigmas = jnp.array(proposal_std)
        return self.proposal_sigmas

    def get_prior_sigmas(self):
        proposal_std = []
        for p in self.prior_info:
            pi = self.prior_info[p]
            if pi["dist"] == "uniform":
                proposal_std.append((pi["max"] - pi["min"]))
            elif pi["dist"] == "norm":
                proposal_std.append(pi["scale"])
        self.prior_sigmas = jnp.array(proposal_std)
        return self.prior_sigmas

    def get_minimizer_bounds(self, n_sigma=3):
        bounds = []
        for p in self.prior_info:
            pi = self.prior_info[p]
            if pi["dist"] == "uniform":
                bounds.append((pi["min"], pi["max"]))
            elif pi["dist"] == "norm":
                bounds.append(
                    (
                        pi["loc"] - n_sigma * pi["scale"],
                        pi["loc"] + n_sigma * pi["scale"],
                    )
                )
            else:
                raise (
                    NotImplementedError(f"Prior form {pi['dist']} is not implemented.")
                )
        return bounds

    def get_pocomc_prior(self):
        prior_dist = []
        for p in self.prior_info:
            pi = self.prior_info[p]
            if pi["dist"] == "uniform":
                prior_dist.append(scipy.stats.uniform(pi["min"], pi["max"] - pi["min"]))
            elif pi["dist"] == "norm":
                prior_dist.append(scipy.stats.norm(pi["loc"], pi["scale"]))
        return prior_dist
