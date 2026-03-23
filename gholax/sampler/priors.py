import warnings
from datetime import datetime

import jax
import jax.numpy as jnp
import scipy.stats


class Prior(object):
    """Manage parameter priors, reference values, and proposal distributions.

    Supports uniform and normal prior distributions, parameter normalization,
    and initial position generation for samplers.
    """

    def __init__(self, config, derived_params=[], fixed_params=[], joint_priors={},
                 linear_constraint_priors={}):
        """Initialize priors from parameter configuration.

        Args:
            config: Dict mapping parameter names to their config (must include 'prior' key).
            derived_params: List of derived parameter names.
            fixed_params: List of fixed parameter names.
            joint_priors: Dict of multivariate Gaussian prior groups. Each key is a group
                name, each value has 'params', 'mean', and 'cov' entries.
        """
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

        self.joint_prior_groups = {}
        self.joint_prior_params = set()

        for name, group in joint_priors.items():
            params = group["params"]
            mean = jnp.array(group["mean"], dtype=float)
            cov = jnp.array(group["cov"], dtype=float)
            n = len(params)
            cinv = jnp.linalg.inv(cov)
            sign, log_det = jnp.linalg.slogdet(cov)
            log_norm = -0.5 * (n * jnp.log(2 * jnp.pi) + log_det)
            self.joint_prior_groups[name] = {
                "params": params,
                "mean": mean,
                "cinv": cinv,
                "log_norm": log_norm,
            }
            for i, p in enumerate(params):
                self.joint_prior_params.add(p)
                # Overwrite stub prior_info with marginal normal so downstream
                # methods (get_prior_sigmas, initial_position, etc.) work unchanged.
                existing = self.prior_info.get(p, {})
                marginal_std = float(jnp.sqrt(cov[i, i]))
                self.prior_info[p] = {
                    "dist": "norm",
                    "loc": float(mean[i]),
                    "scale": marginal_std,
                }
                if "ref" in existing:
                    self.prior_info[p]["ref"] = existing["ref"]
                if "proposal" in existing:
                    self.prior_info[p]["proposal"] = existing["proposal"]

        self.linear_constraint_groups = {}
        for name, group in linear_constraint_priors.items():
            params = group["params"]
            coeffs = jnp.array(group["coeffs"], dtype=float)
            bound = float(group["bound"])
            L = float(group.get("L", 10))
            self.linear_constraint_groups[name] = {
                "params": params,
                "coeffs": coeffs,
                "bound": bound,
                "L": L,
            }

        self.get_prior_sigmas()
        self.get_reference_values()

    def uniform(self, x, x_l, x_r, L=10):
        """Compute log-probability of a smooth uniform prior using error functions."""
        delta = (x_r - x_l)
        return jnp.log(0.5 * (jax.lax.erf(L * (x - x_l)/delta) - jax.lax.erf(L * (x - x_r)/delta)))

    def normal(self, x, x_mean, sigma_x):
        """Compute log-probability of a normal prior."""
        return -((x - x_mean) ** 2) / (2 * sigma_x**2)

    def log_prior(self, params_values):
        """Compute the total log-prior for all parameters."""
        logp = 0
        for p in params_values:
            if p in self.joint_prior_params:
                continue  # handled by joint prior below
            pi = self.prior_info[p]
            if pi["dist"] == "uniform":
                logp += self.uniform(params_values[p], pi["min"], pi["max"])
            elif pi["dist"] == "norm":
                logp += self.normal(params_values[p], pi["loc"], pi["scale"])
            else:
                raise (
                    NotImplementedError(f"Prior form {pi['dist']} is not implemented.")
                )

        for group in self.joint_prior_groups.values():
            residual = jnp.stack(
                [params_values[p] - group["mean"][i]
                 for i, p in enumerate(group["params"])]
            )
            logp += -0.5 * residual @ group["cinv"] @ residual + group["log_norm"]

        for group in self.linear_constraint_groups.values():
            linear_combo = jnp.sum(jnp.stack(
                [group["coeffs"][i] * params_values[p]
                 for i, p in enumerate(group["params"])]
            ))
            scale = jnp.sum(jnp.abs(group["coeffs"]) * self.prior_sigmas[
                jnp.array([self.params.index(p) for p in group["params"]])
            ])
            logp += jnp.log(0.5 * (1 - jax.lax.erf(
                group["L"] * (linear_combo - group["bound"]) / scale
            )))

        return logp

    def initial_position(self, random_start=True, key=None, normalize=False):
        """Generate an initial parameter position for a sampler chain.

        Args:
            random_start: If True, draw from the prior/proposal. If False, use reference.
            key: JAX random key (generated from timestamp if None).
            normalize: If True, return position in normalized parameter space.

        Returns:
            Dict mapping parameter names to initial values.
        """
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
        """Get reference parameter values, falling back to prior means."""
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
        """Get reference values as a JAX array and cache the result."""
        self.reference_values = jnp.array(list(self.get_reference_point().values()))

        return self.reference_values

    def get_proposal_sigmas(self):
        """Get proposal widths from proposal/prior config."""
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
        """Get prior widths (range for uniform, scale for normal)."""
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
        """Get parameter bounds for optimization (min/max for uniform, n_sigma for normal)."""
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
        """Get scipy prior distributions for use with pocoMC sampler."""
        prior_dist = []
        for p in self.prior_info:
            pi = self.prior_info[p]
            if pi["dist"] == "uniform":
                prior_dist.append(scipy.stats.uniform(pi["min"], pi["max"] - pi["min"]))
            elif pi["dist"] == "norm":
                prior_dist.append(scipy.stats.norm(pi["loc"], pi["scale"]))
        return prior_dist
