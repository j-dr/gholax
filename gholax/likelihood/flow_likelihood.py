from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import jax
import yaml

import equinox as eqx

from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal
from flowjax.bijections import RationalQuadraticSpline

from gholax.sampler.priors import Prior

def build_flow_from_config(config: Dict[str, Any], key: jax.Array):
    """
    Reconstruct the FlowJAX flow architecture from a config dict.
    This uses masked_autoregressive_flow + RQ spline as in your example.

    Expected config keys (with defaults):
      - knots: int (default 8)
      - interval: float (default 4.0)
      - Any other kwargs supported by masked_autoregressive_flow in your install
        (e.g. n_transforms/hidden_dims/etc.) can be passed via config["maf_kwargs"].
    """

    if config['type'] == 'maf':
        ndim = len(list(config['priors'].keys()))
        base = Normal(jnp.zeros((ndim,)))  # unit Normal, scale defaults to 1 if omitted in your version
        flow = masked_autoregressive_flow(key,
                                      base_dist=base,
                                      transformer=RationalQuadraticSpline(knots=config["knots"], interval=config["interval"]),
                                     )
    else:
        raise NotImplementedError("Currently not supported.")

    return flow

@dataclass
class FlowLikelihood:
    """Implements a likelihood using a normalizing flow to model the posterior distribution.
    The flow is trained on samples from the posterior (e.g. from GetDist) and then can be
    used to evaluate the likelihood at new parameter values. The likelihood is obtained by
    subtracting the log prior from the log posterior given by the flow,
    and adding the appropriate Jacobian terms for any standardization.
    """
    flow: Any
    mu: jax.Array
    sig: jax.Array
    priors: Dict
    params: List[str]
    prior: Any
    
    @classmethod
    def load(self,
             prefix: str,
             *,
             flow_file: Optional[str] = None,
             meta_file: Optional[str] = None,
             config_file: Optional[str] = None,
             eps: float = 1e-6,
             seed: int = 0) -> "FlowLikelihood":
        """
        Loads:
          - {prefix}_flow.eqx
          - {prefix}_config.yaml (architecture)

        Currently does
        """
        flow_file = flow_file or (prefix + "_flow.eqx")
        config_file = config_file or (prefix + "_config.yaml")
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        mu = jnp.asarray(config["mean"])
        sig = jnp.asarray(config["std"])
        priors = config['priors']
        params = list(priors.keys())

        # Prior expects {param: {'prior': {dist, min/max or loc/scale}}}
        prior_config = {p: {'prior': priors[p]} for p in priors}
        prior_obj = Prior(prior_config)

        # Rebuild architecture, then load parameters into it
        key = jax.random.key(seed)
        flow_template = build_flow_from_config(config, key=key)
        flow = eqx.tree_deserialise_leaves(flow_file, flow_template)

        return self(flow=flow, mu=mu, sig=sig, priors=priors, params=params, prior=prior_obj)

    def compute(self, params: Dict) -> jax.Array:
        """
        theta: (..., D) in the same column order as self.param_order
               (or (D,) single point).
        returns: (...) log prior
        """
        theta = jnp.asarray([params[p] for p in self.params])

        u_std = (theta - self.mu) / self.sig

        logp_u_std = self.flow.log_prob(u_std)

        # Jacobian for standardization u_std = (u - mu)/sig
        logJ_std = -jnp.sum(jnp.log(self.sig))

        logp_theta = logp_u_std + logJ_std

        log_prior = self.prior.log_prior(params)

        return logp_theta - log_prior