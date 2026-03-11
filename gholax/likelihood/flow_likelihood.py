from __future__ import annotations

from typing import Any, Dict, List

import jax.numpy as jnp
import jax
import yaml

import equinox as eqx

from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal
from flowjax.bijections import RationalQuadraticSpline
from datetime import datetime

from gholax.sampler.priors import Prior
from .likelihood import Likelihood

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

    if config.get('type', 'maf') == 'maf':
        ndim = len(list(config['params'].keys()))
        base = Normal(jnp.zeros((ndim,)))  # unit Normal, scale defaults to 1 if omitted in your version
        flow = masked_autoregressive_flow(key,
                                      base_dist=base,
                                      transformer=RationalQuadraticSpline(knots=config.get("knots",16), interval=config.get("interval", 4)),
                                     )
    else:
        raise NotImplementedError("Currently not supported.")

    return flow

class FlowLikelihood(Likelihood):
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
    
    def __init__(self, config):
        """Initialize the Nx2PT angular power spectrum likelihood from config."""
        c = config["likelihood"]["FlowLikelihood"]
        base_path = c.get("base_path")
        flow_file = f'{base_path}_flow.eqx'
        config_file = f'{base_path}_config.yaml'

        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.mu = jnp.asarray(config["mean"])
        self.sig = jnp.asarray(config["std"])
        priors = config['params']
        self.params = list(priors.keys())

        # Prior expects {param: {'prior': {dist, min/max or loc/scale}}}
#        prior_config = {p: {'prior': priors[p]} for p in priors}
        self.prior = Prior(priors)

        # Rebuild architecture, then load parameters into it
        key = jax.random.key(int(datetime.now().strftime("%Y%m%d%s")))

        flow_template = build_flow_from_config(config, key=key)
        self.flow = eqx.tree_deserialise_leaves(flow_file, flow_template)
        
        self.likelihood_pipeline = []


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