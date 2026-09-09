import json

import jax
import jax.numpy as jnp
import jaxopt

from .base import BaseSampler


class Minimize(BaseSampler):
    """L-BFGS minimizer for finding the maximum a posteriori (MAP) point.

    Uses jaxopt L-BFGS with parallel starts across available JAX devices
    to minimize the negative log-posterior in normalized parameter space.
    """

    def __init__(self, config):
        """Initialize the minimizer from config.

        Args:
            config: Full config dict containing 'sampler' -> 'Minimize' section.
        """
        c = config["sampler"]["Minimize"]
        self._sampler_cfg = c

        self.random_start = c.get("random_start", True)

    def run(self, model, output_file):
        """Run L-BFGS minimization across parallel chains.

        Args:
            model: Model instance with log_posterior_scaled_params and prior.
            output_file: Base path for output files (results saved as JSON).

        Returns:
            Tuple of (samples array with shape (n_devices, 1, n_params+1),
            parameter names list).
        """
        from ..util.distributed import build_mesh, gather_to_host, is_io_process

        self.mesh = build_mesh(self._sampler_cfg)

        (
            rng_key,
            param_names,
            prior,
            sigmas,
            reference,
            log_posterior,
            jlp,
            n_devices,
            initial_positions,
        ) = self._init_chains(model, jit_logpost=False)

        jnlp = jax.jit(lambda p: -log_posterior(p))
        vgrad = jax.value_and_grad(jnlp)
        solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

        minimize_map = self._map_chains(solver.run)
        print("Running minimization", flush=True)
        res = minimize_map(initial_positions)

        optimal_positions = gather_to_host(res.params)
        optimal_values = gather_to_host(res.state.value)

        if is_io_process():
            with open(f"{output_file}.minimization_results.json", "w") as fp:
                json.dump(
                    {
                        "x_opt": optimal_positions.tolist(),
                        "x_opt_physical": jnp.asarray(
                            prior.constrain(jnp.asarray(optimal_positions))
                        ).tolist(),
                        "value": optimal_values.tolist(),
                    },
                    fp,
                )

        samples = prior.constrain(jnp.asarray(optimal_positions))
        log_density = -optimal_values

        samples = samples[:, None, :]
        log_density = log_density[:, None]
        samples = jnp.concatenate(
            [samples, log_density[..., None]], axis=-1
        )
        param_names.append("log_posterior")

        return samples, param_names
