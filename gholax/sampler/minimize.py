import jax.numpy as jnp

from .base import BaseSampler
from .seeding import SeedingConfig


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
        self.seeding_config = SeedingConfig.from_sampler_config(
            c, sampler="Minimize"
        )

    def run(self, model, output_file):
        """Run L-BFGS minimization across parallel chains.

        Args:
            model: Model instance with log_posterior_scaled_params and prior.
            output_file: Base path for output files (results saved as JSON).

        Returns:
            Tuple of (samples array with shape (n_devices, 1, n_params+1),
            parameter names list).
        """
        from ..util.distributed import build_mesh

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

        # mode="per_chain": every chain keeps its own optimum.  The shared
        # "auto" mode would collapse identical starts onto one solve and
        # re-tile poor chains onto the best one, which is exactly what a
        # minimizer must not do.
        seeder = self._seeder()
        optimal_positions = seeder.minimize(
            log_posterior, initial_positions, n_devices, output_file,
            mode="per_chain",
        )
        optimal_values = seeder.map_values

        samples = prior.constrain(jnp.asarray(optimal_positions))
        log_density = -optimal_values

        samples = samples[:, None, :]
        log_density = log_density[:, None]
        samples = jnp.concatenate(
            [samples, log_density[..., None]], axis=-1
        )
        param_names.append("log_posterior")

        return samples, param_names
