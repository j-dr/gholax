import json
from datetime import datetime

import jax
import jax.numpy as jnp
import jaxopt


class Minimize(object):
    def __init__(self, config):
        c = config["sampler"]["Minimize"]

        self.random_start = c.get("random_start", True)

    def run(self, model, output_file):
        rng_key = jax.random.key(int(datetime.now().strftime("%Y%m%d%s")))
        param_names = model.prior.params
        prior = model.prior

        sigmas = prior.get_prior_sigmas()
        reference = prior.get_reference_values()
        log_posterior = model.log_posterior_scaled_params

        n_devices = jax.local_device_count()
        keys = jax.random.split(rng_key, n_devices + 1)
        rng_key = keys[0]
        initial_keys = keys[1:]
        initial_positions = jnp.array(
            [
                list(
                    prior.initial_position(
                        random_start=self.random_start, key=k, normalize=True
                    ).values()
                )
                for k in initial_keys
            ]
        )

        jnlp = jax.jit(lambda p: -log_posterior(p))
        vgrad = jax.value_and_grad(jnlp)
        solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

        minimize_pmap = jax.pmap(solver.run, in_axes=(0))
        print("Running minimization", flush=True)
        res = minimize_pmap(initial_positions)

        optimal_positions = res.params
        optimal_values = res.state.value

        with open(f"{output_file}.minimization_results.json", "w") as fp:
            json.dump(
                {
                    "x_opt": optimal_positions.tolist(),
                    "value": optimal_values.tolist(),
                },
                fp,
            )

        samples = optimal_positions * sigmas[None, :] + reference[None, :]
        log_density = -optimal_values

        samples = samples[:, None, :]
        log_density = log_density[:, None]
        samples = jnp.concatenate(
            [samples, log_density[..., None]], axis=-1
        )
        param_names.append("log_posterior")

        return samples, param_names
