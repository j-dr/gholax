from datetime import datetime

import blackjax
import jax
import jax.numpy as jnp


class NUTS(object):
    def __init__(self, config):
        c = config["sampler"]["NUTS"]

        self.n_steps_warmup = c.get("n_steps_warmup", 1000)
        self.n_steps = c.get("n_steps", 10000)
        self.random_start = c.get("random_start", True)
        self.diagonal_mass_matrix = c.get("diagonal_mass_matrix", False)

    def run(self, model, output_file):
        rng_key = jax.random.key(int(datetime.now().strftime("%Y%m%d%s")))
        param_names = model.prior.params
        prior = model.prior

        log_posterior = model.log_posterior

        n_devices = jax.local_device_count()
        keys = jax.random.split(rng_key, n_devices + 1)
        rng_key = keys[0]
        initial_keys = keys[1:]
        initial_positions = jnp.array(
            [
                list(
                    prior.initial_position(
                        random_start=self.random_start, key=k
                    ).values()
                )
                for k in initial_keys
            ]
        )

        print("Running window adaptation", flush=True)
        jlp = jax.jit(log_posterior)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            jlp,
            is_mass_matrix_diagonal=self.diagonal_mass_matrix,
            progress_bar=False,
        )
        warmup_pmap = jax.pmap(
            warmup.run, in_axes=(0, 0, None), static_broadcasted_argnums=2
        )
        keys = jax.random.split(rng_key, 1 + n_devices)
        rng_key = keys[0]
        warmup_keys = keys[1:]
        (state, parameters), _ = warmup_pmap(
            warmup_keys, initial_positions, self.n_steps_warmup
        )

        keys = jax.random.split(rng_key, 1 + n_devices)
        rng_key = keys[0]
        sample_keys = keys[1:]

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        pmap_inference_loop = jax.pmap(
            inference_loop,
            in_axes=(0, None, 0, None),
            static_broadcasted_argnums=(1, 3),
        )
        inverse_mass_matrix = jnp.mean(parameters["inverse_mass_matrix"], axis=0)
        step_size = jnp.mean(parameters["step_size"], axis=0)

        kernel = blackjax.nuts(
            jlp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
        ).step

        print("Running inference loop", flush=True)
        states = pmap_inference_loop(sample_keys, kernel, state, self.n_steps)
        #        states = inference_loop(sample_key, kernel, state, self.n_steps)

        samples = states.position
        samples = jnp.vstack([samples.T, states.logdensity[..., None].T]).T
        param_names.append("log_posterior")

        return samples, param_names
