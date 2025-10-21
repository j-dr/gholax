import json
import os
from datetime import datetime

import blackjax
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from blackjax.diagnostics import potential_scale_reduction


class NUTS(object):
    def __init__(self, config):
        c = config["sampler"]["NUTS"]

        self.n_steps_warmup = c.get("n_steps_warmup", 500)
        self.target_r_minus_one = c.get("target_r_minus_one", 0.1)
        self.n_steps_incr = c.get("n_steps_incr", 50)
        self.n_steps_min = c.get("n_steps_min", 250)
        self.random_start = c.get("random_start", True)
        self.restart = c.get("restart", False)
        self.diagonal_mass_matrix = c.get("diagonal_mass_matrix", True)
        self.minimize_and_sample = c.get("minimize_and_sample", False)
        self.pathfinder_adaptation = c.get("pathfinder_adaptation", False)

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

        jlp = jax.jit(log_posterior)

        if (os.path.exists(f"{output_file}.nuts_warmup_parameters.json")) & (
            self.restart
        ):
            with open(f"{output_file}.nuts_warmup_parameters.json", "r") as fp:
                warmup_parameters = json.load(fp)

            inverse_mass_matrix = jnp.array(warmup_parameters["inverse_mass_matrix"])
            step_size = jnp.array(warmup_parameters["step_size"])
            if os.path.exists(f"{output_file}.samples_chk.npy"):
                samples = np.load(f"{output_file}.samples_chk.npy")
                log_density = np.load(f"{output_file}.logposterior_chk.npy")
                initial_state = (samples[:, -1, :] - reference[None, :]) / sigmas[
                    None, :
                ]
            else:
                samples = None
                log_density = None
                initial_state = jnp.array(warmup_parameters["initial_state"])

            nuts = blackjax.nuts(
                jlp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
            )
            init_pmap = jax.pmap(nuts.init, in_axes=(0))
            states = init_pmap(initial_state)
            #            print(initial_state, flush=True)
            #            print(initial_state.shape, flush=True)
            kernel = nuts.step

        else:
            if self.minimize_and_sample:
                # minimize negative log posterior
                jnlp = jax.jit(lambda p: -log_posterior(p))
                vgrad = jax.value_and_grad(jnlp)
                solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

                minimize_pmap = jax.pmap(solver.run, in_axes=(0))
                print("Running minimization before sampling", flush=True)
                res = minimize_pmap(initial_positions)
                initial_positions = res.params
                with open(f"{output_file}.minimization_results.json", "w") as fp:
                    json.dump(
                        {
                            "x_opt": initial_positions.tolist(),
                            "value": res.state.value.tolist(),
                        },
                        fp,
                    )

                chi2_ratio = res.state.value / np.min(res.state.value)
                initial_positions_min = jnp.tile(
                    initial_positions[jnp.argmin(res.state.value)], n_devices
                ).reshape(n_devices, -1)
                initial_positions = jnp.where(
                    chi2_ratio[:, None] > 2, initial_positions_min, initial_positions
                )

            if self.pathfinder_adaptation:
                print("Running pathfinder adaptation", flush=True)

                warmup = blackjax.pathfinder_adaptation(
                    blackjax.nuts,
                    jlp,
                    # is_mass_matrix_diagonal=self.diagonal_mass_matrix,
                    # progress_bar=False,
                )
            else:
                print("Running window adaptation", flush=True)

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
            (states, parameters), _ = warmup_pmap(
                warmup_keys, initial_positions, self.n_steps_warmup
            )

            inverse_mass_matrix = jnp.median(parameters["inverse_mass_matrix"], axis=0)
            step_size = jnp.median(parameters["step_size"], axis=0)

            warmup_parameters = {
                "inverse_mass_matrix": inverse_mass_matrix.tolist(),
                "step_size": step_size.tolist(),
                "initial_state": states.position.tolist(),
            }

            with open(f"{output_file}.nuts_warmup_parameters.json", "w") as fp:
                json.dump(warmup_parameters, fp)
            nuts = blackjax.nuts(
                jlp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
            )
            kernel = nuts.step
            samples = None

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

        print("Running inference loop", flush=True)
        rhat = 10000

        if samples is None:
            counter = 0
            n_steps = 0
        else:
            counter = 0
            n_steps = samples.shape[1]

        while (rhat - 1 > self.target_r_minus_one) | (n_steps < self.n_steps_min):
            if counter == 0:
                states = pmap_inference_loop(
                    sample_keys, kernel, states, self.n_steps_incr
                )
            else:
                init_pmap = jax.pmap(nuts.init, in_axes=(0))
                states = init_pmap(states.position[:, -1, :])
                states = pmap_inference_loop(
                    sample_keys, kernel, states, self.n_steps_incr
                )

            if (counter == 0) & (n_steps == 0):
                samples = states.position
                log_density = states.logdensity
            else:
                samples = np.hstack([samples, states.position])
                log_density = np.hstack([log_density, states.logdensity])

            rhat = jnp.mean(potential_scale_reduction(samples))

            print(f"n_samples = {samples.shape[1]}", flush=True)
            print(f"rhat - 1 = {rhat - 1}", flush=True)

            counter += 1
            np.save(
                f"{output_file}.samples_chk.npy",
                samples * sigmas[None, None, :] + reference[None, None, :],
            )
            np.save(f"{output_file}.logposterior_chk.npy", log_density)
            n_steps = samples.shape[1]

        samples = samples * sigmas[None, None, :] + reference[None, None, :]
        samples = jnp.vstack([samples.T, log_density[..., None].T]).T
        param_names.append("log_posterior")

        return samples, param_names
