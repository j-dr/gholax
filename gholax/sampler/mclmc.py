import json
import os
from datetime import datetime

import blackjax
import blackjax.mcmc.adjusted_mclmc
import blackjax.mcmc.integrators
import blackjax.mcmc.mclmc
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.diagnostics import potential_scale_reduction


class MCLMC(object):
    """Microcanonical Langevin Monte Carlo sampler using blackjax.

    Wraps blackjax's MCLMC (unadjusted) and adjusted MCLMC samplers with
    automatic adaptation, convergence checking via R-hat, parallel chains
    via jax.pmap, and checkpoint restart.
    """

    def __init__(self, config):
        """Initialize MCLMC sampler from config.

        Args:
            config: Full config dict containing 'sampler' -> 'MCLMC' section.
        """
        c = config["sampler"]["MCLMC"]

        self.adjusted = c.get("adjusted", False)
        self.n_steps_warmup = c.get("n_steps_warmup", 5000)
        self.target_r_minus_one = c.get("target_r_minus_one", 0.1)
        self.n_steps_incr = c.get("n_steps_incr", 50)
        self.n_steps_min = c.get("n_steps_min", 250)
        self.diagonal_preconditioning = c.get("diagonal_preconditioning", True)
        self.random_start = c.get("random_start", True)
        self.restart = c.get("restart", False)
        self.minimize_and_sample = c.get("minimize_and_sample", True)
        self.step_size_init = c.get("step_size_init", 0.01)

        # Adaptation tuning fractions
        self.frac_tune1 = c.get("frac_tune1", 0.1)
        self.frac_tune2 = c.get("frac_tune2", 0.1)
        self.frac_tune3 = c.get("frac_tune3", 0.1)

        # Adjusted-only parameters
        self.target_acceptance_rate = c.get("target_acceptance_rate", 0.65)

    def run(self, model, output_file):
        """Run the MCLMC sampler until convergence.

        Performs adaptation to find L and step_size, then iteratively runs
        inference until R-hat converges below target_r_minus_one.

        Args:
            model: Model instance with log_posterior_scaled_params and prior.
            output_file: Base path for output files.

        Returns:
            Tuple of (samples array, parameter names list).
        """
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

        checkpoint_file = f"{output_file}.mclmc_warmup_parameters.json"

        if os.path.exists(checkpoint_file) and self.restart:
            with open(checkpoint_file, "r") as fp:
                warmup_parameters = json.load(fp)

            L = jnp.array(warmup_parameters["L"])
            step_size = jnp.array(warmup_parameters["step_size"])
            inverse_mass_matrix = jnp.array(
                warmup_parameters["inverse_mass_matrix"]
            )

            if os.path.exists(f"{output_file}.samples_chk.npy"):
                samples = np.load(f"{output_file}.samples_chk.npy")
                log_density = np.load(f"{output_file}.logposterior_chk.npy")
                initial_state = (
                    samples[:, -1, :] - reference[None, :]
                ) / sigmas[None, :]
            else:
                samples = None
                log_density = None
                initial_state = jnp.array(warmup_parameters["initial_state"])

            sampler = self._build_sampler(
                jlp, L, step_size, inverse_mass_matrix
            )
            rng_key, *init_keys = jax.random.split(rng_key, n_devices + 1)
            init_keys = jnp.array(init_keys)
            init_pmap = jax.pmap(sampler.init, in_axes=(0, 0))
            states = init_pmap(initial_state, init_keys)
            kernel = sampler.step

        else:
            if self.minimize_and_sample:
                jnlp = jax.jit(lambda p: -log_posterior(p))
                vgrad = jax.value_and_grad(jnlp)
                solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

                minimize_pmap = jax.pmap(solver.run, in_axes=(0))
                print("Running minimization before sampling", flush=True)
                res = minimize_pmap(initial_positions)
                initial_positions = res.params
                with open(
                    f"{output_file}.minimization_results.json", "w"
                ) as fp:
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
                    (chi2_ratio[:, None] - 1) > 0,
                    initial_positions_min,
                    initial_positions,
                )

            print(
                f"Running MCLMC adaptation "
                f"({'adjusted' if self.adjusted else 'unadjusted'}, "
                f"{self.n_steps_warmup} steps)",
                flush=True,
            )

            rng_key, init_key, tune_key = jax.random.split(rng_key, 3)

            if self.adjusted:
                initial_state = blackjax.mcmc.adjusted_mclmc.init(
                    position=initial_positions[0],
                    logdensity_fn=jlp,
                )
            else:
                initial_state = blackjax.mcmc.mclmc.init(
                    position=initial_positions[0],
                    logdensity_fn=jlp,
                    rng_key=init_key,
                )

            dim = initial_positions.shape[1]
            initial_params = MCLMCAdaptationState(
                L=jnp.sqrt(dim),
                step_size=self.step_size_init,
                inverse_mass_matrix=jnp.ones((dim,)),
            )

            if self.adjusted:
                state, params = self._adapt_adjusted(
                    jlp, initial_state, tune_key, initial_params
                )
            else:
                state, params = self._adapt_unadjusted(
                    jlp, initial_state, tune_key, initial_params
                )

            L = params.L
            step_size = params.step_size
            inverse_mass_matrix = params.inverse_mass_matrix

            print(
                f"Adaptation complete: L={float(L):.4f}, "
                f"step_size={float(step_size):.6f}",
                flush=True,
            )

            sampler = self._build_sampler(
                jlp, L, step_size, inverse_mass_matrix
            )

            positions = jnp.tile(state.position, (n_devices, 1))
            rng_key, *init_keys = jax.random.split(rng_key, n_devices + 1)
            init_keys = jnp.array(init_keys)
            init_pmap = jax.pmap(sampler.init, in_axes=(0, 0))
            states = init_pmap(positions, init_keys)
            kernel = sampler.step

            warmup_parameters = {
                "L": float(L),
                "step_size": float(step_size),
                "inverse_mass_matrix": inverse_mass_matrix.tolist(),
                "initial_state": states.position.tolist(),
                "adjusted": self.adjusted,
            }
            with open(checkpoint_file, "w") as fp:
                json.dump(warmup_parameters, fp)

            samples = None

        # Inference loop
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

        while (rhat - 1 > self.target_r_minus_one) | (
            n_steps < self.n_steps_min
        ):
            if counter == 0:
                states = pmap_inference_loop(
                    sample_keys, kernel, states, self.n_steps_incr
                )
            else:
                rng_key, *reinit_keys = jax.random.split(
                    rng_key, n_devices + 1
                )
                reinit_keys = jnp.array(reinit_keys)
                init_pmap = jax.pmap(sampler.init, in_axes=(0, 0))
                states = init_pmap(states.position[:, -1, :], reinit_keys)
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

            keys = jax.random.split(rng_key, 1 + n_devices)
            rng_key = keys[0]
            sample_keys = keys[1:]

        samples = samples * sigmas[None, None, :] + reference[None, None, :]
        samples = jnp.vstack([samples.T, log_density[..., None].T]).T
        param_names.append("log_posterior")

        return samples, param_names

    def _build_sampler(self, jlp, L, step_size, inverse_mass_matrix):
        """Build the production sampler via as_top_level_api."""
        if self.adjusted:
            num_integration_steps = jnp.clip(
                jnp.round(L / step_size).astype(int), 1
            )
            return blackjax.adjusted_mclmc(
                logdensity_fn=jlp,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                num_integration_steps=num_integration_steps,
            )
        else:
            return blackjax.mclmc(
                logdensity_fn=jlp,
                L=L,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
            )

    def _adapt_unadjusted(self, jlp, initial_state, rng_key, initial_params):
        """Run unadjusted MCLMC adaptation."""
        kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=jlp,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix,
        )

        state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=self.n_steps_warmup,
            state=initial_state,
            rng_key=rng_key,
            frac_tune1=self.frac_tune1,
            frac_tune2=self.frac_tune2,
            frac_tune3=self.frac_tune3,
            diagonal_preconditioning=self.diagonal_preconditioning,
            params=initial_params,
        )

        return state, params

    def _adapt_adjusted(self, jlp, initial_state, rng_key, initial_params):
        """Run adjusted MCLMC adaptation."""

        def kernel(
            rng_key,
            state,
            avg_num_integration_steps,
            step_size,
            inverse_mass_matrix,
        ):
            num_steps = jnp.clip(
                jnp.round(avg_num_integration_steps).astype(int), 1
            )
            k = blackjax.mcmc.adjusted_mclmc.build_kernel(
                logdensity_fn=jlp,
                inverse_mass_matrix=inverse_mass_matrix,
            )
            return k(rng_key, state, step_size, num_steps)

        state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=self.n_steps_warmup,
            state=initial_state,
            rng_key=rng_key,
            target=self.target_acceptance_rate,
            frac_tune1=self.frac_tune1,
            frac_tune2=self.frac_tune2,
            frac_tune3=self.frac_tune3,
            diagonal_preconditioning=self.diagonal_preconditioning,
            params=initial_params,
        )

        return state, params
