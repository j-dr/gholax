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
    """No-U-Turn Sampler using blackjax.

    Wraps blackjax's NUTS sampler with window adaptation warmup, convergence
    checking via R-hat, parallel chains via jax.pmap, and checkpoint restart.
    """

    WARMUP_ALGORITHMS = ("window", "adaptive_window", "meads")

    def __init__(self, config):
        """Initialize NUTS sampler from config.

        Args:
            config: Full config dict containing 'sampler' -> 'NUTS' section.
        """
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
        self.target_acceptance_rate = c.get("target_acceptance_rate", 0.65)
        self.step_size_init = c.get("step_size_init", 0.05)
        self.parallel_warmup = c.get("parallel_warmup", False)
        self.mass_matrix_init = c.get("mass_matrix_init", "ones")  # "ones" or "hessian"

        self.warmup_algorithm = c.get("warmup_algorithm", "window")
        if self.warmup_algorithm not in self.WARMUP_ALGORITHMS:
            raise ValueError(
                f"warmup_algorithm must be one of {self.WARMUP_ALGORITHMS}, "
                f"got '{self.warmup_algorithm}'"
            )

        # Adaptive window parameters
        self.adaptive_warmup_stage_steps = c.get("adaptive_warmup_stage_steps", 100)
        self.adaptive_warmup_max_steps = c.get("adaptive_warmup_max_steps", 1000)
        self.adaptive_warmup_min_steps = c.get("adaptive_warmup_min_steps", 200)
        self.adaptive_warmup_rtol_mass = c.get("adaptive_warmup_rtol_mass", 0.05)
        self.adaptive_warmup_rtol_step = c.get("adaptive_warmup_rtol_step", 0.05)

        # MEADS parameters
        self.meads_warmup_steps = c.get("meads_warmup_steps", 150)
        self.meads_step_size_tuning_steps = c.get("meads_step_size_tuning_steps", 100)

    def _hessian_mass_matrix(self, jlp, position):
        """Estimate diagonal inverse mass matrix from the Hessian of the log posterior.

        Uses forward-over-reverse AD (JVP over grad) to compute only the
        diagonal of the Hessian, avoiding materializing the full dim×dim matrix.
        Elements are clamped to [1e-6, 1e6] to guard against degenerate
        curvature far from the MAP.

        Args:
            jlp: JIT-compiled log posterior function (scalar output).
            position: 1D JAX array of parameter values (normalized space).

        Returns:
            1D JAX array of shape (dim,) representing the diagonal
            inverse mass matrix.
        """
        jnlp = lambda p: -jlp(p)
        grad_fn = jax.grad(jnlp)
        diag_H = jax.vmap(
            lambda ei: jax.jvp(grad_fn, (position,), (ei,))[1] @ ei
        )(jnp.eye(len(position)))
        return 1.0 / jnp.clip(diag_H, 1e-6, 1e6)

    def _adaptive_window_warmup(self, jlp, rng_key, initial_position, initial_inverse_mass_matrix=None):
        """Run window adaptation in stages, stopping when mass matrix and step size converge."""
        prev_mass = None
        prev_step = None
        position = initial_position
        total_steps = 0

        # Use provided initial mass matrix only for the first stage; subsequent
        # stages warm-start from the previous stage's adapted mass matrix.
        current_imm = initial_inverse_mass_matrix

        while total_steps < self.adaptive_warmup_max_steps:
            rng_key, sub_key = jax.random.split(rng_key)

            warmup_kwargs = dict(
                is_mass_matrix_diagonal=self.diagonal_mass_matrix,
                progress_bar=False,
                initial_step_size=self.step_size_init,
                target_acceptance_rate=self.target_acceptance_rate,
            )
            if current_imm is not None:
                warmup_kwargs["initial_inverse_mass_matrix"] = current_imm

            warmup = blackjax.window_adaptation(blackjax.nuts, jlp, **warmup_kwargs)
            (state, parameters), _ = warmup.run(
                sub_key, position, self.adaptive_warmup_stage_steps
            )
            current_imm = None  # Only use initial guess for first stage

            mass = parameters["inverse_mass_matrix"]
            step = parameters["step_size"]
            total_steps += self.adaptive_warmup_stage_steps

            if prev_mass is not None and total_steps >= self.adaptive_warmup_min_steps:
                mass_change = float(
                    jnp.max(jnp.abs(mass - prev_mass) / (jnp.abs(prev_mass) + 1e-10))
                )
                step_change = float(
                    jnp.abs(step - prev_step) / (jnp.abs(prev_step) + 1e-10)
                )

                print(
                    f"Adaptive warmup step {total_steps}: "
                    f"max_rel_mass_change={mass_change:.4f}, "
                    f"rel_step_change={step_change:.4f}",
                    flush=True,
                )

                if (
                    mass_change < self.adaptive_warmup_rtol_mass
                    and step_change < self.adaptive_warmup_rtol_step
                ):
                    print(
                        f"Warmup converged after {total_steps} steps", flush=True
                    )
                    return state, parameters

            prev_mass = mass
            prev_step = step
            position = state.position

        print(
            f"Warmup reached max {self.adaptive_warmup_max_steps} steps "
            f"without convergence",
            flush=True,
        )
        return state, parameters

    def _meads_warmup(self, jlp, rng_key, initial_positions):
        """Use MEADS cross-chain adaptation to estimate mass matrix, then
        tune NUTS step size via dual averaging."""
        from blackjax.adaptation.step_size import dual_averaging_adaptation

        n_chains = initial_positions.shape[0]

        print(
            f"Running MEADS warmup ({self.meads_warmup_steps} steps, "
            f"{n_chains} chains)",
            flush=True,
        )

        meads = blackjax.meads_adaptation(jlp, num_chains=n_chains)
        rng_key, meads_key = jax.random.split(rng_key)
        (meads_states, meads_params), meads_info = meads.run(
            meads_key, initial_positions, self.meads_warmup_steps
        )

        # Log convergence of the mass matrix estimate
        position_sigmas = meads_info.adaptation_state.position_sigma
        for i in range(1, self.meads_warmup_steps):
            prev = position_sigmas[i - 1]
            curr = position_sigmas[i]
            rel_change = float(
                jnp.max(jnp.abs(curr - prev) / (jnp.abs(prev) + 1e-10))
            )
            print(
                f"  MEADS step {i}: max rel change in position_sigma = "
                f"{rel_change:.6f}",
                flush=True,
            )

        # Use MEADS position_sigma^2 as diagonal inverse mass matrix
        position_sigma = meads_params["momentum_inverse_scale"]
        inverse_mass_matrix = position_sigma**2

        print(
            f"MEADS complete. Tuning NUTS step size "
            f"({self.meads_step_size_tuning_steps} steps)",
            flush=True,
        )

        # Tune NUTS step size via dual averaging on a single chain
        nuts = blackjax.nuts(
            jlp,
            inverse_mass_matrix=inverse_mass_matrix,
            step_size=float(meads_params["step_size"]),
        )

        da_init, da_update, da_final = dual_averaging_adaptation(
            target=self.target_acceptance_rate
        )
        da_state = da_init(float(meads_params["step_size"]))

        # Use the first chain's final position
        state = nuts.init(meads_states.position[0])

        for i in range(self.meads_step_size_tuning_steps):
            rng_key, step_key = jax.random.split(rng_key)
            nuts_kernel = blackjax.nuts(
                jlp,
                inverse_mass_matrix=inverse_mass_matrix,
                step_size=jnp.exp(da_state.log_step_size),
            )
            state, info = nuts_kernel.step(step_key, state)
            da_state = da_update(da_state, info.acceptance_rate)

        step_size = da_final(da_state)
        print(f"Tuned step size: {float(step_size):.6f}", flush=True)

        parameters = {
            "inverse_mass_matrix": inverse_mass_matrix,
            "step_size": step_size,
        }
        return state, parameters

    def run(self, model, output_file):
        """Run the NUTS sampler until convergence.

        Performs warmup adaptation, then iteratively runs inference until
        R-hat converges below target_r_minus_one.

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
                    (chi2_ratio[:, None] - 1) > 0,
                    initial_positions_min,
                    initial_positions,
                )

            if self.mass_matrix_init == "hessian":
                print("Estimating initial mass matrix from Hessian diagonal...", flush=True)
                init_imm = self._hessian_mass_matrix(jlp, initial_positions[0])
                print(f"  imm range: [{float(init_imm.min()):.4f}, {float(init_imm.max()):.4f}]", flush=True)
            else:
                init_imm = None

            if self.warmup_algorithm == "adaptive_window":
                print("Running adaptive window warmup", flush=True)
                keys = jax.random.split(rng_key, 2)
                rng_key = keys[0]
                state, parameters = self._adaptive_window_warmup(
                    jlp, keys[1], initial_positions[0],
                    initial_inverse_mass_matrix=init_imm,
                )
                inverse_mass_matrix = parameters["inverse_mass_matrix"]
                step_size = parameters["step_size"]
                states = jnp.tile(state.position, (n_devices, 1))
                nuts = blackjax.nuts(
                    jlp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
                )
                init_pmap = jax.pmap(nuts.init, in_axes=(0))
                states = init_pmap(states)

            elif self.warmup_algorithm == "meads":
                keys = jax.random.split(rng_key, 2)
                rng_key = keys[0]
                state, parameters = self._meads_warmup(
                    jlp, keys[1], initial_positions
                )
                inverse_mass_matrix = parameters["inverse_mass_matrix"]
                step_size = parameters["step_size"]
                states = jnp.tile(state.position, (n_devices, 1))
                nuts = blackjax.nuts(
                    jlp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
                )
                init_pmap = jax.pmap(nuts.init, in_axes=(0))
                states = init_pmap(states)

            elif self.pathfinder_adaptation:
                print("Running pathfinder adaptation", flush=True)

                warmup = blackjax.pathfinder_adaptation(
                    blackjax.nuts,
                    jlp,
                )
                if self.parallel_warmup:
                    warmup_pmap = jax.pmap(
                        warmup.run, in_axes=(0, 0, None), static_broadcasted_argnums=2
                    )
                    keys = jax.random.split(rng_key, 1 + n_devices)
                    rng_key = keys[0]
                    warmup_keys = keys[1:]
                    (states, parameters), _ = warmup_pmap(
                        warmup_keys, initial_positions, self.n_steps_warmup
                    )
                    inverse_mass_matrix = jnp.median(
                        parameters["inverse_mass_matrix"], axis=0
                    )
                    step_size = jnp.median(parameters["step_size"], axis=0)

                    with open(f"{output_file}.nuts_inverse_mass_matrix.json", "w") as fp:
                        json.dump(parameters["inverse_mass_matrix"].tolist(), fp)

                    with open(f"{output_file}.nuts_step_size.json", "w") as fp:
                        json.dump(parameters["step_size"].tolist(), fp)
                else:
                    keys = jax.random.split(rng_key, 2)
                    (state, parameters), _ = warmup.run(
                        keys[0],
                        initial_positions[0],
                        self.n_steps_warmup,
                    )
                    inverse_mass_matrix = parameters["inverse_mass_matrix"]
                    step_size = parameters["step_size"]
                    states = jnp.tile(state.position, (n_devices, 1))
                    nuts = blackjax.nuts(
                        jlp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
                    )
                    init_pmap = jax.pmap(nuts.init, in_axes=(0))
                    states = init_pmap(states)

            else:
                print("Running window adaptation", flush=True)

                warmup_kwargs = dict(
                    is_mass_matrix_diagonal=self.diagonal_mass_matrix,
                    progress_bar=False,
                    initial_step_size=self.step_size_init,
                    target_acceptance_rate=self.target_acceptance_rate,
                )
                if init_imm is not None:
                    warmup_kwargs["initial_inverse_mass_matrix"] = init_imm

                warmup = blackjax.window_adaptation(blackjax.nuts, jlp, **warmup_kwargs)
                if self.parallel_warmup:
                    warmup_pmap = jax.pmap(
                        warmup.run, in_axes=(0, 0, None), static_broadcasted_argnums=2
                    )
                    keys = jax.random.split(rng_key, 1 + n_devices)
                    rng_key = keys[0]
                    warmup_keys = keys[1:]
                    (states, parameters), _ = warmup_pmap(
                        warmup_keys, initial_positions, self.n_steps_warmup
                    )
                    inverse_mass_matrix = jnp.median(
                        parameters["inverse_mass_matrix"], axis=0
                    )
                    step_size = jnp.median(parameters["step_size"], axis=0)

                    with open(f"{output_file}.nuts_inverse_mass_matrix.json", "w") as fp:
                        json.dump(parameters["inverse_mass_matrix"].tolist(), fp)

                    with open(f"{output_file}.nuts_step_size.json", "w") as fp:
                        json.dump(parameters["step_size"].tolist(), fp)
                else:
                    keys = jax.random.split(rng_key, 2)
                    (state, parameters), _ = warmup.run(
                        keys[0],
                        initial_positions[0],
                        self.n_steps_warmup,
                    )
                    inverse_mass_matrix = parameters["inverse_mass_matrix"]
                    step_size = parameters["step_size"]
                    states = jnp.tile(state.position, (n_devices, 1))
                    nuts = blackjax.nuts(
                        jlp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
                    )
                    init_pmap = jax.pmap(nuts.init, in_axes=(0))
                    states = init_pmap(states)

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

            keys = jax.random.split(rng_key, 1 + n_devices)
            rng_key = keys[0]
            sample_keys = keys[1:]
            

        samples = samples * sigmas[None, None, :] + reference[None, None, :]
        samples = jnp.vstack([samples.T, log_density[..., None].T]).T
        param_names.append("log_posterior")

        return samples, param_names
