import json
import os

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseSampler


class NUTS(BaseSampler):
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
        self._sampler_cfg = c

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
        from ..util.distributed import build_mesh, gather_to_host, is_io_process

        self.mesh = build_mesh(self._sampler_cfg)
        if self.mesh is not None:
            if self.warmup_algorithm == "meads":
                raise ValueError(
                    "warmup_algorithm 'meads' is not supported in mesh mode "
                    "(n_chains/model_shards or multi-process runs); use "
                    "'adaptive_window' or 'window'."
                )
            if self.parallel_warmup:
                raise ValueError(
                    "parallel_warmup is not supported in mesh mode; warmup "
                    "runs on a single chain with the model-sharded posterior."
                )

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
        ) = self._init_chains(model)

        # Kernels driven inside the chain shard_map must close over the raw
        # (ctx-active) posterior — jlp is already shard_map-wrapped in mesh
        # mode and cannot nest. Host-driven warmup keeps using jlp.
        kernel_lp = log_posterior if self.mesh is not None else jlp

        if (os.path.exists(f"{output_file}.nuts_warmup_parameters.json")) & (
            self.restart
        ):
            with open(f"{output_file}.nuts_warmup_parameters.json", "r") as fp:
                warmup_parameters = json.load(fp)

            inverse_mass_matrix = jnp.array(warmup_parameters["inverse_mass_matrix"])
            step_size = jnp.array(warmup_parameters["step_size"])
            if os.path.exists(f"{output_file}.samples_chk.npy"):
                # The checkpoint stores physical-space samples, but the
                # convergence loop accumulates normalized ones and rescales on
                # every write. Convert back on load, otherwise each restart
                # re-applies sigma/reference to the whole resumed prefix.
                samples = (
                    np.load(f"{output_file}.samples_chk.npy")
                    - np.asarray(reference)[None, None, :]
                ) / np.asarray(sigmas)[None, None, :]
                log_density = np.load(f"{output_file}.logposterior_chk.npy")
                initial_state = samples[:, -1, :]
            else:
                samples = None
                log_density = None
                initial_state = jnp.array(warmup_parameters["initial_state"])

            nuts = blackjax.nuts(
                kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
            )
            states = self._map_chains(nuts.init)(jnp.asarray(initial_state))
            kernel = nuts.step

        else:
            if self.minimize_and_sample:
                initial_positions = self._minimize_and_sample(
                    log_posterior, initial_positions, n_devices, output_file
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
                    kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
                )
                states = self._map_chains(nuts.init)(states)

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
                    kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
                )
                states = self._map_chains(nuts.init)(states)

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
                        kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
                    )
                    states = self._map_chains(nuts.init)(states)

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
                        kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
                    )
                    states = self._map_chains(nuts.init)(states)

            warmup_parameters = {
                "inverse_mass_matrix": inverse_mass_matrix.tolist(),
                "step_size": step_size.tolist(),
                "initial_state": np.asarray(
                    gather_to_host(states.position)
                ).tolist(),
            }

            if is_io_process():
                with open(f"{output_file}.nuts_warmup_parameters.json", "w") as fp:
                    json.dump(warmup_parameters, fp)

            nuts = blackjax.nuts(
                kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size
            )
            kernel = nuts.step
            samples = None
            log_density = None

        keys = jax.random.split(rng_key, 1 + n_devices)
        rng_key = keys[0]
        sample_keys = keys[1:]

        if self.mesh is None:
            pmap_inference_loop = self._make_pmap_inference_loop()
        else:
            pmap_inference_loop = self._make_mesh_inference_loop(
                kernel, self.n_steps_incr
            )

        def reinit_fn(states, rng_key):
            init_map = self._map_chains(nuts.init)
            return init_map(states.position[:, -1, :]), rng_key

        samples, log_density = self._run_convergence_loop(
            pmap_inference_loop,
            kernel,
            states,
            rng_key,
            sample_keys,
            samples,
            log_density,
            sigmas,
            reference,
            n_devices,
            output_file,
            reinit_fn,
        )

        return self._finalize_samples(
            samples, log_density, sigmas, reference, param_names
        )
