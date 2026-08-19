import inspect
import json
import os

import blackjax
import blackjax.mcmc.adjusted_mclmc
import blackjax.mcmc.integrators
import blackjax.mcmc.mclmc
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

from .base import BaseSampler

# Older blackjax (e.g. 1.2.5) derives initial adaptation params internally and
# has no `params` kwarg on the unadjusted tuner.
_UNADJ_ADAPT_HAS_PARAMS = (
    "params" in inspect.signature(blackjax.mclmc_find_L_and_step_size).parameters
)
_ADJ_ADAPT_HAS_PARAMS = (
    "params"
    in inspect.signature(blackjax.adjusted_mclmc_find_L_and_step_size).parameters
)


class MCLMC(BaseSampler):
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
        self.warmup_tolerance = c.get("warmup_tolerance", 0.2)
        self.max_warmup_rounds = c.get("max_warmup_rounds", 10)
        self.target_r_minus_one = c.get("target_r_minus_one", 0.1)
        self.n_steps_incr = c.get("n_steps_incr", 50)
        self.n_steps_min = c.get("n_steps_min", 250)
        self.diagonal_preconditioning = c.get("diagonal_preconditioning", True)
        self.random_start = c.get("random_start", True)
        self.restart = c.get("restart", False)
        self.minimize_and_sample = c.get("minimize_and_sample", True)
        self.chains_per_device = int(c.get("chains_per_device", 1))
        if self.chains_per_device < 1:
            raise ValueError("chains_per_device must be >= 1")
        self.step_size_init = c.get("step_size_init", 0.01)
        # "ones", "hessian", or "mclmc" (NaN-guarded multi-chain MCLMC
        # within-chain variance estimate, see BaseSampler._mclmc_mass_matrix).
        self.mass_matrix_init = c.get("mass_matrix_init", "ones")
        # Optional path to a previous run's .mclmc_warmup_parameters.json used
        # to warm-start adaptation.
        self.warmup_init_file = c.get("warmup_init_file", None)
        # Floor for L to prevent phase-3 collapse when mass matrix is well-tuned.
        # L >= L_floor_factor * sqrt(dim) * step_size
        self.L_floor_factor = c.get("L_floor_factor", 1.0)

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
        n_chains = n_devices * self.chains_per_device

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
            rng_key, *init_keys = jax.random.split(rng_key, n_chains + 1)
            init_keys = jnp.array(init_keys)
            init_map = self._map_chains(sampler.init, chain_axes=(0, 0))
            states = init_map(jnp.asarray(initial_state), init_keys)
            kernel = sampler.step

        else:
            if self.minimize_and_sample:
                initial_positions = self._minimize_and_sample(
                    log_posterior, initial_positions, n_chains, output_file
                )

            print(
                f"Running MCLMC adaptation "
                f"({'adjusted' if self.adjusted else 'unadjusted'}, "
                f"{self.n_steps_warmup} steps/round, "
                f"tol={self.warmup_tolerance:.0%})",
                flush=True,
            )

            rng_key, init_key = jax.random.split(rng_key)

            if self.adjusted:
                warmup_state = blackjax.mcmc.adjusted_mclmc.init(
                    position=initial_positions[0],
                    logdensity_fn=jlp,
                )
            else:
                warmup_state = blackjax.mcmc.mclmc.init(
                    position=initial_positions[0],
                    logdensity_fn=jlp,
                    rng_key=init_key,
                )

            dim = initial_positions.shape[1]

            if self.mass_matrix_init == "hessian":
                print("Estimating initial mass matrix from Hessian diagonal...", flush=True)
                x_h = initial_positions[0]
                if not self.minimize_and_sample:
                    x_h = self._best_fit_position(jlp, x_h)
                init_imm = self._hessian_mass_matrix(jlp, x_h)
                print(f"  imm range: [{float(init_imm.min()):.4f}, {float(init_imm.max()):.4f}]", flush=True)
            elif self.mass_matrix_init == "mclmc":
                print("Estimating initial mass matrix from short MCLMC runs...", flush=True)
                rng_key, mm_key = jax.random.split(rng_key)
                init_imm, mm_info = self._mclmc_mass_matrix(
                    jlp, initial_positions, mm_key
                )
                print(
                    f"  rung={mm_info['rung']}, "
                    f"n_survivors={mm_info['n_survivors']}, "
                    f"imm range: [{float(init_imm.min()):.4f}, "
                    f"{float(init_imm.max()):.4f}]",
                    flush=True,
                )
            else:
                init_imm = jnp.ones((dim,))

            if self.warmup_init_file is not None:
                with open(self.warmup_init_file, "r") as fp:
                    winit = json.load(fp)
                warmup_params = MCLMCAdaptationState(
                    L=jnp.asarray(winit["L"]),
                    step_size=jnp.asarray(winit["step_size"]),
                    inverse_mass_matrix=jnp.array(winit["inverse_mass_matrix"]),
                )
                print(
                    f"Warm-starting adaptation from {self.warmup_init_file}",
                    flush=True,
                )
            else:
                warmup_params = MCLMCAdaptationState(
                    L=jnp.sqrt(dim),
                    step_size=self.step_size_init,
                    inverse_mass_matrix=init_imm,
                )

            state, params = self._adapt_with_convergence(
                jlp, warmup_state, rng_key, warmup_params,
                output_file=output_file,
                warm_start=self.warmup_init_file is not None,
            )
            rng_key, _ = jax.random.split(rng_key)

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

            positions = jnp.tile(state.position, (n_chains, 1))
            rng_key, *init_keys = jax.random.split(rng_key, n_chains + 1)
            init_keys = jnp.array(init_keys)
            init_map = self._map_chains(sampler.init, chain_axes=(0, 0))
            states = init_map(positions, init_keys)
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
            log_density = None

        # Inference loop
        keys = jax.random.split(rng_key, 1 + n_chains)
        rng_key = keys[0]
        sample_keys = keys[1:]

        pmap_inference_loop = self._make_pmap_inference_loop()

        # Built once: rebuilding the mapped init every batch retriggers
        # tracing/compilation.
        reinit_map = self._map_chains(sampler.init, chain_axes=(0, 0))

        def reinit_fn(states, rng_key):
            rng_key, *reinit_keys = jax.random.split(rng_key, n_chains + 1)
            reinit_keys = jnp.array(reinit_keys)
            states = reinit_map(states.position[:, -1, :], reinit_keys)
            return states, rng_key

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
            n_chains,
            output_file,
            reinit_fn,
        )

        return self._finalize_samples(
            samples, log_density, sigmas, reference, param_names
        )

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

    def _adapt_with_convergence(self, jlp, initial_state, rng_key, initial_params,
                                output_file=None, warm_start=False):
        """Run adaptation rounds until L, step_size, and inverse_mass_matrix converge.

        Calls the underlying adaptation function repeatedly, checking relative
        change between successive rounds. Stops when all three quantities change
        by less than warmup_tolerance, or after max_warmup_rounds rounds.

        After every round the current parameters are checkpointed to
        {output_file}.mclmc_warmup_intermediate.json; with restart=True an
        existing checkpoint resumes adaptation from that round. warm_start=True
        marks initial_params as trusted so convergence may trigger on round 1.

        Args:
            jlp: JIT-compiled log posterior function.
            initial_state: Initial MCLMC state.
            rng_key: JAX random key.
            initial_params: Initial MCLMCAdaptationState.
            output_file: Base path for the intermediate checkpoint (optional).
            warm_start: Whether initial_params come from a previous run.

        Returns:
            Tuple of (final state, final MCLMCAdaptationState).
        """
        from ..util.distributed import is_io_process

        state = initial_state
        params = initial_params
        dim = initial_params.inverse_mass_matrix.shape[0]
        start_round = 1
        max_rel_change = float("nan")

        params_injectable = (
            _ADJ_ADAPT_HAS_PARAMS if self.adjusted else _UNADJ_ADAPT_HAS_PARAMS
        )
        if not params_injectable:
            print(
                "Warning: installed blackjax does not accept initial params for "
                "MCLMC adaptation; initial L/step_size/inverse_mass_matrix guess "
                "(mass_matrix_init, warmup_init_file) is ignored.",
                flush=True,
            )

        ckpt_file = (
            f"{output_file}.mclmc_warmup_intermediate.json" if output_file else None
        )
        if ckpt_file and self.restart and os.path.exists(ckpt_file):
            with open(ckpt_file, "r") as fp:
                ck = json.load(fp)
            params = MCLMCAdaptationState(
                L=jnp.asarray(ck["L"]),
                step_size=jnp.asarray(ck["step_size"]),
                inverse_mass_matrix=jnp.array(ck["inverse_mass_matrix"]),
            )
            position = jnp.array(ck["position"])
            rng_key, init_key = jax.random.split(rng_key)
            if self.adjusted:
                state = blackjax.mcmc.adjusted_mclmc.init(
                    position=position, logdensity_fn=jlp
                )
            else:
                state = blackjax.mcmc.mclmc.init(
                    position=position, logdensity_fn=jlp, rng_key=init_key
                )
            start_round = ck["round"] + 1
            warm_start = True
            print(
                f"Resuming MCLMC adaptation from checkpointed round {ck['round']}",
                flush=True,
            )

        for round_num in range(start_round, self.max_warmup_rounds + 1):
            rng_key, tune_key = jax.random.split(rng_key)
            prev_params = params

            if self.adjusted:
                state, params = self._adapt_adjusted(jlp, state, tune_key, params)
            else:
                state, params = self._adapt_unadjusted(jlp, state, tune_key, params)

            # Clamp L to prevent phase-3 collapse when ESS is high (blackjax
            # phase 3 computes L = 0.4 * step_size * mean(num_steps_3 / ess),
            # which collapses when the mass matrix is already well-tuned).
            L_min = float(self.L_floor_factor) * float(jnp.sqrt(dim)) * float(params.step_size)
            if float(params.L) < L_min:
                print(
                    f"Warmup round {round_num}: L={float(params.L):.6f} below floor "
                    f"{L_min:.6f}; clamping.",
                    flush=True,
                )
                params = params._replace(L=jnp.asarray(L_min))

            rel_L = abs(float(params.L) - float(prev_params.L)) / (abs(float(prev_params.L)) + 1e-10)
            rel_ss = abs(float(params.step_size) - float(prev_params.step_size)) / (abs(float(prev_params.step_size)) + 1e-10)
            rel_imm = float(
                jnp.max(
                    jnp.abs(params.inverse_mass_matrix - prev_params.inverse_mass_matrix)
                    / (jnp.abs(prev_params.inverse_mass_matrix) + 1e-10)
                )
            )
            max_rel_change = max(rel_L, rel_ss, rel_imm)

            print(
                f"Warmup round {round_num}: "
                f"L={float(params.L):.4f}, "
                f"step_size={float(params.step_size):.6f}, "
                f"max_rel_change={max_rel_change:.4f}",
                flush=True,
            )

            if ckpt_file and is_io_process():
                with open(ckpt_file, "w") as fp:
                    json.dump(
                        {
                            "L": float(params.L),
                            "step_size": float(params.step_size),
                            "inverse_mass_matrix": np.asarray(
                                params.inverse_mass_matrix
                            ).tolist(),
                            "position": np.asarray(state.position).tolist(),
                            "round": round_num,
                        },
                        fp,
                    )

            # When params cannot be injected, round-1 output is unrelated to
            # the (ignored) initial/checkpointed params, so require a second
            # executed round before declaring convergence.
            may_converge = (
                round_num > start_round
                if not params_injectable
                else (round_num > 1 or warm_start)
            )
            if may_converge and max_rel_change < self.warmup_tolerance:
                print(
                    f"Warmup converged after {round_num} rounds "
                    f"(max_rel_change={max_rel_change:.4f} < tol={self.warmup_tolerance:.0%})",
                    flush=True,
                )
                break
        else:
            print(
                f"Warning: Warmup did not converge within {self.max_warmup_rounds} rounds "
                f"(max_rel_change={max_rel_change:.4f}, tol={self.warmup_tolerance:.0%})",
                flush=True,
            )

        return state, params

    def _adapt_unadjusted(self, jlp, initial_state, rng_key, initial_params,
                          diagonal_preconditioning=None):
        """Run unadjusted MCLMC adaptation."""
        if diagonal_preconditioning is None:
            diagonal_preconditioning = self.diagonal_preconditioning

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
            diagonal_preconditioning=diagonal_preconditioning,
            **({"params": initial_params} if _UNADJ_ADAPT_HAS_PARAMS else {}),
        )

        return state, params

    def _adapt_adjusted(self, jlp, initial_state, rng_key, initial_params,
                        diagonal_preconditioning=None):
        """Run adjusted MCLMC adaptation."""
        if diagonal_preconditioning is None:
            diagonal_preconditioning = self.diagonal_preconditioning

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
            diagonal_preconditioning=diagonal_preconditioning,
            **({"params": initial_params} if _ADJ_ADAPT_HAS_PARAMS else {}),
        )

        return state, params
