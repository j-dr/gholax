import inspect
import json
import os

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseSampler

# blackjax < 1.3 forwards unknown window_adaptation kwargs to the kernel,
# so only pass initial_inverse_mass_matrix when explicitly supported.
_HAS_INITIAL_IMM = (
    "initial_inverse_mass_matrix"
    in inspect.signature(blackjax.window_adaptation).parameters
)


class NUTS(BaseSampler):
    """No-U-Turn Sampler using blackjax.

    Wraps blackjax's NUTS sampler with window adaptation warmup, convergence
    checking via R-hat, parallel chains via jax.pmap, and checkpoint restart.
    """

    WARMUP_ALGORITHMS = (
        "window", "adaptive_window", "meads", "chees", "pooled_window"
    )

    def __init__(self, config):
        """Initialize NUTS sampler from config.

        Args:
            config: Full config dict containing 'sampler' -> 'NUTS' section.
        """
        c = config["sampler"]["NUTS"]
        self._sampler_cfg = c

        self.n_steps_warmup = c.get("n_steps_warmup", 500)
        self.target_r_minus_one = c.get("target_r_minus_one", 0.1)
        self.n_steps_incr = c.get("n_steps_incr", 10)
        self.n_steps_min = c.get("n_steps_min", 250)
        self.random_start = c.get("random_start", True)
        self.restart = c.get("restart", False)
        self.diagonal_mass_matrix = c.get("diagonal_mass_matrix", True)
        self.minimize_and_sample = c.get("minimize_and_sample", False)
        self.minimize_n_starts = c.get("minimize_n_starts", 4)
        # Tree-depth cap for the sampling kernel (warmup has its own,
        # pooled_window_max_doublings). Under vmap lockstep one deep-tree
        # chain stalls the whole batch; 8 bounds a step at 256 leapfrogs.
        self.max_num_doublings = c.get("max_num_doublings", 10)
        # Auto-cap sampling depth from the converged warmup depth
        # distribution unless the config pins max_num_doublings.
        self.max_num_doublings_auto = "max_num_doublings" not in c
        self.depth_cap_quantile = c.get("depth_cap_quantile", 0.9)
        self.minimize_start_scale = c.get("minimize_start_scale", 0.5)
        self.pathfinder_adaptation = c.get("pathfinder_adaptation", False)
        self.target_acceptance_rate = c.get("target_acceptance_rate", 0.65)
        self.step_size_init = c.get("step_size_init", 0.05)
        # Divergences remain diagnostics by default for backwards
        # compatibility.  Production runs can opt into a fail-closed rate
        # threshold; with fail_on_divergence and no threshold, any divergent
        # transition raises.
        self.max_divergence_rate = c.get("max_divergence_rate", None)
        if self.max_divergence_rate is not None:
            self.max_divergence_rate = float(self.max_divergence_rate)
            if not 0.0 <= self.max_divergence_rate <= 1.0:
                raise ValueError("max_divergence_rate must be between 0 and 1")
        self.fail_on_divergence = bool(c.get("fail_on_divergence", False))
        self.divergence_check_min_steps = int(
            c.get("divergence_check_min_steps", 0)
        )
        if self.divergence_check_min_steps < 0:
            raise ValueError("divergence_check_min_steps must be >= 0")
        # Search for an initial step size (doubling/halving heuristic) unless
        # the config pins one explicitly.
        self.step_size_search = "step_size_init" not in c
        self.parallel_warmup = c.get("parallel_warmup", False)
        self.chains_per_device = int(c.get("chains_per_device", 1))
        if self.chains_per_device < 1:
            raise ValueError("chains_per_device must be >= 1")
        self.mass_matrix_init = c.get("mass_matrix_init", "ones")  # "ones", "hessian", or "mclmc"
        # Optional path to a previous run's .nuts_warmup_parameters.json used
        # to warm-start adaptation.
        self.warmup_init_file = c.get("warmup_init_file", None)

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

        # ChEES parameters
        self.chees_warmup_steps = c.get("chees_warmup_steps", 200)
        self.chees_learning_rate = c.get("chees_learning_rate", 0.25)

        # Pooled window parameters (convergence rtols shared with
        # adaptive_window: adaptive_warmup_rtol_mass / adaptive_warmup_rtol_step)
        self.pooled_window_steps = c.get("pooled_window_steps", 25)
        self.pooled_window_max_steps = c.get("pooled_window_max_steps", 200)
        self.pooled_window_max_doublings = c.get("pooled_window_max_doublings", 10)
        # A short correlated tail does not give the between/within ratio a
        # universal tau/T null distribution.  Gate relative chain agreement
        # directly with the maximum ordinary PSR (R-hat) over coordinates,
        # after enough tail draws have accumulated.  This is a warmup safety
        # check, not evidence that every posterior mode has been explored.
        if "pooled_window_mixing_rhat" in c:
            self.pooled_window_mixing_rhat = float(
                c["pooled_window_mixing_rhat"]
            )
        elif "pooled_window_mixing_gate" in c:
            # Compatibility for exploratory configurations written before
            # pooled_window_mixing_rhat.  The former quantity was compared
            # to a dimensionless excess, so retain that monotonic behavior
            # without claiming that it has a calibrated tau/T interpretation.
            self.pooled_window_mixing_rhat = 1.0 + float(
                c["pooled_window_mixing_gate"]
            )
        else:
            self.pooled_window_mixing_rhat = 1.2
        if self.pooled_window_mixing_rhat < 1.0:
            raise ValueError("pooled_window_mixing_rhat must be >= 1")
        self.pooled_window_min_tail_steps = int(
            c.get("pooled_window_min_tail_steps", 20)
        )
        self.pooled_window_consecutive_windows = int(
            c.get("pooled_window_consecutive_windows", 2)
        )
        if self.pooled_window_min_tail_steps < 2:
            raise ValueError("pooled_window_min_tail_steps must be >= 2")
        if self.pooled_window_consecutive_windows < 1:
            raise ValueError("pooled_window_consecutive_windows must be >= 1")
        self.pooled_window_max_window = c.get("pooled_window_max_window", 40)
        # A converged pooled window is followed by a short fixed-metric
        # calibration run.  This is deliberately separate from the growing
        # metric windows: the step size returned by a window was tuned while
        # using the *previous* metric, so it is not calibrated for the metric
        # that will be used for sampling.
        self.pooled_window_terminal_steps = int(
            c.get("pooled_window_terminal_steps", self.pooled_window_steps)
        )
        if self.pooled_window_terminal_steps < 1:
            raise ValueError("pooled_window_terminal_steps must be >= 1")
        # Pooled convergence is a safety gate.  Keep the default fail-closed;
        # an explicit opt-in is available for exploratory/debugging runs.
        self.pooled_window_allow_unconverged = c.get(
            "pooled_window_allow_unconverged",
            c.get("allow_unconverged_warmup", False),
        )

    def _adaptive_window_warmup(self, jlp, rng_key, initial_position,
                                initial_inverse_mass_matrix=None,
                                output_file=None, initial_step_size=None):
        """Run window adaptation in stages, stopping when mass matrix and step size converge.

        After every stage the current warmup parameters are checkpointed to
        {output_file}.nuts_warmup_intermediate.json; with restart=True an
        existing checkpoint resumes adaptation from that stage. Passing
        initial_step_size (warm start) seeds the convergence check with the
        provided parameters so adaptation can stop after a single stage.
        """
        from ..util.distributed import is_io_process

        prev_mass = None
        prev_step = None
        position = initial_position
        total_steps = 0
        step_size_init = self.step_size_init

        # Use provided initial mass matrix only for the first stage; subsequent
        # stages warm-start from the previous stage's adapted mass matrix.
        current_imm = initial_inverse_mass_matrix
        if current_imm is not None and not _HAS_INITIAL_IMM:
            print(
                "Installed blackjax window_adaptation does not accept "
                "initial_inverse_mass_matrix; ignoring initial mass matrix.",
                flush=True,
            )
            current_imm = None

        if initial_step_size is not None:
            step_size_init = float(initial_step_size)
            if initial_inverse_mass_matrix is not None:
                prev_mass = jnp.asarray(initial_inverse_mass_matrix)
                prev_step = jnp.asarray(step_size_init)

        ckpt_file = (
            f"{output_file}.nuts_warmup_intermediate.json" if output_file else None
        )
        if ckpt_file and self.restart and os.path.exists(ckpt_file):
            with open(ckpt_file, "r") as fp:
                ck = json.load(fp)
            position = jnp.array(ck["position"])
            prev_mass = jnp.array(ck["inverse_mass_matrix"])
            prev_step = jnp.asarray(ck["step_size"])
            total_steps = ck["total_steps"]
            current_imm = None
            print(
                f"Resuming adaptive warmup from checkpointed step {total_steps}",
                flush=True,
            )

        while total_steps < self.adaptive_warmup_max_steps:
            rng_key, sub_key = jax.random.split(rng_key)

            warmup_kwargs = dict(
                is_mass_matrix_diagonal=self.diagonal_mass_matrix,
                progress_bar=False,
                initial_step_size=step_size_init,
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

            if ckpt_file and is_io_process():
                with open(ckpt_file, "w") as fp:
                    json.dump(
                        {
                            "inverse_mass_matrix": np.asarray(mass).tolist(),
                            "step_size": float(step),
                            "position": np.asarray(state.position).tolist(),
                            "total_steps": total_steps,
                        },
                        fp,
                    )

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

    def _jitter_positions(self, rng_key, position, inverse_mass_matrix,
                          n_chains, scale=0.5):
        """Overdisperse chain starts around a single warmup endpoint using the
        adapted mass matrix, so identical starts don't bias R-hat low."""
        noise = jax.random.normal(rng_key, (n_chains, position.shape[0]))
        if jnp.ndim(inverse_mass_matrix) == 2:
            chol = jnp.linalg.cholesky(inverse_mass_matrix)
            return position[None, :] + scale * noise @ chol.T
        return (
            position[None, :]
            + jnp.sqrt(inverse_mass_matrix)[None, :] * scale * noise
        )

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

    def _chees_kernel(self, lp, step_size, inverse_mass_matrix,
                      trajectory_length_adjusted, halton_max_bits):
        """Jittered dynamic-HMC kernel with ChEES-tuned parameters, rebuilt
        from serializable scalars (mirrors blackjax chees_adaptation.run)."""
        from blackjax.mcmc.dynamic_hmc import halton_sequence

        def integration_steps_fn(arg):
            return jnp.asarray(
                jnp.ceil(
                    halton_sequence(arg, halton_max_bits)
                    * trajectory_length_adjusted
                ),
                dtype=int,
            )

        return blackjax.dynamic_hmc(
            lp,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            next_random_arg_fn=lambda i: i + 1,
            integration_steps_fn=integration_steps_fn,
        )

    def _chees_warmup(self, jlp, rng_key, initial_positions,
                      initial_step_size=None):
        """Pooled cross-chain ChEES warmup: every step all chains share the
        step-size / trajectory-length statistics, so far fewer sequential
        steps are needed than in per-chain window adaptation. blackjax 1.2.5
        adapts no mass matrix here (identity is hard-coded), which relies on
        sampling in the sigma-normalized parameter space. Adaptation runs
        vmapped on a single device (it is short); the tuned scalars are
        shared by all devices afterwards.
        """
        import optax
        from blackjax.adaptation.base import get_filter_adapt_info_fn

        n_chains = initial_positions.shape[0]
        step0 = (
            float(initial_step_size)
            if initial_step_size is not None
            else self.step_size_init
        )
        print(
            f"Running ChEES warmup ({self.chees_warmup_steps} steps, "
            f"{n_chains} chains)",
            flush=True,
        )
        # The Halton jitter bit budget must also cover post-warmup sampling.
        max_sampling_steps = 2**15
        warmup = blackjax.chees_adaptation(
            jlp,
            num_chains=n_chains,
            target_acceptance_rate=self.target_acceptance_rate,
            adaptation_info_fn=get_filter_adapt_info_fn(
                adapt_state_keys={
                    "log_step_size_moving_average",
                    "log_trajectory_length_moving_average",
                }
            ),
        )
        (last_states, parameters), info = warmup.run(
            rng_key,
            initial_positions,
            step0,
            optax.adam(self.chees_learning_rate),
            self.chees_warmup_steps,
            max_sampling_steps=max_sampling_steps,
        )
        # Recover the scalar hidden in parameters["integration_steps_fn"] so
        # the tuned kernel can be serialized and rebuilt on restart.
        ad = info.adaptation_state
        trajectory_length_adjusted = float(
            jnp.exp(
                ad.log_trajectory_length_moving_average[-1]
                - ad.log_step_size_moving_average[-1]
            )
        )
        halton_max_bits = int(
            np.ceil(np.log2(self.chees_warmup_steps + max_sampling_steps))
        )
        print(
            f"ChEES tuned step_size={float(parameters['step_size']):.5f}, "
            f"trajectory_length_adjusted={trajectory_length_adjusted:.2f}",
            flush=True,
        )
        return last_states, {
            "inverse_mass_matrix": parameters["inverse_mass_matrix"],
            "step_size": parameters["step_size"],
            "trajectory_length_adjusted": trajectory_length_adjusted,
            "halton_max_bits": halton_max_bits,
        }

    def _pooled_window_warmup(self, jlp, rng_key, initial_positions,
                              initial_inverse_mass_matrix=None,
                              initial_step_size=None, output_file=None):
        """Torsten-style cross-chain windowed warmup that keeps NUTS end to end.

        All chains step together through windows that double in length each
        boundary (Stan-style, from pooled_window_steps up to
        pooled_window_max_window) with a fixed diagonal
        inverse mass matrix, while the step size is dual-averaged every step
        from the cross-chain mean acceptance (Stan-style within-window
        feedback). At each window boundary the mass matrix is re-estimated
        from the pooled within-chain position variance, Stan-regularized
        toward the current inverse mass matrix, and
        dual averaging is restarted at its averaged step size (as blackjax
        window_adaptation does at slow-window ends). Pooling yields
        n_chains samples per sequential step, so far fewer sequential steps
        are needed than in single-chain window adaptation. Boundaries are
        checkpointed to {output_file}.nuts_warmup_intermediate.json and
        resumable with restart=True. Stops when mass matrix and step size
        stabilize (adaptive_warmup_rtol_mass/step; warm starts seed the
        check) or at pooled_window_max_steps.

        Returns:
            Tuple of (final per-chain positions, parameters dict).
        """
        from blackjax.adaptation.step_size import (
            dual_averaging_adaptation,
            find_reasonable_step_size,
        )
        from blackjax.mcmc import nuts as nuts_mcmc

        from ..util.distributed import is_io_process

        n_chains, dim = initial_positions.shape
        n_window = self.pooled_window_steps
        kernel = nuts_mcmc.build_kernel()

        da_init, da_update, da_final = dual_averaging_adaptation(
            target=self.target_acceptance_rate
        )

        # Adaptation runs pmap(vmap): K chains vmapped per device, devices
        # synchronized through an axis-name pmean so the per-step
        # dual-averaging update sees the acceptance averaged across ALL
        # chains — every device advances an identical da_state. This uses
        # all GPUs during warmup (previously all chains vmapped on device 0)
        # and spreads the memory footprint.
        # Depth cap bounds per-step cost while the metric is still poor: an
        # untuned mass matrix drives NUTS to max depth, and vmap lockstep
        # makes every chain pay the deepest tree.
        nd = jax.local_device_count()
        assert n_chains % nd == 0
        K = n_chains // nd
        max_doublings = self.pooled_window_max_doublings
        vkernel = jax.vmap(
            lambda k, s, step, imm: kernel(
                k, s, jlp, step, imm, max_num_doublings=max_doublings
            ),
            in_axes=(0, 0, None, None),
        )
        vinit = jax.vmap(lambda p: nuts_mcmc.init(p, jlp))

        def _dev_split(tree):
            return jax.tree.map(
                lambda x: x.reshape((nd, K) + x.shape[1:]), tree
            )

        def _flat_positions(states):
            return np.asarray(states.position).reshape(n_chains, -1)

        # imm / da_state are traced arguments so every window reuses one
        # compilation. Step size feeds back per step (Stan-style), which is
        # what keeps dual averaging stable; updating it only at boundaries
        # gives no within-window feedback and ping-pongs.
        def _run_window(dev_key, states, da_state, imm, n_steps):
            def one_step(carry, k):
                states, da_state = carry
                step = jnp.exp(da_state.log_step_size)
                states, infos = vkernel(
                    jax.random.split(k, K), states, step, imm
                )
                acc = jax.lax.pmean(jnp.mean(infos.acceptance_rate), "d")
                da_state = da_update(da_state, acc)
                return (states, da_state), (
                    states.position,
                    acc,
                    infos.num_integration_steps,
                )

            (states, da_state), (pos, acc, n_leap) = jax.lax.scan(
                one_step, (states, da_state), jax.random.split(dev_key, n_steps)
            )
            return states, da_state, pos, acc, n_leap

        # ``n_steps`` is static so a final partial window performs exactly the
        # number of transitions left in the strict pooled-step budget.  A
        # dynamic lax.scan length would either fail to compile or tempt the
        # caller to run a full window and merely discard the excess samples.
        # JAX caches the ordinary window and (usually) one short remainder
        # specialization.
        prun_window = jax.pmap(
            _run_window,
            axis_name="d",
            in_axes=(0, 0, None, None),
            static_broadcasted_argnums=(4,),
        )

        def run_window(key, states, da_state, imm, n_steps):
            states, da_dev, pos, acc, n_leap = prun_window(
                jax.random.split(key, nd), states, da_state, imm, n_steps
            )
            # da_state is device-invariant (pmean-synchronized); host
            # consumers get flat (n_steps, n_chains, ...) arrays.
            da_state = jax.tree.map(lambda x: x[0], da_dev)
            pos = jnp.moveaxis(pos, 0, 1).reshape(n_steps, n_chains, -1)
            n_leap = jnp.moveaxis(n_leap, 0, 1).reshape(n_steps, n_chains)
            return states, da_state, pos, acc[0], n_leap

        positions = initial_positions
        imm = (
            jnp.asarray(initial_inverse_mass_matrix)
            if initial_inverse_mass_matrix is not None
            else jnp.ones(dim)
        )
        step_size = jnp.asarray(
            initial_step_size
            if initial_step_size is not None
            else self.step_size_init
        )
        prev_mass = None
        prev_step = None
        # ``next_window_chunks`` is the size of the next growing boundary,
        # rather than the size of the boundary just completed.  Keeping this
        # distinction in the checkpoint makes a restart schedule-identical.
        next_window_chunks = 1
        stable_boundaries = 0
        if initial_inverse_mass_matrix is not None and initial_step_size is not None:
            # Warm start: seed the convergence check so a single window can
            # suffice.
            prev_mass = imm
            prev_step = step_size
        total_steps = 0

        ckpt_file = (
            f"{output_file}.nuts_warmup_intermediate.json" if output_file else None
        )
        if ckpt_file and self.restart and os.path.exists(ckpt_file):
            with open(ckpt_file, "r") as fp:
                ck = json.load(fp)
            if "positions" in ck:  # written by this warmup mode
                positions = jnp.array(ck["positions"])
                imm = jnp.array(ck["inverse_mass_matrix"])
                step_size = jnp.asarray(ck["step_size"])
                prev_mass = imm
                prev_step = step_size
                total_steps = ck["total_steps"]
                stable_boundaries = int(ck.get("stable_boundaries", 0))
                if "next_window_chunks" in ck:
                    next_window_chunks = int(ck["next_window_chunks"])
                else:
                    # Checkpoints written before this field was introduced
                    # stored the completed boundary.  Advance it once when
                    # reading those checkpoints.
                    completed_chunks = int(ck.get("window_chunks", 1))
                    next_window_chunks = min(
                        2 * completed_chunks,
                        max(1, self.pooled_window_max_window // n_window),
                    )
                print(
                    f"Resuming pooled window warmup from checkpointed step "
                    f"{total_steps}",
                    flush=True,
                )

        if self.step_size_search and initial_step_size is None and total_steps == 0:
            # Halve/double from a large guess until acceptance crosses target:
            # a few gradient evaluations replace whole windows spent crawling
            # from a mismatched hand-set initial step size.
            rng_key, srch_key = jax.random.split(rng_key)
            step_size = jnp.asarray(
                find_reasonable_step_size(
                    srch_key,
                    lambda eps: lambda k, s: kernel(
                        k, s, jlp, eps, imm,
                        max_num_doublings=max_doublings,
                    ),
                    nuts_mcmc.init(positions[0], jlp),
                    0.5,
                    target_accept=self.target_acceptance_rate,
                )
            )
            print(
                f"Initial step size search: {float(step_size):.3g}", flush=True
            )

        print(
            f"Running pooled window warmup ({n_chains} chains, "
            f"{n_window}-step windows, max {self.pooled_window_max_steps} "
            f"steps)",
            flush=True,
        )
        da_state = da_init(float(step_size))
        states = _dev_split(vinit(positions))

        n_leap = None
        calibration_leap = None
        converged = False
        calibrated = False
        # Stan-style growing windows built from k base-length scans (one
        # compiled executable): short windows early for fast metric feedback,
        # doubling each boundary so the tail statistics and mixing check are
        # judged on enough samples to be useful.
        k_max = max(1, self.pooled_window_max_window // n_window)
        k = min(next_window_chunks, k_max)
        while total_steps < self.pooled_window_max_steps:
            pos_c, acc_c, leap_c = [], [], []
            # A growing boundary may be truncated at the hard budget.  In
            # particular, this prevents max_steps=42 with 5-step windows
            # from silently consuming 45 transitions.
            remaining = self.pooled_window_max_steps - total_steps
            boundary_steps = min(k * n_window, remaining)
            if boundary_steps <= 0:
                break
            n_subwindows = (boundary_steps + n_window - 1) // n_window
            for i in range(n_subwindows):
                sub_steps = min(n_window, boundary_steps - i * n_window)
                rng_key, sub_key = jax.random.split(rng_key)
                states, da_state, p_i, a_i, l_i = run_window(
                    sub_key, states, da_state, imm, sub_steps
                )
                pos_c.append(p_i)
                acc_c.append(a_i)
                leap_c.append(l_i)
                total_steps += sub_steps
            pos = jnp.concatenate(pos_c, axis=0)
            acc = jnp.concatenate(acc_c, axis=0)
            n_leap = jnp.concatenate(leap_c, axis=0)
            w_len = boundary_steps

            # Diagonal imm from the pooled WITHIN-chain variance (mean over
            # chains of each chain's variance across its window samples).
            # Total variance over the flattened pool adds the between-chain
            # spread, which for unmixed chains measures the overdispersed
            # starts, not the posterior. Stan's shrinkage shape is kept but
            # shrinks toward the current imm (seed / previous boundary), so
            # a good seed is sticky against early noise. The first 20% of
            # each window is a transient buffer (chains jumping in from
            # overdispersed starts contaminate the within variance without
            # showing up between chains). n_eff is the PER-CHAIN tail
            # sample count, not multiplied by chains: autocorrelation makes
            # per-chain trajectory length the information bottleneck, and
            # the resulting cross-window EMA smooths the noisy short-tail
            # estimate. It is deflated by the between/within mixing ratio
            # so unmixed chains collapse the weight and the seed stays
            # sticky.
            # pos: (w_len, n_chains, dim)
            tail_start = min(max(w_len // 5, 0), max(w_len - 1, 0))
            tail = pos[tail_start:]
            tail_ddof = 1 if tail.shape[0] > 1 else 0
            within = jnp.mean(jnp.var(tail, axis=0, ddof=tail_ddof), axis=0)
            # Mixing diagnostic: between-chain variance of chain means over
            # within; >> 1 means the chains are unmixed.
            between_ddof = 1 if n_chains > 1 else 0
            between = jnp.var(
                jnp.mean(tail, axis=0), axis=0, ddof=between_ddof
            )
            bw_ratio = float(jnp.max(between / (within + 1e-30)))
            tail_rhat = float("inf")
            mixing_ready = tail.shape[0] >= self.pooled_window_min_tail_steps
            if mixing_ready and n_chains > 1:
                from blackjax.diagnostics import potential_scale_reduction

                tail_rhat = float(
                    jnp.max(
                        potential_scale_reduction(
                            tail, chain_axis=1, sample_axis=0
                        )
                    )
                )
            mixing_ok = (
                mixing_ready
                and tail_rhat <= self.pooled_window_mixing_rhat
            )
            # Keep the boundary update conservative when chains disagree, but
            # do not interpret this shrinkage factor as an ESS estimate.
            disagreement = max(tail_rhat - 1.0, 0.0) if mixing_ready else 1.0
            tail_weight_steps = tail.shape[0] / (1.0 + disagreement)
            w = tail_weight_steps / (tail_weight_steps + 5)
            if imm.ndim == 2:
                # Dense (hessian_dense) metric is frozen through warmup: the
                # boundary estimator is diagonal-only, so windows tune eps
                # and verify mixing against the fixed dense metric.
                mass = imm
            else:
                mass = w * within + (1 - w) * imm
            # Boundary: take the averaged step size and restart dual
            # averaging around it with the new mass matrix (blackjax
            # window_adaptation's slow_final).
            step = da_final(da_state)
            da_state = da_init(float(step))
            mean_acc = float(jnp.mean(acc))

            converged = False
            if prev_mass is not None:
                mass_change = float(
                    jnp.max(jnp.abs(mass - prev_mass) / (jnp.abs(prev_mass) + 1e-10))
                )
                step_change = float(
                    jnp.abs(step - prev_step) / (jnp.abs(prev_step) + 1e-10)
                )
                print(
                    f"Pooled warmup step {total_steps} (window {w_len}): "
                    f"max_rel_mass_change={mass_change:.4f}, "
                    f"rel_step_change={step_change:.4f}, "
                    f"mean_acceptance={mean_acc:.3f}, "
                    f"max_between_within_ratio={bw_ratio:.2f}, "
                    f"max_tail_rhat={tail_rhat:.3f}, "
                    f"tail_steps={tail.shape[0]}"
                    + (
                        " (mixing gate not met)"
                        if not mixing_ok
                        else ""
                    ),
                    flush=True,
                )
                stable = (
                    mass_change < self.adaptive_warmup_rtol_mass
                    and step_change < self.adaptive_warmup_rtol_step
                    and mixing_ok
                )
                stable_boundaries = stable_boundaries + 1 if stable else 0
                converged = (
                    stable_boundaries
                    >= self.pooled_window_consecutive_windows
                )

            prev_mass = mass
            prev_step = step
            imm = mass
            step_size = jnp.asarray(step)
            completed_window_chunks = k
            next_window_chunks = min(2 * k, k_max)
            k = next_window_chunks

            # A convergence boundary is useful only if the hard budget can
            # also accommodate the required fixed-metric calibration phase.
            # Do this before checkpointing so the checkpoint cannot claim a
            # converged run that has no calibration budget left.
            if converged and (
                self.pooled_window_max_steps - total_steps
                < self.pooled_window_terminal_steps
            ):
                converged = False

            if ckpt_file and is_io_process():
                with open(ckpt_file, "w") as fp:
                    json.dump(
                        {
                            "inverse_mass_matrix": np.asarray(mass).tolist(),
                            "step_size": float(step),
                            "positions": _flat_positions(states).tolist(),
                            "total_steps": total_steps,
                            "between_within_ratio": bw_ratio,
                            "tail_rhat": tail_rhat,
                            "stable_boundaries": stable_boundaries,
                            # Keep the old field for readers of existing
                            # checkpoints, but make the restart contract
                            # explicit with the next boundary size.
                            "window_chunks": completed_window_chunks,
                            "next_window_chunks": next_window_chunks,
                            "warmup_converged": bool(converged),
                            "calibrated": False,
                        },
                        fp,
                    )

            if converged:
                print(f"Pooled warmup convergence detected at {total_steps} steps", flush=True)
                if imm.ndim == 2:
                    # Swap the MAP-local dense metric for the pooled sample
                    # covariance of the converged (mixed) tail: the MAP
                    # Hessian's principal axes misalign away from the MAP
                    # (observed 26% divergent sampling vs 10% for diagonal),
                    # while the mixed-chain covariance captures the global
                    # correlation structure. Shrunk toward its diagonal and
                    # eigenvalue-guarded like the Hessian init; eps is then
                    # re-seeded and calibrated against the new metric below.
                    xc = tail - tail.mean(axis=0, keepdims=True)
                    n_cov = xc.shape[0] * xc.shape[1]
                    C = jnp.einsum("tcd,tce->de", xc, xc) / max(n_cov - 1, 1)
                    alpha = dim / (dim + n_cov)
                    C = (1 - alpha) * C + alpha * jnp.diag(jnp.diag(C))
                    lam, V = jnp.linalg.eigh(C)
                    pos = jnp.isfinite(lam) & (lam > 0)
                    med = jnp.nanmedian(jnp.where(pos, lam, jnp.nan))
                    med = jnp.where(
                        jnp.isfinite(med) & (med > 0), med, jnp.asarray(1.0)
                    )
                    lam = jnp.where(pos, lam, med)
                    lam = jnp.clip(
                        jnp.maximum(lam, 1e-4 * med), 1e-6, 1e6
                    )
                    imm = (V * lam) @ V.T
                    ev = jnp.linalg.eigvalsh(imm)
                    print(
                        f"Dense metric updated from pooled tail covariance "
                        f"({n_cov} samples): eigenvalue range "
                        f"[{float(ev.min()):.3e}, {float(ev.max()):.3e}]",
                        flush=True,
                    )
                    rng_key, srch_key = jax.random.split(rng_key)
                    step = float(
                        find_reasonable_step_size(
                            srch_key,
                            lambda eps: lambda k, s: kernel(
                                k, s, jlp, eps, imm,
                                max_num_doublings=max_doublings,
                            ),
                            nuts_mcmc.init(
                                jnp.asarray(_flat_positions(states))[0], jlp
                            ),
                            float(step),
                            target_accept=self.target_acceptance_rate,
                        )
                    )
                    print(
                        f"Step size re-seeded for updated metric: {step:.3g}",
                        flush=True,
                    )
                # Calibrate epsilon with the just-adapted metric held fixed.
                # The ordinary boundary's da_final was tuned while ``imm``
                # still denoted the preceding metric.  Use only transitions
                # within the remaining hard budget and retain their depth
                # observations for the optional automatic depth cap.
                terminal_steps = min(
                    max(0, int(self.pooled_window_terminal_steps)),
                    self.pooled_window_max_steps - total_steps,
                )
                if terminal_steps:
                    rng_key, terminal_key = jax.random.split(rng_key)
                    terminal_da = da_init(float(step))
                    states, terminal_da, _, _, terminal_leap = run_window(
                        terminal_key,
                        states,
                        terminal_da,
                        imm,
                        terminal_steps,
                    )
                    step = da_final(terminal_da)
                    step_size = jnp.asarray(step)
                    calibration_leap = terminal_leap
                    total_steps += terminal_steps
                    calibrated = True
                    print(
                        f"Pooled fixed-metric calibration: {terminal_steps} steps, "
                        f"step_size={float(step):.5g}",
                        flush=True,
                    )
                    # Keep the final calibrated endpoint available to a
                    # restart.  It is also useful if a run is interrupted
                    # immediately after calibration.
                    if ckpt_file and is_io_process():
                        with open(ckpt_file, "w") as fp:
                            json.dump(
                                {
                                    "inverse_mass_matrix": np.asarray(imm).tolist(),
                                    "step_size": float(step),
                                    "positions": _flat_positions(states).tolist(),
                                    "total_steps": total_steps,
                                    "between_within_ratio": bw_ratio,
                                    "tail_rhat": tail_rhat,
                                    "stable_boundaries": stable_boundaries,
                                    "window_chunks": completed_window_chunks,
                                    "next_window_chunks": k,
                                    "warmup_converged": True,
                                    "calibrated": True,
                                },
                                fp,
                            )
                print(
                    f"Pooled warmup converged after {total_steps} steps",
                    flush=True,
                )
                break
        else:
            print(
                f"Pooled warmup reached max {self.pooled_window_max_steps} "
                f"steps without convergence",
                flush=True,
            )

        if not converged:
            message = (
                "Pooled warmup failed to converge within "
                f"{self.pooled_window_max_steps} steps; refusing to start "
                "sampling. Set pooled_window_allow_unconverged=true only "
                "for an explicitly exploratory run."
            )
            if not self.pooled_window_allow_unconverged:
                raise RuntimeError(message)
            print("WARNING: " + message, flush=True)

        parameters = {
            "inverse_mass_matrix": imm,
            "step_size": step_size,
            "warmup_converged": bool(converged),
            "warmup_calibrated": bool(calibrated),
        }
        if (
            self.max_num_doublings_auto
            and calibration_leap is not None
            and converged
            and calibrated
        ):
            # Sampling depth cap from the last (converged) window: under vmap
            # lockstep the deepest chain sets the per-step cost, so cap just
            # above the bulk of the depth distribution instead of blackjax's
            # default 10.
            depth = jnp.ceil(
                jnp.log2(calibration_leap.astype(jnp.float32) + 1.0)
            )
            qs = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            dq = [int(jnp.quantile(depth, q)) for q in qs]
            print(
                "Warmup tree depth percentiles: "
                + ", ".join(f"q{q:g}={d}" for q, d in zip(qs, dq)),
                flush=True,
            )
            q_depth = int(jnp.quantile(depth, self.depth_cap_quantile))
            self.max_num_doublings = min(q_depth + 1, self.max_num_doublings)
            parameters["max_num_doublings"] = self.max_num_doublings
            print(
                f"Sampling depth cap: {self.max_num_doublings} "
                f"(q{self.depth_cap_quantile:g} warmup depth {q_depth})",
                flush=True,
            )
        return jnp.asarray(_flat_positions(states)), parameters

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
            if self.warmup_algorithm in ("meads", "chees", "pooled_window"):
                raise ValueError(
                    f"warmup_algorithm '{self.warmup_algorithm}' is not "
                    "supported in mesh mode (n_chains/model_shards or "
                    "multi-process runs); use 'adaptive_window' or 'window'."
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
        n_chains = n_devices * self.chains_per_device

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
            if self.max_num_doublings_auto and "max_num_doublings" in warmup_parameters:
                self.max_num_doublings = int(warmup_parameters["max_num_doublings"])
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

            if self.warmup_algorithm == "chees":
                algo = self._chees_kernel(
                    kernel_lp, step_size, inverse_mass_matrix,
                    warmup_parameters["trajectory_length_adjusted"],
                    warmup_parameters["halton_max_bits"],
                )
                rga = jnp.full(
                    n_chains,
                    warmup_parameters["random_generator_arg"],
                    dtype=jnp.int32,
                )
                states = self._map_chains(algo.init, chain_axes=(0, 0))(
                    jnp.asarray(initial_state), rga
                )
            else:
                algo = blackjax.nuts(
                    kernel_lp, inverse_mass_matrix=inverse_mass_matrix,
                    step_size=step_size,
                    max_num_doublings=self.max_num_doublings,
                )
                states = self._map_chains(algo.init)(jnp.asarray(initial_state))
            kernel = algo.step

        else:
            # A pooled-window warmup checkpoint supersedes minimization and
            # the mass-matrix seed: the warmup resume loads positions, metric
            # and step size from it, so redoing the MAP search and Hessian
            # would be pure waste (they only feed the warmup's cold start).
            warm_ckpt = False
            if (
                self.restart
                and self.warmup_algorithm == "pooled_window"
                and os.path.exists(
                    f"{output_file}.nuts_warmup_intermediate.json"
                )
            ):
                with open(
                    f"{output_file}.nuts_warmup_intermediate.json"
                ) as fp:
                    # "positions" marks a pooled-window checkpoint (other
                    # warmup modes write different intermediates).
                    warm_ckpt = "positions" in json.load(fp)
            if warm_ckpt:
                print(
                    "Warmup checkpoint found; skipping minimization and "
                    "mass matrix initialization",
                    flush=True,
                )
            if self.minimize_and_sample and not warm_ckpt:
                initial_positions = self._minimize_and_sample(
                    log_posterior, initial_positions, n_chains, output_file
                )

            if warm_ckpt:
                init_imm = None
            elif self.mass_matrix_init == "hessian":
                print("Estimating initial mass matrix from Hessian diagonal...", flush=True)
                x_h = initial_positions[0]
                if not self.minimize_and_sample:
                    x_h = self._best_fit_position(jlp, x_h)
                init_imm = self._hessian_mass_matrix(jlp, x_h)
                print(f"  imm range: [{float(init_imm.min()):.4f}, {float(init_imm.max()):.4f}]", flush=True)
            elif self.mass_matrix_init == "hessian_dense":
                print("Estimating dense initial mass matrix from full Hessian...", flush=True)
                x_h = initial_positions[0]
                if not self.minimize_and_sample:
                    x_h = self._best_fit_position(jlp, x_h)
                init_imm = self._hessian_mass_matrix_dense(jlp, x_h)
                ev = jnp.linalg.eigvalsh(init_imm)
                print(
                    f"  dense imm eigenvalue range: "
                    f"[{float(ev.min()):.3e}, {float(ev.max()):.3e}]",
                    flush=True,
                )
            elif self.mass_matrix_init == "mclmc":
                print("Estimating initial mass matrix from MCLMC chains...", flush=True)
                rng_key, mm_key = jax.random.split(rng_key)
                init_imm, mm_info = self._mclmc_mass_matrix(
                    jlp, initial_positions, mm_key
                )
                print(
                    f"  rung={mm_info['rung']} survivors={mm_info['n_survivors']} "
                    f"imm range: [{float(init_imm.min()):.3e}, "
                    f"{float(init_imm.max()):.3e}]",
                    flush=True,
                )
            else:
                init_imm = None

            init_step = None
            if self.warmup_init_file is not None:
                with open(self.warmup_init_file, "r") as fp:
                    winit = json.load(fp)
                init_imm = jnp.array(winit["inverse_mass_matrix"])
                init_step = float(np.asarray(winit["step_size"]))
                print(
                    f"Warm-starting adaptation from {self.warmup_init_file}",
                    flush=True,
                )

            if self.warmup_algorithm == "adaptive_window":
                print("Running adaptive window warmup", flush=True)
                keys = jax.random.split(rng_key, 3)
                rng_key = keys[0]
                state, parameters = self._adaptive_window_warmup(
                    jlp, keys[1], initial_positions[0],
                    initial_inverse_mass_matrix=init_imm,
                    output_file=output_file, initial_step_size=init_step,
                )
                inverse_mass_matrix = parameters["inverse_mass_matrix"]
                step_size = parameters["step_size"]
                states = self._jitter_positions(
                    keys[2], state.position, inverse_mass_matrix, n_chains
                )
                nuts = blackjax.nuts(
                    kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size,
                    max_num_doublings=self.max_num_doublings,
                )
                states = self._map_chains(nuts.init)(states)

            elif self.warmup_algorithm == "meads":
                keys = jax.random.split(rng_key, 3)
                rng_key = keys[0]
                state, parameters = self._meads_warmup(
                    jlp, keys[1], initial_positions
                )
                inverse_mass_matrix = parameters["inverse_mass_matrix"]
                step_size = parameters["step_size"]
                states = self._jitter_positions(
                    keys[2], state.position, inverse_mass_matrix, n_chains
                )
                nuts = blackjax.nuts(
                    kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size,
                    max_num_doublings=self.max_num_doublings,
                )
                states = self._map_chains(nuts.init)(states)

            elif self.warmup_algorithm == "chees":
                if init_imm is not None:
                    print(
                        "ChEES adaptation uses an identity mass matrix; "
                        "ignoring initial mass matrix.",
                        flush=True,
                    )
                keys = jax.random.split(rng_key, 2)
                rng_key = keys[0]
                last_states, parameters = self._chees_warmup(
                    jlp, keys[1], initial_positions,
                    initial_step_size=init_step,
                )
                inverse_mass_matrix = parameters["inverse_mass_matrix"]
                step_size = parameters["step_size"]
                chees_algo = self._chees_kernel(
                    kernel_lp, step_size, inverse_mass_matrix,
                    parameters["trajectory_length_adjusted"],
                    parameters["halton_max_bits"],
                )
                # Per-chain adaptation endpoints are already distinct
                # (overdispersed) starts; no jitter needed.
                states = self._map_chains(chees_algo.init, chain_axes=(0, 0))(
                    last_states.position, last_states.random_generator_arg
                )

            elif self.warmup_algorithm == "pooled_window":
                # Free minimization/Hessian executables and their workspace;
                # the single-device 128-chain warmup needs the full pool.
                jax.clear_caches()
                keys = jax.random.split(rng_key, 2)
                rng_key = keys[0]
                final_positions, parameters = self._pooled_window_warmup(
                    jlp, keys[1], initial_positions,
                    initial_inverse_mass_matrix=init_imm,
                    initial_step_size=init_step, output_file=output_file,
                )
                inverse_mass_matrix = parameters["inverse_mass_matrix"]
                step_size = parameters["step_size"]
                # Per-chain warmup endpoints are already distinct
                # (overdispersed) starts; no jitter needed.
                nuts = blackjax.nuts(
                    kernel_lp, inverse_mass_matrix=inverse_mass_matrix,
                    step_size=step_size,
                    max_num_doublings=self.max_num_doublings,
                )
                states = self._map_chains(nuts.init)(final_positions)

            elif self.pathfinder_adaptation:
                print("Running pathfinder adaptation", flush=True)

                warmup = blackjax.pathfinder_adaptation(
                    blackjax.nuts,
                    jlp,
                )
                if self.parallel_warmup:
                    warmup_map = self._map_chains(
                        lambda k, p: warmup.run(k, p, self.n_steps_warmup),
                        chain_axes=(0, 0),
                    )
                    keys = jax.random.split(rng_key, 1 + n_chains)
                    rng_key = keys[0]
                    warmup_keys = keys[1:]
                    (states, parameters), _ = warmup_map(
                        warmup_keys, initial_positions
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
                    states = self._jitter_positions(
                        keys[1], state.position, inverse_mass_matrix, n_chains
                    )
                    nuts = blackjax.nuts(
                        kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size,
                        max_num_doublings=self.max_num_doublings,
                    )
                    states = self._map_chains(nuts.init)(states)

            else:
                print("Running window adaptation", flush=True)

                warmup_kwargs = dict(
                    is_mass_matrix_diagonal=self.diagonal_mass_matrix,
                    progress_bar=False,
                    initial_step_size=(
                        init_step if init_step is not None else self.step_size_init
                    ),
                    target_acceptance_rate=self.target_acceptance_rate,
                )
                if init_imm is not None:
                    if _HAS_INITIAL_IMM:
                        warmup_kwargs["initial_inverse_mass_matrix"] = init_imm
                    else:
                        print(
                            "Installed blackjax window_adaptation does not accept "
                            "initial_inverse_mass_matrix; ignoring initial mass "
                            "matrix.",
                            flush=True,
                        )

                warmup = blackjax.window_adaptation(blackjax.nuts, jlp, **warmup_kwargs)
                if self.parallel_warmup:
                    warmup_map = self._map_chains(
                        lambda k, p: warmup.run(k, p, self.n_steps_warmup),
                        chain_axes=(0, 0),
                    )
                    keys = jax.random.split(rng_key, 1 + n_chains)
                    rng_key = keys[0]
                    warmup_keys = keys[1:]
                    (states, parameters), _ = warmup_map(
                        warmup_keys, initial_positions
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
                    states = self._jitter_positions(
                        keys[1], state.position, inverse_mass_matrix, n_chains
                    )
                    nuts = blackjax.nuts(
                        kernel_lp, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size,
                        max_num_doublings=self.max_num_doublings,
                    )
                    states = self._map_chains(nuts.init)(states)

            warmup_parameters = {
                "inverse_mass_matrix": inverse_mass_matrix.tolist(),
                "step_size": step_size.tolist(),
                "initial_state": np.asarray(
                    gather_to_host(states.position)
                ).tolist(),
            }
            if "max_num_doublings" in parameters:
                warmup_parameters["max_num_doublings"] = int(
                    parameters["max_num_doublings"]
                )
            if self.warmup_algorithm == "chees":
                warmup_parameters.update(
                    trajectory_length_adjusted=parameters[
                        "trajectory_length_adjusted"
                    ],
                    halton_max_bits=parameters["halton_max_bits"],
                    random_generator_arg=int(
                        np.asarray(last_states.random_generator_arg)[0]
                    ),
                )

            if is_io_process():
                with open(f"{output_file}.nuts_warmup_parameters.json", "w") as fp:
                    json.dump(warmup_parameters, fp)

            if self.warmup_algorithm == "chees":
                algo = chees_algo
            else:
                algo = blackjax.nuts(
                    kernel_lp, inverse_mass_matrix=inverse_mass_matrix,
                    step_size=step_size,
                    max_num_doublings=self.max_num_doublings,
                )
            kernel = algo.step
            samples = None
            log_density = None

        keys = jax.random.split(rng_key, 1 + n_chains)
        rng_key = keys[0]
        sample_keys = keys[1:]

        if self.mesh is None:
            pmap_inference_loop = self._make_pmap_inference_loop(
                collect_info="stats"
            )
        else:
            pmap_inference_loop = self._make_mesh_inference_loop(
                kernel, self.n_steps_incr
            )

        # Built once: rebuilding the mapped init every batch retriggers
        # tracing/compilation (notably in mesh mode).
        if self.warmup_algorithm == "chees":
            init_map = self._map_chains(algo.init, chain_axes=(0, 0))

            def reinit_fn(states, rng_key):
                return (
                    init_map(
                        states.position[:, -1, :],
                        states.random_generator_arg[:, -1],
                    ),
                    rng_key,
                )

        else:
            init_map = self._map_chains(algo.init)

            def reinit_fn(states, rng_key):
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
            n_chains,
            output_file,
            reinit_fn,
            max_divergence_rate=self.max_divergence_rate,
            fail_on_divergence=self.fail_on_divergence,
            divergence_check_min_steps=self.divergence_check_min_steps,
        )

        return self._finalize_samples(
            samples, log_density, sigmas, reference, param_names
        )
