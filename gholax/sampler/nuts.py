import inspect
import json
import os

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseSampler



def _stale_warmup_parameters(output_file):
    """True when a finished-warmup file coexists with a newer intermediate
    checkpoint, i.e. a later warmup was started for the same prefix."""
    if not output_file:
        return False
    params = f"{output_file}.nuts_warmup_parameters.json"
    inter = f"{output_file}.nuts_warmup_intermediate.json"
    return (
        os.path.exists(params)
        and os.path.exists(inter)
        and os.path.getmtime(inter) > os.path.getmtime(params)
    )

def _spd_sqrt(A, power=0.5):
    lam, V = jnp.linalg.eigh(A)
    lam = jnp.clip(lam, 1e-30)
    return (V * lam**power) @ V.T


def _spd_guard(S):
    """Symmetrize and floor eigenvalues so the result is SPD."""
    S = 0.5 * (S + S.T)
    lam, V = jnp.linalg.eigh(S)
    lam = jnp.maximum(lam, 1e-6 * jnp.max(lam))
    return (V * lam) @ V.T


def _fisher_metric(C, G, rank, cutoff, reg, n=None):
    """Inverse mass matrix minimizing the sample Fisher divergence to a
    standard normal (Seyboldt, Carlson & Carpenter 2026): C = cov(draws),
    G = cov(scores).  Per-coordinate scales sigma^2 = sqrt(diag C / diag G)
    are read off first; the correlation structure comes from the geometric
    mean of the sigma-whitened covariances, Tikhonov-regularized toward the
    identity there (reg, raised to at least dim/n so directions n < dim
    samples cannot identify fall back to the diagonal scale).  rank None ->
    full whitened geometric mean; "auto"/int -> keep only eigen-directions
    with eigenvalue outside [1/cutoff, cutoff] (int: that many farthest
    from 1)."""
    dim = C.shape[0]
    if n is not None:
        reg = max(reg, dim / max(n, 1))
    cx = jnp.clip(jnp.diag(C), 1e-30)
    cg = jnp.clip(jnp.diag(G), 1e-30)
    sig = jnp.sqrt(jnp.sqrt(cx / cg))
    Cw = C / jnp.outer(sig, sig) + reg * jnp.eye(dim)
    Gw = G * jnp.outer(sig, sig) + reg * jnp.eye(dim)
    # Sw = Gw^-1/2 (Gw^1/2 Cw Gw^1/2)^1/2 Gw^-1/2 solves Sw Gw Sw = Cw
    Gh, Gmh = _spd_sqrt(Gw, 0.5), _spd_sqrt(Gw, -0.5)
    Sw = Gmh @ _spd_sqrt(Gh @ Cw @ Gh, 0.5) @ Gmh
    if rank is not None:
        lam, U = jnp.linalg.eigh(0.5 * (Sw + Sw.T))
        if rank == "auto":
            keep = (lam >= cutoff) | (lam <= 1.0 / cutoff)
        else:
            order = jnp.argsort(-jnp.abs(jnp.log(jnp.clip(lam, 1e-30))))
            keep = jnp.zeros(dim, bool).at[order[: int(rank)]].set(True)
        lam_k = jnp.where(keep, lam - 1.0, 0.0)
        Sw = jnp.eye(dim) + (U * lam_k) @ U.T
        print(f"  fisher low-rank metric: kept {int(jnp.sum(keep))} directions "
              f"outside [1/{cutoff:g}, {cutoff:g}]", flush=True)
    return _spd_guard(Sw * jnp.outer(sig, sig))


class NUTS(BaseSampler):
    """No-U-Turn Sampler using blackjax.

    Wraps blackjax's NUTS sampler with window adaptation warmup, convergence
    checking via R-hat, parallel chains via jax.pmap, and checkpoint restart.
    """

    WARMUP_ALGORITHMS = ("window", "adaptive_window", "pooled_window")

    def __init__(self, config):
        """Initialize NUTS sampler from config.

        Args:
            config: Full config dict containing 'sampler' -> 'NUTS' section.
        """
        c = config["sampler"]["NUTS"]
        self._sampler_cfg = c

        self.n_steps_warmup = c.get("n_steps_warmup", 500)
        self.target_r_minus_one = c.get("target_r_minus_one", 0.1)
        # Optional worst-dimension ESS floor for the stopping rule (ANDed
        # with the R-hat gate); None disables it.
        self.target_min_ess = c.get("target_min_ess", None)
        if self.target_min_ess is not None:
            self.target_min_ess = float(self.target_min_ess)
            if self.target_min_ess <= 0:
                raise ValueError("target_min_ess must be > 0")
        self.n_steps_incr = c.get("n_steps_incr", 10)
        # With an ESS target and no explicit n_steps_min, let ESS govern the
        # minimum run length instead of the fixed default.
        if "n_steps_min" in c or self.target_min_ess is None:
            self.n_steps_min = c.get("n_steps_min", 250)
        else:
            self.n_steps_min = 0
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
        # sampling cap = q-quantile warmup depth + margin; a larger margin
        # applies when warmup depth was itself capped (quantile at the cap)
        self.depth_cap_margin = int(c.get("depth_cap_margin", 1))
        self.depth_cap_margin_saturated = int(c.get("depth_cap_margin_saturated", 2))
        self.minimize_start_scale = c.get("minimize_start_scale", 0.5)
        # Pathfinder chain seeding: replaces MAP tiling with draws from the
        # ELBO-best Gaussians along L-BFGS paths (the MAP is still found
        # and saved).
        self.pathfinder_init = c.get("pathfinder_init", True)
        self.pathfinder_resample = c.get("pathfinder_resample", False)
        self.pathfinder_n_paths = c.get(
            "pathfinder_n_paths", self.minimize_n_starts
        )
        self.pathfinder_elbo_samples = c.get("pathfinder_elbo_samples", 20)
        self.pathfinder_maxiter = c.get("pathfinder_maxiter", 100)
        self.pathfinder_maxcor = c.get("pathfinder_maxcor", 10)
        self.pathfinder_start_scale = c.get(
            "pathfinder_start_scale", self.minimize_start_scale
        )
        self.target_acceptance_rate = c.get("target_acceptance_rate", 0.8)
        self.step_size_init = c.get("step_size_init", 0.05)
        
        # Divergences remain diagnostics by default for backwards
        # compatibility.  Production runs can opt into a fail-closed rate
        # threshold; with fail_on_divergence and no threshold, any divergent
        # transition raises.
        self.max_divergence_rate = c.get("max_divergence_rate", 0.5)
        
        if self.max_divergence_rate is not None and self.max_divergence_rate >= 0.0:
            self.max_divergence_rate = float(self.max_divergence_rate)
            if not 0.0 <= self.max_divergence_rate <= 1.0:
                raise ValueError("max_divergence_rate must be between 0 and 1")
        self.fail_on_divergence = bool(c.get("fail_on_divergence", True))
        self.divergence_check_min_steps = int(
            c.get("divergence_check_min_steps", 10)
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
        # "ones", "hessian", "hessian_dense", or "pathfinder" (default when
        # pathfinder_init is on, else hessian_dense)
        self.mass_matrix_init = c.get(
            "mass_matrix_init",
            "pathfinder" if self.pathfinder_init else "hessian_dense",
        )
        if self.mass_matrix_init == "fisher_seeds" and not self.pathfinder_init:
            raise ValueError("mass_matrix_init: fisher_seeds requires pathfinder_init: true")
        if self.mass_matrix_init == "pathfinder" and not self.pathfinder_init:
            raise ValueError(
                "mass_matrix_init: pathfinder requires pathfinder_init: true"
            )
        
        # Optional path to a previous run's .nuts_warmup_parameters.json used
        # to warm-start adaptation.
        self.warmup_init_file = c.get("warmup_init_file", None)
        self.warmup_algorithm = c.get("warmup_algorithm", "pooled_window")
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

        # Pooled window parameters (convergence rtols shared with
        # adaptive_window: adaptive_warmup_rtol_mass / adaptive_warmup_rtol_step)
        self.pooled_window_steps = c.get("pooled_window_steps", 25)
        self.pooled_window_max_steps = c.get("pooled_window_max_steps", 200)
        self.pooled_window_max_doublings = c.get("pooled_window_max_doublings", 8)
        
        # A short correlated tail does not give the between/within ratio a
        # universal tau/T null distribution.  Gate relative chain agreement
        # directly with the maximum ordinary PSR (R-hat) over coordinates,
        # after enough tail draws have accumulated.  This is a warmup safety
        # check, not evidence that every posterior mode has been explored.
        self.pooled_window_mixing_rhat = float(
            c.get("pooled_window_mixing_rhat", 1.2)
        )
        # Gate on this quantile of the per-parameter tail R-hat (1.0 = max):
        # a few genuinely multimodal nuisance parameters otherwise pin the
        # max near 2 and the gate can never open.
        self.pooled_window_mixing_quantile = float(
            c.get("pooled_window_mixing_quantile", 1.0)
        )
        if not 0.0 < self.pooled_window_mixing_quantile <= 1.0:
            raise ValueError("pooled_window_mixing_quantile must be in (0, 1]")
        if self.pooled_window_mixing_rhat < 1.0:
            raise ValueError("pooled_window_mixing_rhat must be >= 1")
        self.pooled_window_min_tail_steps = int(
            c.get("pooled_window_min_tail_steps", 8)
        )
        self.pooled_window_consecutive_windows = int(
            c.get("pooled_window_consecutive_windows", 2)
        )
        # Metric-change statistic: "rms_diag" (RMS relative change of the
        # per-parameter variances; noise floor ~sqrt(2/n_eff)) or the older
        # "max_diag" (max over parameters; floor inflated by sqrt(2 ln d)).
        self.pooled_window_mass_stat = c.get("pooled_window_mass_stat", "rms_diag")
        if self.pooled_window_mass_stat not in ("rms_diag", "max_diag"):
            raise ValueError("pooled_window_mass_stat must be 'rms_diag' or 'max_diag'")
        # tolerances: pooled_window_rtol_* > an explicit adaptive_warmup_rtol_*
        # (legacy shared key) > 0.1 for rms_diag / the adaptive default
        def _rtol(key, legacy):
            if key in c:
                return float(c[key])
            if legacy in c or self.pooled_window_mass_stat == "max_diag":
                return float(getattr(self, legacy))
            return 0.1
        self.pooled_window_rtol_mass = _rtol("pooled_window_rtol_mass", "adaptive_warmup_rtol_mass")
        self.pooled_window_rtol_step = _rtol("pooled_window_rtol_step", "adaptive_warmup_rtol_step")
        if self.pooled_window_min_tail_steps < 2:
            raise ValueError("pooled_window_min_tail_steps must be >= 2")
        if self.pooled_window_consecutive_windows < 1:
            raise ValueError("pooled_window_consecutive_windows must be >= 1")
        self.pooled_window_max_window = c.get("pooled_window_max_window", 40)
        # Metric estimator at boundaries: "covariance" (pooled within-chain
        # covariance of draws) or "fisher" (Seyboldt, Carlson & Carpenter
        # 2026: geometric mean of cov(draws) and cov(scores)^-1, which
        # conditions stiff and soft directions equally and needs no extra
        # gradient calls since scores come with the NUTS states).
        self.pooled_window_metric_estimator = c.get(
            "pooled_window_metric_estimator", "covariance"
        )
        if self.pooled_window_metric_estimator not in ("covariance", "fisher"):
            raise ValueError("pooled_window_metric_estimator must be 'covariance' or 'fisher'")
        # fisher low-rank: keep eigen-directions of the whitened geometric
        # mean with eigenvalue outside [1/c, c]; relative Tikhonov regularizer
        self.pooled_window_fisher_cutoff = float(c.get("pooled_window_fisher_cutoff", 1.5))
        self.pooled_window_fisher_reg = float(c.get("pooled_window_fisher_reg", 1e-4))
        # Dense-update estimator: None = shrunk full covariance; "auto" =
        # diagonal + low rank keeping only correlation eigenvalues outside the
        # Marchenko-Pastur noise bulk [(1-sqrt(d/n))^2, (1+sqrt(d/n))^2];
        # an int = fixed number of directions farthest from 1.
        self.pooled_window_dense_rank = c.get("pooled_window_dense_rank", None)
        if self.pooled_window_dense_rank not in (None, "auto") and not (
            isinstance(self.pooled_window_dense_rank, int)
            and self.pooled_window_dense_rank >= 0
        ):
            raise ValueError("pooled_window_dense_rank must be None, 'auto' or a non-negative int")

        # Re-estimate a dense metric from the pooled within-chain tail
        # covariance at every boundary with a long-enough tail (Stan-style),
        # instead of freezing the MAP Hessian until convergence.
        self.pooled_window_dense_update = bool(
            c.get("pooled_window_dense_update", True)
        )
        
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
            "pooled_window_allow_unconverged", True
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

            warmup_kwargs = {'is_mass_matrix_diagonal':self.diagonal_mass_matrix,
                             'progress_bar':False,
                             'initial_step_size':step_size_init,
                             'target_acceptance_rate':self.target_acceptance_rate}
            
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

    @staticmethod
    def _stuck_chains(pos):
        """Chains that never moved in a window, and the ones that did.

        A chain whose every proposal diverges never moves.  Left alone it
        feeds frozen values into the pooled tail covariance and the mixing
        R-hat for the rest of warmup, and is still frozen at sampling.
        Positions are exchangeable before convergence, so re-seeding from a
        chain that did move is initialization, not a chain edit.

        Args:
            pos: (window_steps, n_chains, dim) window positions.

        Returns:
            (stuck_idx, live_idx).  Both empty when nothing is stuck or when
            no chain moved at all (a global failure re-seeding cannot fix).
        """
        empty = jnp.zeros((0,), dtype=int)
        if pos.shape[0] < 2 or pos.shape[1] < 2:
            return empty, empty
        moved = jnp.abs(jnp.diff(pos, axis=0)).sum(axis=(0, 2)) > 0
        stuck, live = jnp.where(~moved)[0], jnp.where(moved)[0]
        if stuck.size == 0 or live.size == 0:
            return empty, empty
        return stuck, live

    def _pooled_window_warmup(self, jlp, rng_key, initial_positions,
                              initial_inverse_mass_matrix=None,
                              initial_step_size=None, output_file=None):
        """Cross-chain windowed warmup that keeps NUTS end to end.

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
        param_names = getattr(self, "_param_names", None)
        kernel = nuts_mcmc.build_kernel()

        da_init, da_update, da_final = dual_averaging_adaptation(
            target=self.target_acceptance_rate
        )

        # Adaptation runs pmap(vmap): K chains vmapped per device, devices
        # synchronized through an axis-name pmean so the per-step
        # dual-averaging update sees the acceptance averaged across ALL
        # chains — every device advances an identical da_state. This uses
        # all GPUs during warmup and spreads the memory footprint.
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

        #reshape for pmap
        def _dev_split(tree):
            return jax.tree.map(
                lambda x: x.reshape((nd, K) + x.shape[1:]), tree
            )

        #inv of _dev_split
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
                    states.logdensity_grad,
                    acc,
                    infos.num_integration_steps,
                )

            (states, da_state), (pos, grad, acc, n_leap) = jax.lax.scan(
                one_step, (states, da_state), jax.random.split(dev_key, n_steps)
            )
            return states, da_state, pos, grad, acc, n_leap

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
            states, da_dev, pos, grad, acc, n_leap = prun_window(
                jax.random.split(key, nd), states, da_state, imm, n_steps
            )
            # da_state is device-invariant (pmean-synchronized); host
            # consumers get flat (n_steps, n_chains, ...) arrays.
            da_state = jax.tree.map(lambda x: x[0], da_dev)
            pos = jnp.moveaxis(pos, 0, 1).reshape(n_steps, n_chains, -1)
            grad = jnp.moveaxis(grad, 0, 1).reshape(n_steps, n_chains, -1)
            n_leap = jnp.moveaxis(n_leap, 0, 1).reshape(n_steps, n_chains)
            return states, da_state, pos, grad, acc[0], n_leap

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
        resumed_dense_updated = False
        if initial_inverse_mass_matrix is not None and initial_step_size is not None:
            # Warm start: seed the convergence check so a single window can
            # suffice.
            prev_mass = imm
            prev_step = step_size
        total_steps = 0
        resumed_converged = False

        ckpt_file = (
            f"{output_file}.nuts_warmup_intermediate.json" if output_file else None
        )
        if ckpt_file and self.restart and os.path.exists(ckpt_file):
            with open(ckpt_file, "r") as fp:
                ck = json.load(fp)
            if bool(ck.get("sample_transform", False)) != bool(
                getattr(getattr(self, "_prior", None), "transform", False)
            ):
                raise RuntimeError(
                    "Restart refused: pooled warmup checkpoint uses a "
                    "different sampling coordinate convention "
                    "(sample_transform mismatch)."
                )
            if "positions" in ck:  # written by this warmup mode
                positions = jnp.array(ck["positions"])
                imm = jnp.array(ck["inverse_mass_matrix"])
                step_size = jnp.asarray(ck["step_size"])
                prev_mass = imm
                prev_step = step_size
                total_steps = ck["total_steps"]
                stable_boundaries = int(ck.get("stable_boundaries", 0))
                resumed_dense_updated = bool(ck.get("dense_updated", False))
                resumed_converged = bool(ck.get("warmup_converged", False)) and not bool(
                    ck.get("calibrated", False)
                )
                bw_ratio = ck.get("between_within_ratio", float("nan"))
                tail_rhat = ck.get("tail_rhat", float("nan"))
                completed_window_chunks = int(ck.get("window_chunks", 1))
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
        def dense_from_tail(tail, step, states, rng_key, reseed=True,
                            grad_tail=None):
            """Dense imm from the pooled within-chain tail (covariance of
            draws, or the Fisher-divergence geometric mean of draw and
            score covariances when scores are given and the estimator is
            "fisher"), eigenvalue-guarded, with eps re-seeded."""
            xc = tail - tail.mean(axis=0, keepdims=True)
            n_cov = xc.shape[0] * xc.shape[1]
            C = jnp.einsum("tcd,tce->de", xc, xc) / max(n_cov - 1, 1)
            rank = self.pooled_window_dense_rank
            if self.pooled_window_metric_estimator == "fisher" and grad_tail is not None:
                gc = grad_tail - grad_tail.mean(axis=0, keepdims=True)
                G = jnp.einsum("tcd,tce->de", gc, gc) / max(n_cov - 1, 1)
                C = _fisher_metric(
                    C, G, rank, self.pooled_window_fisher_cutoff,
                    self.pooled_window_fisher_reg, n=n_cov,
                )
            elif rank is None:
                alpha = dim / (dim + n_cov)
                C = (1 - alpha) * C + alpha * jnp.diag(jnp.diag(C))
            else:
                # diagonal + low rank: keep correlation eigen-directions above noise
                sd = jnp.sqrt(jnp.clip(jnp.diag(C), 1e-30))
                R = C / jnp.outer(sd, sd)
                lr, Ur = jnp.linalg.eigh(R)
                if rank == "auto":
                    # keep directions above noise on either side
                    q = np.sqrt(dim / max(n_cov, 1))
                    keep = (lr > (1 + q) ** 2) | (lr < (1 - q) ** 2)
                else:
                    order = jnp.argsort(-jnp.abs(lr - 1.0))
                    keep = jnp.zeros(dim, bool).at[order[: int(rank)]].set(True)
                n_keep = int(jnp.sum(keep))
                lr_k = jnp.where(keep, lr - 1.0, 0.0)
                R_lr = jnp.eye(dim) + (Ur * lr_k) @ Ur.T
                C = R_lr * jnp.outer(sd, sd)
                print(f"  low-rank metric: kept {n_keep} correlation "
                      f"directions outside the noise bulk", flush=True)
            lam, V = jnp.linalg.eigh(C)
            ok = jnp.isfinite(lam) & (lam > 0)
            med = jnp.nanmedian(jnp.where(ok, lam, jnp.nan))
            med = jnp.where(jnp.isfinite(med) & (med > 0), med, jnp.asarray(1.0))
            lam = jnp.where(ok, lam, med)
            lam = jnp.clip(jnp.maximum(lam, 1e-4 * med), 1e-6, 1e6)
            imm = (V * lam) @ V.T
            ev = jnp.linalg.eigvalsh(imm)
            print(
                f"Dense metric updated from pooled tail covariance "
                f"({n_cov} samples): eigenvalue range "
                f"[{float(ev.min()):.3e}, {float(ev.max()):.3e}]",
                flush=True,
            )
            if not reseed:
                return imm, step, rng_key
            rng_key, srch_key = jax.random.split(rng_key)
            step = float(
                find_reasonable_step_size(
                    srch_key,
                    lambda eps: lambda k, s: kernel(
                        k, s, jlp, eps, imm, max_num_doublings=max_doublings,
                    ),
                    nuts_mcmc.init(jnp.asarray(_flat_positions(states))[0], jlp),
                    float(step),
                    target_accept=self.target_acceptance_rate,
                )
            )
            print(f"Step size re-seeded for updated metric: {step:.3g}", flush=True)
            return imm, step, rng_key

        calibration_leap = None
        converged = False
        calibrated = False
        dense_updated = resumed_dense_updated
        last_diag = {}

        def calibrate(states, step, imm, rng_key, total_steps):
            """Fixed-metric epsilon calibration after convergence; returns
            (states, step, calibration_leap, total_steps, calibrated, rng_key)
            and writes the calibrated checkpoint."""
            terminal_steps = min(
                max(0, int(self.pooled_window_terminal_steps)),
                self.pooled_window_max_steps - total_steps,
            )
            if not terminal_steps:
                return states, step, None, total_steps, False, rng_key
            rng_key, terminal_key = jax.random.split(rng_key)
            terminal_da = da_init(float(step))
            states, terminal_da, _, _, _, terminal_leap = run_window(
                terminal_key, states, terminal_da, imm, terminal_steps
            )
            step = da_final(terminal_da)
            total_steps += terminal_steps
            print(
                f"Pooled fixed-metric calibration: {terminal_steps} steps, "
                f"step_size={float(step):.5g}",
                flush=True,
            )
            if ckpt_file and is_io_process():
                with open(ckpt_file, "w") as fp:
                    json.dump(
                        {
                            "inverse_mass_matrix": np.asarray(imm).tolist(),
                            "step_size": float(step),
                            "positions": _flat_positions(states).tolist(),
                            "sample_transform": bool(getattr(getattr(self, "_prior", None), "transform", False)),
                            "total_steps": total_steps,
                            "between_within_ratio": bw_ratio,
                            "tail_rhat": tail_rhat,
                            "stable_boundaries": stable_boundaries,
                            "window_chunks": completed_window_chunks,
                            "next_window_chunks": k,
                            "warmup_converged": True,
                            "calibrated": True,
                            **last_diag,
                        },
                        fp,
                    )
            return states, step, terminal_leap, total_steps, True, rng_key
        # Stan-style growing windows built from k base-length scans (one
        # compiled executable): short windows early for fast metric feedback,
        # doubling each boundary so the tail statistics and mixing check are
        # judged on enough samples to be useful.
        k_max = max(1, self.pooled_window_max_window // n_window)
        k = min(next_window_chunks, k_max)
        # Even the longest window's pooled tail must overdetermine the
        # dim x dim covariance, or dense updates can never happen (the
        # runtime gate below would skip every boundary).
        max_win = k_max * n_window
        max_tail = max_win - max_win // 5
        if (
            self.pooled_window_dense_update
            and imm.ndim == 2
            and n_chains * max_tail <= dim
        ):
            raise ValueError(
                f"pooled_window_dense_update can never run: the longest "
                f"window's pooled tail has {n_chains} * {max_tail} <= {dim} "
                f"(n_params) samples; raise chains_per_device, "
                f"pooled_window_steps, or pooled_window_max_window."
            )
        while total_steps < self.pooled_window_max_steps:
            if resumed_converged:
                print(
                    "Resuming at a converged boundary; running calibration",
                    flush=True,
                )
                converged = True
                states, step, calibration_leap, total_steps, calibrated, rng_key = (
                    calibrate(states, float(step_size), imm, rng_key, total_steps)
                )
                step_size = jnp.asarray(step)
                print(f"Pooled warmup converged after {total_steps} steps", flush=True)
                break
            pos_c, grad_c, acc_c, leap_c = [], [], [], []

            remaining = self.pooled_window_max_steps - total_steps
            boundary_steps = min(k * n_window, remaining)
            if boundary_steps <= 0:
                break
            n_subwindows = (boundary_steps + n_window - 1) // n_window
            for i in range(n_subwindows):
                sub_steps = min(n_window, boundary_steps - i * n_window)
                rng_key, sub_key = jax.random.split(rng_key)
                states, da_state, p_i, g_i, a_i, l_i = run_window(
                    sub_key, states, da_state, imm, sub_steps
                )
                pos_c.append(p_i)
                grad_c.append(g_i)
                acc_c.append(a_i)
                leap_c.append(l_i)
                total_steps += sub_steps
            pos = jnp.concatenate(pos_c, axis=0)
            grad = jnp.concatenate(grad_c, axis=0)
            acc = jnp.concatenate(acc_c, axis=0)
            n_leap = jnp.concatenate(leap_c, axis=0)
            w_len = boundary_steps

            # re-seed stuck chains at window edges
            stuck, live = self._stuck_chains(pos)
            if stuck.size:
                rng_key, seed_key = jax.random.split(rng_key)
                donors = jax.random.choice(seed_key, live, (stuck.size,))
                flat = jnp.asarray(_flat_positions(states))
                states = _dev_split(vinit(flat.at[stuck].set(flat[donors])))
                pos = pos.at[:, stuck, :].set(pos[:, donors, :])
                grad = grad.at[:, stuck, :].set(grad[:, donors, :])
                print(
                    f"Re-seeded {stuck.size} stuck chain(s) "
                    f"{stuck.tolist()} from live chains",
                    flush=True,
                )

            # Per-chain diagnostics: fraction of window steps each chain
            # moved and its current log density (checkpointed for post-hoc
            # questions like "which chains never left the seed").
            chain_moved = np.asarray(
                (jnp.abs(jnp.diff(pos, axis=0)).sum(-1) > 0).mean(0)
            )
            chain_logp = np.asarray(jnp.asarray(states.logdensity)).reshape(-1)
            n_slow = int((chain_moved < 0.5).sum())
            last_diag.update(chain_logp=chain_logp.tolist(),
                             chain_moved_frac=chain_moved.tolist())
            print(
                f"  chains: logp median {np.median(chain_logp):.1f} "
                f"[{chain_logp.min():.1f}, {chain_logp.max():.1f}], "
                f"{n_slow} moved <50% of steps",
                flush=True,
            )

            # pos: (w_len, n_chains, dim)
            tail_start = min(max(w_len // 5, 0), max(w_len - 1, 0))
            tail = pos[tail_start:]
            grad_tail = grad[tail_start:]
            tail_ddof = 1 if tail.shape[0] > 1 else 0
            within = jnp.mean(jnp.var(tail, axis=0, ddof=tail_ddof), axis=0)

            between_ddof = 1 if n_chains > 1 else 0
            between = jnp.var(
                jnp.mean(tail, axis=0), axis=0, ddof=between_ddof
            )
            bw_ratio = float(jnp.max(between / (within + 1e-30)))
            tail_rhat = float("inf")
            mixing_ready = tail.shape[0] >= self.pooled_window_min_tail_steps

            # coordinates no chain has moved in have zero within-variance
            # and an undefined (inf) R-hat; judge mixing on the rest and
            # report the frozen count separately
            moving = within > 0
            n_frozen = int(jnp.sum(~moving))
            if mixing_ready and n_chains > 1 and bool(jnp.any(moving)):
                from blackjax.diagnostics import potential_scale_reduction

                rhat_vec = potential_scale_reduction(
                    tail[:, :, moving], chain_axis=1, sample_axis=0
                )
                q = self.pooled_window_mixing_quantile
                tail_rhat = float(
                    jnp.max(rhat_vec) if q >= 1.0 else jnp.quantile(rhat_vec, q)
                )
                if param_names is not None:
                    idx_moving = np.flatnonzero(np.asarray(moving))
                    top = np.argsort(-np.asarray(rhat_vec))[:5]
                    print("  worst tail R-hat: " + ", ".join(
                        f"{param_names[idx_moving[t]]}={float(rhat_vec[t]):.2f}"
                        for t in top)
                        + (f" (gate uses q{q:g}={tail_rhat:.2f})" if q < 1.0 else ""),
                        flush=True)
            mixing_ok = (
                mixing_ready
                and tail_rhat <= self.pooled_window_mixing_rhat
            )
            
            # Keep the boundary update conservative when chains disagree, but
            # do not interpret this shrinkage factor as an ESS estimate.
            
            disagreement = max(tail_rhat - 1.0, 0.0) if mixing_ready else 1.0
            tail_weight_steps = tail.shape[0] / (1.0 + disagreement)
            w = tail_weight_steps / (tail_weight_steps + 5)
            step = da_final(da_state)
            seed_step = float(step)
            if imm.ndim == 2:
                # Dense (hessian_dense) metric: frozen through warmup unless
                # pooled_window_dense_update re-estimates it from the tail.
                # eps is re-searched only on the first update (the
                # Hessian->covariance jump; find_reasonable_step_size has
                # factor-2 granularity); after that dual averaging carries it.
                mass = imm
                # The pooled tail must overdetermine the dim x dim
                # covariance or the estimate is singular.
                tail_ok = tail.shape[0] * n_chains > dim
                if (
                    self.pooled_window_dense_update
                    and mixing_ready
                    and not tail_ok
                ):
                    print(
                        f"Skipping dense metric update: pooled tail has "
                        f"{tail.shape[0] * n_chains} samples <= {dim} "
                        f"params; raise chains_per_device or "
                        f"pooled_window_steps for per-window updates.",
                        flush=True,
                    )
                if self.pooled_window_dense_update and mixing_ready and tail_ok:
                    C, seed_step, rng_key = dense_from_tail(
                        tail, seed_step, states, rng_key,
                        reseed=not dense_updated, grad_tail=grad_tail,
                    )
                    
                    # Evidence-weighted EMA toward the previous metric at
                    # every boundary (w -> 0 when chains disagree), so the
                    # seed metric is relaxed as soon as the pooled tail
                    # carries information instead of waiting for mixing.
                    mass = w * C + (1 - w) * imm
                    dense_updated = True
            else:
                mass = w * within + (1 - w) * imm
            
            # Boundary: take the averaged step size and restart dual
            # averaging around it with the new mass matrix (blackjax
            # window_adaptation's slow_final).
            da_state = da_init(seed_step)
            mean_acc = float(jnp.mean(acc))

            converged = False
            if prev_mass is not None:
                # Dense: per-parameter variances (element-wise ratios blow
                # up on near-zero off-diagonals; Frobenius of the full
                # matrix is dominated by estimator noise ~ sqrt(dim/n_eff)).
                d_new = jnp.diag(mass) if mass.ndim == 2 else mass
                d_old = jnp.diag(prev_mass) if mass.ndim == 2 else prev_mass
                rel = jnp.abs(d_new - d_old) / (jnp.abs(d_old) + 1e-10)
                if self.pooled_window_mass_stat == "rms_diag":
                    mass_change = float(jnp.sqrt(jnp.mean(rel**2)))
                else:
                    mass_change = float(jnp.max(rel))
                step_change = float(
                    jnp.abs(step - prev_step) / (jnp.abs(prev_step) + 1e-10)
                )
                # The max-over-dims change statistic has a noise floor set by
                # the tail-variance estimator: ~sqrt(2/n_eff) per dim,
                # inflated ~sqrt(2 ln d) by the max. A configured rtol below
                # that floor means nothing, so gate against
                # max(rtol, floor); eps co-moves with the metric at roughly
                # half the relative rate.
                mass_floor = 0.0
                if n_chains > 1 and tail.shape[0] > 1:
                    from blackjax.diagnostics import effective_sample_size

                    n_eff = float(jnp.nanmin(effective_sample_size(
                        tail, chain_axis=1, sample_axis=0
                    )))
                    if np.isfinite(n_eff) and n_eff > 0:
                        # per-variance relative error ~sqrt(2/n_eff); the
                        # max over d of them is inflated by ~sqrt(2 ln d),
                        # the RMS is not
                        infl = (
                            np.sqrt(2 * np.log(max(dim, 2)))
                            if self.pooled_window_mass_stat == "max_diag" else 1.0
                        )
                        mass_floor = 1.5 * infl * np.sqrt(2 / n_eff)
                rtol_mass = max(self.pooled_window_rtol_mass, mass_floor)
                rtol_step = max(self.pooled_window_rtol_step, 0.5 * mass_floor)
                stat_name = (
                    "rms_rel_mass_change" if self.pooled_window_mass_stat == "rms_diag"
                    else "max_rel_mass_change"
                )
                print(
                    f"Pooled warmup step {total_steps} (window {w_len}): "
                    f"{stat_name}={mass_change:.4f}, "
                    f"rel_step_change={step_change:.4f}, "
                    f"rtol_mass={rtol_mass:.3f}, rtol_step={rtol_step:.3f} "
                    f"(noise floor {mass_floor:.3f}), "
                    f"mean_acceptance={mean_acc:.3f}, "
                    f"max_between_within_ratio={bw_ratio:.2f}, "
                    f"max_tail_rhat={tail_rhat:.3f}, "
                    f"tail_steps={tail.shape[0]}"
                    + (f", frozen_dims={n_frozen}" if n_frozen else "")
                    + (
                        " (mixing gate not met)"
                        if not mixing_ok
                        else ""
                    ),
                    flush=True,
                )
                stable = (
                    mass_change < rtol_mass
                    and step_change < rtol_step
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
                            "sample_transform": bool(getattr(getattr(self, "_prior", None), "transform", False)),
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
                            "dense_updated": bool(dense_updated),
                            **last_diag,
                        },
                        fp,
                    )

            if converged:
                print(f"Pooled warmup convergence detected at {total_steps} steps", flush=True)
                if imm.ndim == 2 and not self.pooled_window_dense_update:
                    imm, step, rng_key = dense_from_tail(
                        tail, step, states, rng_key
                    )
                states, step, calibration_leap, total_steps, calibrated, rng_key = (
                    calibrate(states, float(step), imm, rng_key, total_steps)
                )
                step_size = jnp.asarray(step)
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
            saturated = q_depth >= self.pooled_window_max_doublings
            margin = (
                self.depth_cap_margin_saturated if saturated
                else self.depth_cap_margin
            )
            self.max_num_doublings = min(q_depth + margin, self.max_num_doublings)
            parameters["max_num_doublings"] = self.max_num_doublings
            print(
                f"Sampling depth cap: {self.max_num_doublings} "
                f"(q{self.depth_cap_quantile:g} warmup depth {q_depth}"
                + (f", at the warmup cap {self.pooled_window_max_doublings}: "
                   f"margin {margin}" if saturated else "")
                + ")",
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
            if self.warmup_algorithm == "pooled_window":
                raise ValueError(
                    "warmup_algorithm 'pooled_window' is not supported in "
                    "mesh mode (n_chains/model_shards or multi-process "
                    "runs); use 'adaptive_window' or 'window'."
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

        kernel_lp = log_posterior if self.mesh is not None else jlp

        if self.restart and _stale_warmup_parameters(output_file):
            print(
                "Ignoring nuts_warmup_parameters.json older than the "
                "nuts_warmup_intermediate.json checkpoint (resuming the newer warmup)",
                flush=True,
            )
        if (
            self.restart
            and os.path.exists(f"{output_file}.nuts_warmup_parameters.json")
            and not _stale_warmup_parameters(output_file)
        ):
            with open(f"{output_file}.nuts_warmup_parameters.json", "r") as fp:
                warmup_parameters = json.load(fp)

            if bool(warmup_parameters.get("sample_transform", False)) != bool(
                getattr(getattr(self, "_prior", None), "transform", False)
            ):
                raise RuntimeError(
                    "Restart refused: checkpointed chain and current config "
                    "use different sampling coordinates (sample_transform "
                    "mismatch). Finish the chain with its original "
                    "convention or start fresh."
                )
            inverse_mass_matrix = jnp.array(warmup_parameters["inverse_mass_matrix"])
            step_size = jnp.array(warmup_parameters["step_size"])
            if self.max_num_doublings_auto and "max_num_doublings" in warmup_parameters:
                self.max_num_doublings = int(warmup_parameters["max_num_doublings"])
            if os.path.exists(f"{output_file}.samples_chk.npy"):
                # The checkpoint stores physical-space samples, but the
                # convergence loop accumulates normalized ones and rescales on
                # every write. Convert back on load, otherwise each restart
                # re-applies sigma/reference to the whole resumed prefix.
                samples = self._from_physical(
                    np.load(f"{output_file}.samples_chk.npy"),
                    np.asarray(sigmas), np.asarray(reference),
                )
                log_density = np.load(f"{output_file}.logposterior_chk.npy")
                initial_state = samples[:, -1, :]
            else:
                samples = None
                log_density = None
                initial_state = jnp.array(warmup_parameters["initial_state"])

            algo = blackjax.nuts(
                kernel_lp, inverse_mass_matrix=inverse_mass_matrix,
                step_size=step_size,
                max_num_doublings=self.max_num_doublings,
            )
            states = self._map_chains(algo.init)(jnp.asarray(initial_state))
            kernel = algo.step

        else:
            # pooled-window warmup checkpoint supersedes minimization and
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
            pf_key = None
            if self.pathfinder_init and not warm_ckpt:
                rng_key, pf_key = jax.random.split(rng_key)
            if self.minimize_and_sample and not warm_ckpt:
                initial_positions = self._minimize_and_sample(
                    log_posterior, initial_positions, n_chains, output_file,
                    pathfinder_key=pf_key,
                )
            x_h = initial_positions[0]
            if pf_key is not None:
                seeds = getattr(self, "_pathfinder_positions", None)
                initial_positions = (
                    seeds if seeds is not None
                    else self._pathfinder_init(jlp, x_h, n_chains, pf_key, output_file)
                )

            if warm_ckpt:
                init_imm = None
            elif self.mass_matrix_init == "hessian":
                print("Estimating initial mass matrix from Hessian diagonal...", flush=True)
                if not self.minimize_and_sample:
                    x_h = self._best_fit_position(jlp, x_h)
                init_imm = self._hessian_mass_matrix(jlp, x_h)
                print(f"  imm range: [{float(init_imm.min()):.4f}, {float(init_imm.max()):.4f}]", flush=True)
            elif self.mass_matrix_init == "hessian_dense":
                print("Estimating dense initial mass matrix from full Hessian...", flush=True)
                if not self.minimize_and_sample:
                    x_h = self._best_fit_position(jlp, x_h)
                init_imm = self._hessian_mass_matrix_dense(jlp, x_h)
                ev = jnp.linalg.eigvalsh(init_imm)
                print(
                    f"  dense imm eigenvalue range: "
                    f"[{float(ev.min()):.3e}, {float(ev.max()):.3e}]",
                    flush=True,
                )
            elif self.mass_matrix_init == "fisher_seeds":
                # Fisher-divergence metric from the Pathfinder seed cloud and
                # its scores: n_chains draws with gradients, no extra warmup
                seeds = jnp.asarray(initial_positions)
                gs = jax.vmap(jax.grad(jlp))(seeds)
                xc = seeds - seeds.mean(0); gc = gs - gs.mean(0)
                C = xc.T @ xc / max(seeds.shape[0] - 1, 1)
                G = gc.T @ gc / max(seeds.shape[0] - 1, 1)
                init_imm = _fisher_metric(
                    C, G, self.pooled_window_dense_rank or "auto",
                    self.pooled_window_fisher_cutoff, self.pooled_window_fisher_reg,
                    n=int(seeds.shape[0]),
                )
                ev = jnp.linalg.eigvalsh(init_imm)
                print(
                    f"Dense initial mass matrix from Fisher divergence of the "
                    f"seed cloud: eigenvalue range [{float(ev.min()):.3e}, "
                    f"{float(ev.max()):.3e}]",
                    flush=True,
                )
            elif self.mass_matrix_init == "pathfinder":
                init_imm = self._pathfinder_imm
                ev = jnp.linalg.eigvalsh(init_imm)
                print(
                    f"Dense initial mass matrix from Pathfinder covariance: "
                    f"eigenvalue range [{float(ev.min()):.3e}, "
                    f"{float(ev.max()):.3e}]",
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

                warmup_kwargs["initial_inverse_mass_matrix"] = init_imm

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
                "sample_transform": bool(
                    getattr(getattr(self, "_prior", None), "transform", False)
                ),
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
            if is_io_process():
                with open(f"{output_file}.nuts_warmup_parameters.json", "w") as fp:
                    json.dump(warmup_parameters, fp)

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
