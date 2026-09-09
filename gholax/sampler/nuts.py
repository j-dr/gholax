import json
import os
from dataclasses import replace

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseSampler
from .warmup import (  # noqa: F401  (re-exported for callers/tests)
    LEGACY_ATTRS,
    NUTS_ALGORITHMS,
    Warmup,
    WarmupConfig,
    WarmupRequest,
    _fisher_metric,
    _spd_guard,
    _spd_sqrt,
    _stale_warmup_parameters,
    stuck_chains,
)


class NUTS(BaseSampler):
    """No-U-Turn Sampler using blackjax.

    Wraps blackjax's NUTS sampler with window adaptation warmup, convergence
    checking via R-hat, parallel chains via jax.pmap, and checkpoint restart.
    """

    WARMUP_ALGORITHMS = NUTS_ALGORITHMS

    def __init__(self, config):
        """Initialize NUTS sampler from config.

        Args:
            config: Full config dict containing 'sampler' -> 'NUTS' section.
        """
        c = config["sampler"]["NUTS"]
        self._sampler_cfg = c

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
        self.minimize_and_sample = c.get("minimize_and_sample", False)
        self.minimize_n_starts = c.get("minimize_n_starts", 4)
        
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

        self.warmup_config = replace(
            WarmupConfig.from_sampler_config(c, sampler="NUTS"),
            restart=self.restart,
        )
        # Mutable: the pooled warmup can cap it from the observed depth
        # distribution (see WarmupResult.max_num_doublings).
        self.max_num_doublings = self.warmup_config.sampling_max_num_doublings

    def __getattr__(self, name):
        """Forward the deprecated flat warmup attributes to warmup_config."""
        field = LEGACY_ATTRS.get(name)
        if field is not None and "warmup_config" in self.__dict__:
            return getattr(self.warmup_config, field)
        raise AttributeError(name)


    def _adaptive_window_warmup(self, jlp, rng_key, initial_position,
                                initial_inverse_mass_matrix=None,
                                output_file=None, initial_step_size=None):
        """Staged window adaptation; see gholax.sampler.warmup.Warmup."""
        result = Warmup(
            self.warmup_config, self._warmup_host("nuts")
        ).run_adaptive_window(
            WarmupRequest(
                jlp,
                initial_position,
                rng_key,
                initial_inverse_mass_matrix=initial_inverse_mass_matrix,
                initial_step_size=initial_step_size,
                output_file=output_file,
            )
        )
        return result.state, {
            "inverse_mass_matrix": result.inverse_mass_matrix,
            "step_size": result.step_size,
        }


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

    _stuck_chains = staticmethod(stuck_chains)

    def _pooled_window_warmup(self, jlp, rng_key, initial_positions,
                              initial_inverse_mass_matrix=None,
                              initial_step_size=None, output_file=None):
        """Pooled cross-chain warmup; see gholax.sampler.warmup.Warmup."""
        result = Warmup(
            self.warmup_config, self._warmup_host("nuts")
        ).run_pooled_window(
            WarmupRequest(
                jlp,
                initial_positions,
                rng_key,
                initial_inverse_mass_matrix=initial_inverse_mass_matrix,
                initial_step_size=initial_step_size,
                output_file=output_file,
            )
        )
        parameters = {
            "inverse_mass_matrix": result.inverse_mass_matrix,
            "step_size": result.step_size,
            "warmup_converged": result.converged,
            "warmup_calibrated": result.calibrated,
        }
        if result.max_num_doublings is not None:
            self.max_num_doublings = result.max_num_doublings
            parameters["max_num_doublings"] = result.max_num_doublings
        return result.positions, parameters


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
