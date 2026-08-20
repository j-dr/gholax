import json
from collections import namedtuple
from datetime import datetime

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from blackjax.diagnostics import potential_scale_reduction

from ..util.distributed import CHAIN_AXIS, gather_to_host, is_io_process

ChainSetup = namedtuple(
    "ChainSetup",
    [
        "rng_key",
        "param_names",
        "prior",
        "sigmas",
        "reference",
        "log_posterior",
        "jlp",
        "n_devices",
        "initial_positions",
    ],
)


class BaseSampler(object):
    """Shared machinery for the pmap-parallel samplers.

    Provides chain initialization, Hessian mass-matrix estimation, L-BFGS
    pre-minimization, the pmapped inference loop, checkpointing, the
    R-hat convergence loop, and final unscaling. Subclasses implement their
    own __init__ (config parsing), warmup/kernel construction, and run().

    Checkpoint file names and formats written here are a public API — they
    are read by gholax.util.postprocess_chain from external notebooks and
    by the samplers' own restart paths. Do not change paths or keys.

    Mesh mode: samplers that support it (NUTS, Minimize) set self.mesh from
    the sampler config (n_chains / model_shards keys, see
    gholax.util.distributed.build_mesh). Chains then map over the mesh
    'chains' axis via shard_map instead of pmap, and each chain's posterior
    is sharded over the 'model' axis. self.mesh is None by default, keeping
    the legacy one-chain-per-local-device pmap behavior.

    chains_per_device (sampler config key, default 1): batch K chains per
    device (per chain group in mesh mode) via jax.vmap inside the device
    map, so total chains = n_devices * K. Host-side arrays (initial
    positions, gathered samples, checkpoints) keep a single flat chain
    axis of size n_devices * K; the (n_devices, K) split exists only
    device-side.
    """

    mesh = None
    chains_per_device = 1

    def _init_chains(self, model, jit_logpost=True):
        """Seed the rng, extract prior scaling, and draw per-chain initial
        positions in normalized parameter space.

        In mesh mode the chain count comes from the mesh's 'chains' axis, the
        rng seed is broadcast from process 0 (all processes must draw
        identical warmup trajectories and initial positions), and jlp is the
        model-sharded jitted posterior. ChainSetup.log_posterior is always
        the raw (unwrapped) function: in mesh mode it is what kernels running
        *inside* the chain shard_map must close over.
        """
        seed = int(datetime.now().strftime("%Y%m%d%s"))
        if self.mesh is not None and jax.process_count() > 1:
            from jax.experimental import multihost_utils

            seed = int(
                multihost_utils.broadcast_one_to_all(jnp.uint32(seed % (2**31)))
            )
        rng_key = jax.random.key(seed)
        param_names = model.prior.params
        prior = model.prior

        sigmas = prior.get_prior_sigmas()
        reference = prior.get_reference_values()
        log_posterior = model.log_posterior_scaled_params

        if self.mesh is not None:
            n_devices = self.mesh.shape[CHAIN_AXIS]
        else:
            n_devices = jax.local_device_count()
        n_chains = n_devices * self.chains_per_device
        keys = jax.random.split(rng_key, n_chains + 1)
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

        if self.mesh is not None:
            # Activates pair-axis sharding on the model's likelihoods as a
            # side effect; log_posterior above then contains model-axis
            # collectives and may only run inside a shard_map.
            jlp = model.sharded_log_posterior_scaled_params(self.mesh)
        else:
            jlp = jax.jit(log_posterior) if jit_logpost else None

        return ChainSetup(
            rng_key,
            param_names,
            prior,
            sigmas,
            reference,
            log_posterior,
            jlp,
            n_devices,
            initial_positions,
        )

    def _chain_map(self, fn, chain_axes):
        """shard_map analog of jax.pmap over the mesh 'chains' axis.

        Args:
            fn: Function of positional args, one chain's worth each.
            chain_axes: Tuple with one entry per positional arg: 0 to split
                the leading (chain) axis, None to replicate.

        Returns:
            Callable mapping over chains; outputs gain a leading chain axis.
            fn executes once per chain group, replicated across the 'model'
            axis devices of that group (whose lockstep is maintained by the
            model-axis collectives inside the sharded posterior). With
            chains_per_device > 1 each group's block of K chains is vmapped
            through fn.
        """
        from jax.sharding import PartitionSpec as P

        in_specs = tuple(P(CHAIN_AXIS) if a == 0 else P() for a in chain_axes)

        if self.chains_per_device > 1:
            vfn = jax.vmap(
                fn, in_axes=tuple(0 if a == 0 else None for a in chain_axes)
            )

            def body(*bargs):
                return vfn(*bargs)

        else:

            def body(*bargs):
                unbatched = [
                    jax.tree.map(lambda x: x[0], b) if a == 0 else b
                    for b, a in zip(bargs, chain_axes)
                ]
                out = fn(*unbatched)
                return jax.tree.map(lambda x: jnp.asarray(x)[None], out)

        mapped = jax.shard_map(
            body,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=P(CHAIN_AXIS),
            check_vma=False,
        )
        return jax.jit(mapped)

    def _map_chains(self, fn, chain_axes=(0,)):
        """Map fn over chains: pmap in legacy mode, _chain_map in mesh mode.

        The mapped callable takes and returns a flat chain axis of size
        n_devices * chains_per_device; with chains_per_device > 1 in legacy
        mode it is reshaped to (n_devices, K) around a pmap(vmap(fn)).
        """
        if self.mesh is None:
            in_axes = tuple(0 if a == 0 else None for a in chain_axes)
            if self.chains_per_device == 1:
                return jax.pmap(fn, in_axes=in_axes)

            K = self.chains_per_device
            pfn = jax.pmap(jax.vmap(fn, in_axes=in_axes), in_axes=in_axes)

            def mapped(*args):
                nd = jax.local_device_count()
                split = tuple(
                    jax.tree.map(
                        lambda x: x.reshape((nd, K) + x.shape[1:]), a
                    )
                    if ax == 0
                    else a
                    for a, ax in zip(args, chain_axes)
                )
                out = pfn(*split)
                return jax.tree.map(
                    lambda x: x.reshape((-1,) + x.shape[2:]), out
                )

            return mapped
        return self._chain_map(fn, chain_axes)

    def _best_fit_position(self, jlp, position, n_adam=300, n_lbfgs_restarts=3,
                           n_starts=None, start_scale=None, rng_key=None):
        """Multi-start Adam warm-start then restarted L-BFGS to the MAP.

        A single descent can settle in a data-likelihood ridge far from the
        joint optimum (observed with FlowLikelihood configs: 3x2pt curvature
        traps the minimizer ~59 logp above the CMB+BAO-compatible basin), so
        The basin search is a vmapped Adam-only pass over all starts (a
        simple scan graph that compiles quickly; memory is fine with
        gradient checkpointing on every likelihood block). L-BFGS — whose
        zoom line search vmaps into an hours-long XLA compile — polishes
        only the argmin winner, reusing the cached single-point compile.
        """
        if n_starts is None:
            n_starts = getattr(self, "minimize_n_starts", 4)
        if start_scale is None:
            start_scale = getattr(self, "minimize_start_scale", 0.5)
        if rng_key is None:
            rng_key = jax.random.key(0)
        jn = jax.jit(lambda p: -jlp(p))
        # Hessian-guided jitter: per-coordinate scale ~ sqrt(imm) at the
        # reference concentrates displacement in soft/degenerate directions
        # (where distinct basins live) and barely moves stiff ones.
        # Normalized so the rms displacement is start_scale * sqrt(n_soft)
        # rather than start_scale * sqrt(dim) — full-space isotropic jitter
        # stranded 7/8 starts ~2e4 logp out.
        dim = position.shape[0]
        imm_ref = self._hessian_mass_matrix(jlp, position)
        w = jnp.sqrt(imm_ref)
        w = w / jnp.max(w)
        starts = [position]
        for i in range(n_starts - 1):
            ki = jax.random.fold_in(rng_key, i)
            for scale in (start_scale, start_scale / 5, 0.0):
                cand = position + scale * w * jax.random.normal(ki, (dim,))
                if bool(jn(cand) < 1e30):
                    break
            starts.append(cand)
        ps, vs = jax.vmap(
            lambda x0: self._adam_descent(jlp, x0, n_adam)
        )(jnp.stack(starts))
        vs = jnp.where(vs < 1e30, vs, jnp.inf)
        i = int(jnp.argmin(vs))
        if not bool(jnp.isfinite(vs[i])):
            print("  minimization failed; using initial position", flush=True)
            return position
        spread = (float(vs.max() - vs.min())
                  if bool(jnp.isfinite(vs).all()) else float("nan"))
        print(f"  adam multi-start: best -logp {float(vs[i]):.2f} "
              f"({n_starts} starts, spread {spread:.2f})", flush=True)
        print("  per-start -logp (start 0 = reference): "
              + ", ".join(f"{float(v):.1f}" for v in vs), flush=True)
        best_p, best_v = self._lbfgs_polish(
            jlp, ps[i], vs[i], n_lbfgs_restarts
        )
        print(f"  best fit logp: {float(-best_v):.2f}", flush=True)
        return best_p

    def _adam_descent(self, jlp, position, n_adam=300):
        """Adam descent tracking the best valid point.

        Adam has no line search, so the -FLT_MAX out-of-box plateau and fp32
        noise that stall jaxopt's zoom line search don't affect it; updates
        that land out of the box (logp <= -1e30) are rejected. Pure JAX
        (vmap-compatible); returns (best position, -logp), -logp >= 1e30 if
        no valid point was seen.
        """
        import optax

        jnlp = jax.jit(lambda p: -jlp(p))
        vgrad = jax.value_and_grad(jnlp)

        opt = optax.adam(optax.cosine_decay_schedule(1e-2, n_adam, 1e-1))

        def adam_step(carry, _):
            p, s, best_p, best_v = carry
            v, g = vgrad(p)
            updates, s = opt.update(g, s, p)
            p_new = optax.apply_updates(p, updates)
            valid = jnlp(p_new) < 1e30
            p = jnp.where(valid, p_new, p)
            better = (v < best_v) & (v > -1e30)
            best_p = jnp.where(better, p, best_p)
            best_v = jnp.where(better, v, best_v)
            return (p, s, best_p, best_v), None

        v0 = jnlp(position)
        (p, _, best_p, best_v), _ = jax.lax.scan(
            adam_step, (position, opt.init(position), position, v0),
            None, length=n_adam,
        )
        vp = jnlp(p)
        take = (vp < 1e30) & (vp <= best_v)
        best_p = jnp.where(take, p, best_p)
        best_v = jnp.where(take, vp, best_v)
        return best_p, best_v

    def _lbfgs_polish(self, jlp, position, value, n_lbfgs_restarts=3):
        """Restarted single-point L-BFGS polish (fresh curvature memory each
        restart), keeping improvements. Returns (position, -logp)."""
        jnlp = jax.jit(lambda p: -jlp(p))
        vgrad = jax.value_and_grad(jnlp)
        solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)
        best_p, best_v = position, value
        for _ in range(n_lbfgs_restarts):
            res = solver.run(best_p)
            val = jnlp(res.params)
            if bool(val < 1e30) and bool(val < best_v - 0.1):
                best_p, best_v = res.params, val
            else:
                break
        return best_p, best_v

    def _hessian_mass_matrix(self, jlp, position):
        """Estimate diagonal inverse mass matrix from the Hessian of the log posterior.

        Uses exact forward-over-reverse Hessian-vector products (the RK4
        spectral-equivalence rewrite made the model forward-mode
        differentiable; finite differences produced clip-saturated entries
        in soft directions). Requires dim HVP evaluations.

        Elements are clamped to [1e-6, 1e6] to guard against degenerate
        curvature far from the MAP.

        Args:
            jlp: JIT-compiled log posterior function (scalar output).
            position: 1D JAX array of parameter values (normalized space).

        Returns:
            1D JAX array of shape (dim,) representing the diagonal
            inverse mass matrix.
        """
        grad_fn = jax.grad(lambda p: -jlp(p))
        dim = len(position)

        def hvp_diag(i):
            v = jnp.zeros(dim).at[i].set(1.0)
            return jax.jvp(grad_fn, (position,), (v,))[1][i]

        diag_H = jax.lax.map(hvp_diag, jnp.arange(dim))
        # Non-positive curvature (unconverged MAP / flat directions) would
        # clip to the floor and give that direction a ~1e6 inverse mass,
        # which collapses dual averaging (observed eps -> 4e-9). Use the
        # median positive curvature as a neutral scale instead.
        pos_med = jnp.nanmedian(jnp.where(diag_H > 0, diag_H, jnp.nan))
        diag_H = jnp.where(diag_H > 0, diag_H, pos_med)
        # Relative floor: near-zero positive curvature is as damaging as
        # negative (caps imm at 1e4 x the median imm).
        diag_H = jnp.maximum(diag_H, 1e-4 * pos_med)
        return 1.0 / jnp.clip(diag_H, 1e-6, 1e6)

    def _mclmc_mass_matrix(self, log_posterior, initial_positions, rng_key,
                           n_tune_steps=600, n_steps=1200, n_chains=8):
        """Estimate a diagonal inverse mass matrix from short MCLMC runs.

        A short single-chain blackjax MCLMC tune finds (L, eps); n_chains
        unadjusted MCLMC chains are then run at a conservative eps/2 from
        jittered starts. Chains with any non-finite position in the second
        half are dropped; the estimate is the mean over surviving chains of
        the within-chain variance over the second half of steps. Fallback
        ladder: retry once at eps/4, then _hessian_mass_matrix, then ones.
        Result is clamped to [1e-8, 1e8].

        Args:
            log_posterior: Log posterior function (scalar output).
            initial_positions: (n, dim) array of starting positions.
            rng_key: JAX random key.
            n_tune_steps: Steps for the single-chain L/eps tune.
            n_steps: Steps per estimation chain.
            n_chains: Number of estimation chains.

        Returns:
            Tuple of (imm, info_dict) where imm is a (dim,) diagonal inverse
            mass matrix and info_dict records n_survivors, eps, and rung.
        """
        import blackjax
        import blackjax.mcmc.integrators
        import blackjax.mcmc.mclmc

        jlp = jax.jit(log_posterior)
        dim = initial_positions.shape[1]
        rng_key, tune_key, init_key, jit_key = jax.random.split(rng_key, 4)

        # Rung 0: short single-chain tune for (L, eps).
        L, eps, tuned_imm = jnp.sqrt(dim), 0.01, jnp.ones(dim)
        try:
            tune_state = blackjax.mcmc.mclmc.init(
                position=initial_positions[0], logdensity_fn=jlp,
                rng_key=init_key,
            )
            kernel = lambda imm: blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=jlp,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=imm,
            )
            _, tp, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel,
                num_steps=n_tune_steps,
                state=tune_state,
                rng_key=tune_key,
                diagonal_preconditioning=True,
            )
            if (
                jnp.isfinite(tp.L)
                and jnp.isfinite(tp.step_size)
                and bool(jnp.all(jnp.isfinite(tp.inverse_mass_matrix)))
            ):
                L, eps, tuned_imm = tp.L, tp.step_size, jnp.maximum(
                    tp.inverse_mass_matrix,
                    1e-8 * jnp.max(tp.inverse_mass_matrix),
                )
            else:
                print(
                    "MCLMC mass-matrix tune produced non-finite params; "
                    "using conservative defaults.",
                    flush=True,
                )
        except Exception as e:
            print(
                f"MCLMC mass-matrix tune failed ({e}); using conservative "
                "defaults.",
                flush=True,
            )

        step_kernel = blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=jlp,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=tuned_imm,
        )

        idx = jnp.arange(n_chains) % initial_positions.shape[0]
        starts = initial_positions[idx]
        starts = starts + 0.01 * jax.random.normal(jit_key, starts.shape)

        def run_chain(pos, key, step_size):
            init_k, run_k = jax.random.split(key)
            state = blackjax.mcmc.mclmc.init(
                position=pos, logdensity_fn=jlp, rng_key=init_k,
            )

            def step(state, k):
                state, _ = step_kernel(k, state, L=L, step_size=step_size)
                return state, state.position

            _, positions = jax.lax.scan(
                step, state, jax.random.split(run_k, n_steps)
            )
            return positions

        def try_rung(step_size, key):
            keys = jax.random.split(key, n_chains)
            positions = jax.vmap(run_chain, in_axes=(0, 0, None))(
                starts, keys, step_size
            )  # (n_chains, n_steps, dim)
            second = positions[:, n_steps // 2:, :]
            finite = jnp.all(jnp.isfinite(second), axis=(1, 2))
            n_surv = int(jnp.sum(finite))
            if n_surv < max(n_chains // 2, 1):
                return None, n_surv
            var = jnp.var(second, axis=1)  # within-chain variance
            imm = jnp.mean(var[finite], axis=0)
            if not bool(jnp.all(jnp.isfinite(imm))):
                return None, n_surv
            # fp32 variance can go slightly negative for tight params
            imm = jnp.maximum(imm, 1e-8 * jnp.max(imm))
            return imm, n_surv

        for rung, step_size in (("eps/2", eps / 2), ("eps/4", eps / 4)):
            rng_key, run_key = jax.random.split(rng_key)
            imm, n_surv = try_rung(step_size, run_key)
            if imm is not None:
                info = {
                    "n_survivors": n_surv,
                    "eps": float(step_size),
                    "rung": rung,
                }
                return jnp.clip(imm, 1e-8, 1e8), info
            print(
                f"MCLMC mass-matrix estimation at {rung} "
                f"(eps={float(step_size):.3e}) failed: only {n_surv}/"
                f"{n_chains} chains finite.",
                flush=True,
            )

        print(
            "MCLMC mass-matrix estimation failed at all step sizes; "
            "falling back to Hessian mass matrix.",
            flush=True,
        )
        try:
            imm = self._hessian_mass_matrix(
                jlp, self._best_fit_position(jlp, initial_positions[0])
            )
            if bool(jnp.all(jnp.isfinite(imm))):
                return (
                    jnp.clip(imm, 1e-8, 1e8),
                    {"n_survivors": 0, "eps": float(eps / 4),
                     "rung": "hessian"},
                )
        except Exception as e:
            print(f"Hessian fallback failed ({e}).", flush=True)
        print(
            "Hessian mass-matrix fallback failed; using ones.", flush=True
        )
        return (
            jnp.ones(dim),
            {"n_survivors": 0, "eps": float(eps / 4), "rung": "ones"},
        )

    def _minimize_and_sample(
        self, log_posterior, initial_positions, n_chains, output_file,
        chi2_threshold=1,
    ):
        """Run L-BFGS from every chain's start, save the results, and move
        chains whose minimum is worse than chi2_threshold x the best one to
        the best position."""
        # minimize negative log posterior
        jnlp = jax.jit(lambda p: -log_posterior(p))
        vgrad = jax.value_and_grad(jnlp)
        solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

        print("Running minimization before sampling", flush=True)
        if bool(jnp.all(initial_positions == initial_positions[0])):
            # identical starts (random_start: false): minimize once and
            # broadcast instead of K concurrent L-BFGS grads (OOM at K=32).
            x_opt = self._best_fit_position(
                log_posterior, initial_positions[0]
            )
            initial_positions = np.tile(
                np.asarray(x_opt), (n_chains, 1)
            )
            values = np.full(n_chains, float(jnlp(x_opt)))
        else:
            minimize_map = self._map_chains(solver.run)
            res = minimize_map(initial_positions)
            initial_positions = gather_to_host(res.params)
            values = gather_to_host(res.state.value)
        if is_io_process():
            with open(f"{output_file}.minimization_results.json", "w") as fp:
                json.dump(
                    {
                        "x_opt": initial_positions.tolist(),
                        "value": values.tolist(),
                    },
                    fp,
                )

        chi2_ratio = values / np.min(values)
        initial_positions_min = jnp.tile(
            initial_positions[np.argmin(values)], n_chains
        ).reshape(n_chains, -1)
        initial_positions = jnp.where(
            chi2_ratio[:, None] > chi2_threshold,
            initial_positions_min,
            initial_positions,
        )
        return initial_positions

    def _make_pmap_inference_loop(self, collect_info=False):
        """Build the pmapped lax.scan inference loop.

        With collect_info=True each step also stacks the kernel's info
        (needed by MetropolisHastings for acceptance tracking). With
        chains_per_device > 1 the per-device loop is vmapped over K chains
        and the flat (n_devices*K,) chain axis of keys/states is split to
        (n_devices, K) around the pmap.
        """

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, info = kernel(rng_key, state)
                if collect_info == "stats":
                    # lightweight NUTS diagnostics: lockstep efficiency needs
                    # per-chain tree sizes, not the full info pytree
                    return state, (state,
                                   jnp.asarray(info.num_integration_steps),
                                   jnp.asarray(info.is_divergent))
                return state, ([state, info] if collect_info else state)

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        if self.chains_per_device == 1:
            return jax.pmap(
                inference_loop,
                in_axes=(0, None, 0, None),
                static_broadcasted_argnums=(1, 3),
            )

        K = self.chains_per_device

        def batched_loop(rng_keys, kernel, initial_states, num_samples):
            return jax.vmap(
                lambda k, s: inference_loop(k, kernel, s, num_samples)
            )(rng_keys, initial_states)

        ploop = jax.pmap(
            batched_loop,
            in_axes=(0, None, 0, None),
            static_broadcasted_argnums=(1, 3),
        )

        def loop(sample_keys, kernel, states, num_samples):
            nd = jax.local_device_count()
            keys = sample_keys.reshape(nd, K)
            states = jax.tree.map(
                lambda x: x.reshape((nd, K) + x.shape[1:]), states
            )
            out = ploop(keys, kernel, states, num_samples)
            return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), out)

        return loop

    def _make_mesh_inference_loop(self, kernel, num_samples, collect_info=False):
        """Mesh analog of _make_pmap_inference_loop.

        kernel and num_samples are closed over (they were static pmap args);
        the returned callable matches the pmap loop's call signature so
        _run_convergence_loop can drive either interchangeably.
        """

        def inference_loop(rng_key, initial_state):
            def one_step(state, k):
                state, info = kernel(k, state)
                return state, ([state, info] if collect_info else state)

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        mapped = self._chain_map(inference_loop, (0, 0))

        def loop(sample_keys, _kernel, states, _num_samples):
            return mapped(sample_keys, states)

        return loop

    def _save_checkpoint(self, output_file, samples, log_density, sigmas, reference):
        """Write physical-space samples and log posterior checkpoints."""
        np.save(
            f"{output_file}.samples_chk.npy",
            samples * sigmas[None, None, :] + reference[None, None, :],
        )
        np.save(f"{output_file}.logposterior_chk.npy", log_density)

    def _run_convergence_loop(
        self,
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
    ):
        """Run n_steps_incr batches until R-hat converges, checkpointing after
        each batch.

        Args:
            n_chains: Total chain count (n_devices * chains_per_device); the
                flat chain axis of sample_keys and gathered arrays.
            reinit_fn: Callable (states, rng_key) -> (states, rng_key) that
                re-initializes the kernel states from the last positions
                between batches (samplers differ in whether init needs keys).
        """
        print("Running inference loop", flush=True)
        rhat = 10000

        if samples is None:
            counter = 0
            n_steps = 0
        else:
            counter = 0
            n_steps = samples.shape[1]

        while (rhat - 1 > self.target_r_minus_one) | (n_steps < self.n_steps_min):
            if counter != 0:
                states, rng_key = reinit_fn(states, rng_key)
            states = pmap_inference_loop(
                sample_keys, kernel, states, self.n_steps_incr
            )
            # exact tuple only: kernel states are NamedTuples (tuple subclass)
            if type(states) is tuple:
                states, n_leap, divergent = states
                n_leap = np.asarray(gather_to_host(n_leap))
                divergent = np.asarray(gather_to_host(divergent))
                # each device's K vmapped chains lockstep to their max tree
                grp = n_leap.reshape(-1, self.chains_per_device,
                                     n_leap.shape[-1])
                eff = float(n_leap.mean() / grp.max(axis=1).mean())
                print(
                    f"leapfrogs/step: mean {n_leap.mean():.0f}, "
                    f"max {n_leap.max()}, lockstep efficiency {eff:.2f}, "
                    f"divergent {divergent.mean():.3f}",
                    flush=True,
                )

            # In multi-process runs each host holds only its shard of the
            # chain axis; gather so R-hat sees all chains and checkpoints are
            # complete (no-op device-to-host transfer otherwise).
            batch_positions = gather_to_host(states.position)
            batch_logdensity = gather_to_host(states.logdensity)
            if (counter == 0) & (n_steps == 0):
                samples = batch_positions
                log_density = batch_logdensity
            else:
                samples = np.hstack([samples, batch_positions])
                log_density = np.hstack([log_density, batch_logdensity])

            # Stopping-rule R-hat uses the latter half of the accumulated
            # samples to exclude the initial transient; checkpoints still
            # store everything.
            half = samples.shape[1] // 2
            rhat = jnp.mean(potential_scale_reduction(samples[:, half:]))

            print(f"n_samples = {samples.shape[1]}", flush=True)
            print(f"rhat - 1 (latter half) = {rhat - 1}", flush=True)

            counter += 1
            if is_io_process():
                self._save_checkpoint(
                    output_file, samples, log_density, sigmas, reference
                )
            n_steps = samples.shape[1]

            keys = jax.random.split(rng_key, 1 + n_chains)
            rng_key = keys[0]
            sample_keys = keys[1:]

        return samples, log_density

    def _finalize_samples(self, samples, log_density, sigmas, reference, param_names):
        """Rescale to physical space and append the log posterior column."""
        samples = samples * sigmas[None, None, :] + reference[None, None, :]
        samples = jnp.vstack([samples.T, log_density[..., None].T]).T
        param_names.append("log_posterior")
        return samples, param_names
