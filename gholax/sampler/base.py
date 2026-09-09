import json
import threading
from collections import namedtuple
from datetime import datetime

import jax
import jax.scipy.linalg
import jax.scipy.special
import jax.numpy as jnp
import jaxopt
import numpy as np
from blackjax.diagnostics import (
    effective_sample_size,
    potential_scale_reduction,
)

from ..util.distributed import CHAIN_AXIS, gather_to_host, is_io_process


def _rank_normalize(samples):
    """Rank-normalize (chains, draws, dim) samples per parameter over the
    pooled draws (Vehtari et al. 2021 bulk statistics): ranks -> normal scores
    via Phi^-1((r - 3/8) / (N + 1/4)).  Invariant under monotone transforms,
    so ESS/R-hat agree between sampling and physical space."""
    n_chains, n_draws, dim = samples.shape
    flat = samples.reshape(-1, dim)
    ranks = jnp.argsort(jnp.argsort(flat, axis=0), axis=0) + 1
    z = jax.scipy.special.ndtri((ranks - 0.375) / (flat.shape[0] + 0.25))
    return z.reshape(n_chains, n_draws, dim)

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
        self._param_names = list(param_names)
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

        # cached for sampling<->physical conversions (constrain/unconstrain)
        self._prior = prior
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
        best = self._adam_multistart(jlp, position, n_adam, n_starts,
                                     start_scale, rng_key)
        if best is None:
            return position
        best_p, best_v = self._lbfgs_polish(jlp, *best, n_lbfgs_restarts)
        print(f"  best fit logp: {float(-best_v):.2f}", flush=True)
        return best_p

    def _adam_multistart(self, jlp, position, n_adam=300, n_starts=None,
                         start_scale=None, rng_key=None):
        """Vmapped multi-start Adam; returns the (position, -logp) of the
        best start, or None when every start failed."""
        if n_starts is None:
            n_starts = getattr(self, "minimize_n_starts", 4)
        if start_scale is None:
            start_scale = getattr(self, "minimize_start_scale", 0.5)
        if rng_key is None:
            rng_key = jax.random.key(0)
        starts = self._jittered_starts(
            jlp, position, n_starts, start_scale, rng_key
        )
        ps, vs = jax.vmap(
            lambda x0: self._adam_descent(jlp, x0, n_adam)
        )(starts)
        vs = jnp.where(vs < 1e30, vs, jnp.inf)
        i = int(jnp.argmin(vs))
        if not bool(jnp.isfinite(vs[i])):
            print("  minimization failed; using initial position", flush=True)
            return None
        spread = (float(vs.max() - vs.min())
                  if bool(jnp.isfinite(vs).all()) else float("nan"))
        print(f"  adam multi-start: best -logp {float(vs[i]):.2f} "
              f"({n_starts} starts, spread {spread:.2f})", flush=True)
        print("  per-start -logp (start 0 = reference): "
              + ", ".join(f"{float(v):.1f}" for v in vs), flush=True)
        return ps[i], vs[i]

    def _jittered_starts(self, jlp, position, n_starts, start_scale, rng_key):
        """position plus n_starts-1 Hessian-guided jittered copies.

        Per-coordinate scale ~ sqrt(imm) at position concentrates the
        displacement in soft/degenerate directions (where distinct basins
        live) and barely moves stiff ones; the scale is backed off until the
        candidate is in-box.
        """
        jn = jax.jit(lambda p: -jlp(p))
        dim = position.shape[0]
        w = jnp.sqrt(self._hessian_mass_matrix(jlp, position))
        w = w / jnp.max(w)
        starts = [position]
        for i in range(n_starts - 1):
            ki = jax.random.fold_in(rng_key, i)
            for scale in (start_scale, start_scale / 5, 0.0):
                cand = position + scale * w * jax.random.normal(ki, (dim,))
                if bool(jn(cand) < 1e30):
                    break
            starts.append(cand)
        return jnp.stack(starts)

    def _pathfinder_init(self, jlp, x_map, n_chains, rng_key, output_file):
        """Multi-path Pathfinder chain seeding (Zhang et al. 2022).

        L-BFGS paths start from x_map and Hessian-jittered copies.  Along
        each path the Gaussian N(theta_l - Sigma_l grad_l, Sigma_l), with
        Sigma_l the L-BFGS inverse-Hessian estimate, is scored by a Monte
        Carlo ELBO and the best iterate kept (done here rather than with
        blackjax's own scan, whose mean has the opposite sign).  Draws are
        pooled and either importance-resampled with mixture-proposal
        weights (pathfinder_resample) or taken as ELBO-weighted mixture
        draws.
        x_map itself is untouched: it stays the MAP for model comparison
        and the Hessian metric.
        """
        from blackjax.vi import pathfinder

        n_paths = getattr(self, "pathfinder_n_paths", 4)
        n_elbo = getattr(self, "pathfinder_elbo_samples", 20)
        resample = getattr(self, "pathfinder_resample", True)
        dim = x_map.shape[0]
        k_start, k_pf, k_draw, k_pick = jax.random.split(rng_key, 4)
        starts = self._jittered_starts(
            jlp, x_map, n_paths,
            getattr(self, "pathfinder_start_scale", 0.5), k_start,
        )
        print(f"Running Pathfinder ({n_paths} paths, resample={resample})",
              flush=True)
        vlp = jax.jit(jax.vmap(jlp))

        def gaussian(pos, grad, alpha, beta, gamma):
            sig = jnp.diag(alpha) + beta @ gamma @ beta.T
            return pos - sig @ grad, jnp.linalg.cholesky(sig)

        def logq(mu, L, x):
            z = jax.scipy.linalg.solve_triangular(L, (x - mu).T, lower=True)
            return (-0.5 * jnp.sum(z**2, axis=0)
                    - jnp.sum(jnp.log(jnp.diag(L)))
                    - 0.5 * dim * jnp.log(2 * jnp.pi))

        @jax.jit
        def elbo_path(key, path):
            def one(k, pos, grad, alpha, beta, gamma):
                mu, L = gaussian(pos, grad, alpha, beta, gamma)
                x = mu + (L @ jax.random.normal(k, (dim, n_elbo))).T
                lp = vlp(x)
                e = jnp.mean(lp - logq(mu, L, x))
                ok = jnp.all(jnp.isfinite(L)) & jnp.all(lp > -1e30)
                return jnp.where(ok, e, -jnp.inf)
            keys = jax.random.split(key, path.position.shape[0])
            # sequential to avoid OOM
            
            return jax.lax.map(
                lambda a: one(*a),
                (keys, path.position, path.grad_position,
                 path.alpha, path.beta, path.gamma),
            )

        approx = jax.jit(lambda k, x0: pathfinder.approximate(
            k, jlp, x0, num_samples=2,
            maxiter=getattr(self, "pathfinder_maxiter", 100),
            maxcor=getattr(self, "pathfinder_maxcor", 10),
        )[1].path)
        gauss, elbos, n_iter = [], [], []
        per_path = 2 * n_chains // n_paths + 1
        for k, x0 in zip(jax.random.split(k_pf, n_paths), starts):
            k1, k2 = jax.random.split(k)
            path = approx(k1, x0)
            e = elbo_path(k2, path)
            l = int(jnp.argmax(e))
            elbos.append(float(e[l])); n_iter.append(l)
            gauss.append(gaussian(path.position[l], path.grad_position[l],
                                  path.alpha[l], path.beta[l], path.gamma[l]))
        print("  per-path best ELBO: " + ", ".join(
            f"{e:.1f} (iter {l})" for e, l in zip(elbos, n_iter)), flush=True)
        phi = jnp.concatenate([
            mu + (L @ jax.random.normal(k, (dim, per_path))).T
            for (mu, L), k in zip(gauss, jax.random.split(k_draw, n_paths))
        ])
        logp = vlp(phi)
        ok = jnp.isfinite(logp) & (logp > -1e30)
        print(f"  finite draws {int(ok.sum())}/{phi.shape[0]}", flush=True)
        phi, logp = phi[ok], logp[ok]
        info = {"elbo": elbos, "best_iter": n_iter, "n_finite": int(ok.sum()),
                "resample": resample}
        # Plain-mixture mode weights paths by exp(ELBO)
        # importance-sampling mode instead uses the uniform mixture the
        # draws actually came from and lets the weights do the work.
        log_pm = jnp.asarray(elbos) - jax.scipy.special.logsumexp(
            jnp.asarray(elbos))
        info["path_weights"] = np.exp(np.asarray(log_pm)).tolist()
        path_of = jnp.repeat(jnp.arange(n_paths), per_path)[ok]
        w = jnp.exp(log_pm)[path_of]
        w = w / w.sum()
        if resample:
            lq = jax.scipy.special.logsumexp(
                jnp.stack([logq(mu, L, phi) for mu, L in gauss]), axis=0
            ) - jnp.log(n_paths)
            logw = logp - lq
            w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
            n_eff = float(1.0 / jnp.sum(w**2))
            info["weight_ess"] = n_eff
            print(f"  importance-weight ESS {n_eff:.1f} of {phi.shape[0]} "
                  f"draws", flush=True)
            if n_eff < 2:
                print("  weights degenerate; falling back to mixture draws",
                      flush=True)
                w = jnp.exp(log_pm)[path_of]
                w = w / w.sum()
        idx = jax.random.choice(
            k_pick, phi.shape[0], (n_chains,),
            replace=phi.shape[0] < n_chains, p=w,
        )
        positions = phi[idx]
        info["n_unique"] = int(jnp.unique(idx).shape[0])
        # ELBO-best path's L-BFGS covariance: bulk-scale dense metric
        # candidate (mass_matrix_init: pathfinder)
        Lb = gauss[int(jnp.argmax(jnp.asarray(elbos)))][1]
        self._pathfinder_imm = Lb @ Lb.T
        print(f"  seeded {n_chains} chains from {info['n_unique']} distinct "
              f"draws; logp range [{float(logp[idx].min()):.1f}, "
              f"{float(logp[idx].max()):.1f}]", flush=True)
        if is_io_process():
            with open(f"{output_file}.pathfinder_init.json", "w") as fp:
                json.dump({**info, "positions": positions.tolist(),
                           "inverse_mass_matrix":
                               np.asarray(self._pathfinder_imm).tolist()}, fp)
        return positions

    def _polish_and_pathfinder(self, log_posterior, position, n_chains,
                               pathfinder_key, output_file):
        """Adam multi-start, then L-BFGS polish (thread, spare device) in
        parallel with Pathfinder seeding from the unpolished winner.
        Returns the polished MAP; seeds go to self._pathfinder_positions."""
        jlp = jax.jit(log_posterior)
        best = self._adam_multistart(jlp, position)
        if best is None:
            self._pathfinder_positions = self._pathfinder_init(
                jlp, position, n_chains, pathfinder_key, output_file
            )
            return position
        p0, v0 = best
        devs = jax.local_devices()
        pol_dev = devs[1] if len(devs) > 1 else devs[0]
        out = {}

        def polish():
            with jax.default_device(pol_dev):
                pp = jax.device_put(p0, pol_dev)
                out["x"], out["v"] = self._lbfgs_polish(jlp, pp, v0)

        print(f"  L-BFGS polish on {pol_dev} in parallel with Pathfinder",
              flush=True)
        t = threading.Thread(target=polish, daemon=True)
        t.start()
        self._pathfinder_positions = self._pathfinder_init(
            jlp, p0, n_chains, pathfinder_key, output_file
        )
        t.join()
        x_opt = jax.device_put(out["x"], devs[0])
        print(f"  best fit logp: {float(-out['v']):.2f}", flush=True)
        return x_opt

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
            p_candidate = optax.apply_updates(p, updates)
            # Evaluate the candidate at the position being recorded.  The
            # previous implementation compared ``v = f(p)`` but stored
            # ``p_new``; on a non-monotone Adam step this paired the old
            # objective with a different position and could select a point
            # that was not actually the best one seen.
            v_candidate = jnlp(p_candidate)
            valid = (
                jnp.isfinite(v_candidate)
                & (v_candidate < 1e30)
                & (v_candidate > -1e30)
            )
            p = jnp.where(valid, p_candidate, p)
            v = jnp.where(valid, v_candidate, v)
            better = valid & (v < best_v)
            best_p = jnp.where(better, p, best_p)
            best_v = jnp.where(better, v, best_v)
            return (p, s, best_p, best_v), None

        v0 = jnlp(position)
        v0_valid = (
            jnp.isfinite(v0) & (v0 < 1e30) & (v0 > -1e30)
        )
        v0 = jnp.where(v0_valid, v0, jnp.inf)
        (p, _, best_p, best_v), _ = jax.lax.scan(
            adam_step, (position, opt.init(position), position, v0),
            None, length=n_adam,
        )
        vp = jnlp(p)
        take = (
            jnp.isfinite(vp)
            & (vp < 1e30)
            & (vp > -1e30)
            & (vp <= best_v)
        )
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
        positive = jnp.isfinite(diag_H) & (diag_H > 0)
        pos_med = jnp.nanmedian(jnp.where(positive, diag_H, jnp.nan))
        # ``nanmedian`` is NaN when the whole diagonal is non-positive or
        # non-finite.  Such a Hessian is possible before the MAP polish (and
        # for genuinely flat targets); use the identity metric in that case
        # so warmup receives finite parameters instead of poisoning dual
        # averaging with NaNs.
        pos_med = jnp.where(
            jnp.isfinite(pos_med) & (pos_med > 0), pos_med, jnp.asarray(1.0)
        )
        diag_H = jnp.where(positive, diag_H, pos_med)
        # Relative floor: near-zero positive curvature is as damaging as
        # negative (caps imm at 1e4 x the median imm).
        diag_H = jnp.maximum(diag_H, 1e-4 * pos_med)
        return 1.0 / jnp.clip(diag_H, 1e-6, 1e6)

    def _hessian_mass_matrix_dense(self, jlp, position):
        """Dense inverse mass matrix from the full Hessian at `position`.

        Same HVP count as the diagonal estimate (one per coordinate), but
        keeps the full matrix: the inverse Hessian approximates the local
        posterior covariance, so a dense metric absorbs the correlations a
        diagonal metric cannot. The eigenvalue analog of the diagonal median
        guard handles indefiniteness: non-positive eigenvalues are replaced
        by the median positive one, with a relative floor of 1e-4 x median.

        Returns:
            (dim, dim) symmetric positive-definite inverse mass matrix.
        """
        grad_fn = jax.grad(lambda p: -jlp(p))
        dim = len(position)

        def hvp_row(i):
            v = jnp.zeros(dim).at[i].set(1.0)
            return jax.jvp(grad_fn, (position,), (v,))[1]

        H = jax.lax.map(hvp_row, jnp.arange(dim))
        H = 0.5 * (H + H.T)
        lam, V = jnp.linalg.eigh(H)
        positive = jnp.isfinite(lam) & (lam > 0)
        pos_med = jnp.nanmedian(jnp.where(positive, lam, jnp.nan))
        pos_med = jnp.where(
            jnp.isfinite(pos_med) & (pos_med > 0), pos_med, jnp.asarray(1.0)
        )
        lam = jnp.where(positive, lam, pos_med)
        lam = jnp.clip(jnp.maximum(lam, 1e-4 * pos_med), 1e-6, 1e6)
        return (V * (1.0 / lam)) @ V.T

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
        chi2_threshold=1, pathfinder_key=None,
    ):
        """Run L-BFGS from every chain's start, save the results, and move
        chains whose minimum is worse than chi2_threshold x the best one to
        the best position.

        With pathfinder_key (identical starts only) the L-BFGS polish of the
        Adam winner runs in a thread on a spare device while Pathfinder seeds
        the chains from the unpolished winner; the seeds are left in
        self._pathfinder_positions and the polished MAP is still saved."""
        # minimize negative log posterior
        jnlp = jax.jit(lambda p: -log_posterior(p))
        vgrad = jax.value_and_grad(jnlp)
        solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

        print("Running minimization before sampling", flush=True)
        self._pathfinder_positions = None
        if bool(jnp.all(initial_positions == initial_positions[0])):
            # identical starts (random_start: false): minimize once and
            # broadcast instead of K concurrent L-BFGS grads (OOM at K=32).
            if pathfinder_key is not None:
                x_opt = self._polish_and_pathfinder(
                    log_posterior, initial_positions[0], n_chains,
                    pathfinder_key, output_file,
                )
            else:
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
                        # physical-space copy: x_opt is sampling-space, whose
                        # meaning depends on the prior transform
                        **(
                            {
                                "x_opt_physical": np.asarray(
                                    self._prior.constrain(
                                        jnp.asarray(initial_positions)
                                    )
                                ).tolist()
                            }
                            if hasattr(self._prior, "constrain")
                            else {}
                        ),
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

    def _to_physical(self, samples, sigmas, reference):
        """Sampling-space -> physical, honoring the prior transform if set."""
        prior = getattr(self, "_prior", None)
        if prior is not None and hasattr(prior, "constrain"):
            return np.asarray(prior.constrain(jnp.asarray(samples)))
        return samples * sigmas[None, None, :] + reference[None, None, :]

    def _from_physical(self, samples, sigmas, reference):
        """Physical -> sampling-space, honoring the prior transform if set."""
        prior = getattr(self, "_prior", None)
        if prior is not None and hasattr(prior, "unconstrain"):
            return np.asarray(prior.unconstrain(jnp.asarray(samples)))
        return (samples - reference) / sigmas

    def _save_checkpoint(self, output_file, samples, log_density, sigmas, reference):
        """Write physical-space samples and log posterior checkpoints."""
        np.save(
            f"{output_file}.samples_chk.npy",
            self._to_physical(samples, sigmas, reference),
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
        max_divergence_rate=None,
        fail_on_divergence=False,
        divergence_check_min_steps=0,
    ):
        """Run n_steps_incr batches until R-hat converges, checkpointing after
        each batch.

        Args:
            n_chains: Total chain count (n_devices * chains_per_device); the
                flat chain axis of sample_keys and gathered arrays.
            reinit_fn: Callable (states, rng_key) -> (states, rng_key) that
                re-initializes the kernel states from the last positions
                between batches (samplers differ in whether init needs keys).
            max_divergence_rate: Optional maximum allowed cumulative fraction
                of divergent transitions.  ``None`` disables enforcement.
            fail_on_divergence: If true, raise when the configured divergence
                threshold is exceeded.  With no explicit threshold, any
                divergence fails the run.  The default is false for backwards
                compatibility; rates are still recorded when diagnostics are
                collected.
            divergence_check_min_steps: Do not enforce a rate until this many
                transitions have been observed.
        """
        print("Running inference loop", flush=True)
        rhat = 10000
        divergent_total = 0
        divergent_hist = None   # (n_chains, n_new_samples) flags this run
        transition_total = 0
        divergence_rates = []
        self.last_divergence_rate = None
        self.divergence_rates = divergence_rates

        if samples is None:
            counter = 0
            n_steps = 0
        else:
            counter = 0
            n_steps = samples.shape[1]

        # Optional worst-dimension ESS floor, ANDed with R-hat: ESS from
        # unmixed chains is overestimated, so it never relaxes the R-hat gate.
        target_ess = getattr(self, "target_min_ess", None)
        min_ess = 0.0

        while (
            (rhat - 1 > self.target_r_minus_one)
            or (target_ess is not None and min_ess < target_ess)
            or (n_steps < self.n_steps_min)
        ):
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
                dflags = np.asarray(divergent, dtype=bool).reshape(
                    -1, divergent.shape[-1]
                )
                divergent_hist = (
                    dflags if divergent_hist is None
                    else np.concatenate([divergent_hist, dflags], axis=1)
                )
                batch_divergent = int(np.asarray(divergent, dtype=bool).sum())
                batch_transitions = int(np.asarray(divergent).size)
                divergent_total += batch_divergent
                transition_total += batch_transitions
                divergence_rate = (
                    float(divergent_total / transition_total)
                    if transition_total
                    else 0.0
                )
                divergence_rates.append(divergence_rate)
                self.last_divergence_rate = divergence_rate
                # each device's K vmapped chains lockstep to their max tree
                grp = n_leap.reshape(-1, self.chains_per_device,
                                     n_leap.shape[-1])
                eff = float(n_leap.mean() / grp.max(axis=1).mean())
                print(
                    f"leapfrogs/step: mean {n_leap.mean():.0f}, "
                    f"max {n_leap.max()}, lockstep efficiency {eff:.2f}, "
                    f"divergent {divergent.mean():.3f} "
                    f"(cumulative {divergence_rate:.3f})"
                    + (
                        f", {int((dflags.mean(1) > 0.5).sum())} chains "
                        f">50% divergent"
                        if dflags.any() else ""
                    ),
                    flush=True,
                )
                threshold = max_divergence_rate
                if fail_on_divergence and threshold is None:
                    threshold = 0.0
                if (
                    threshold is not None
                    and transition_total >= divergence_check_min_steps
                    and divergence_rate > float(threshold)
                ):
                    message = (
                        "NUTS divergence rate "
                        f"{divergence_rate:.3f} exceeds configured maximum "
                        f"{float(threshold):.3f}"
                    )
                    if fail_on_divergence:
                        raise RuntimeError(message)
                    print(f"WARNING: {message}", flush=True)

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
            # A mean can hide a small set of unconverged coordinates in a
            # high-dimensional posterior; stop only when the worst ordinary
            # PSR is below the configured threshold.  Diagnostics use
            # rank-normalized (bulk) samples so a logit-stretched box edge
            # does not depress them.
            zs = _rank_normalize(jnp.asarray(samples))
            rhat = jnp.max(potential_scale_reduction(zs[:, half:]))

            print(f"n_samples = {samples.shape[1]}", flush=True)
            print(f"max rhat - 1 (latter half, bulk) = {rhat - 1}", flush=True)
            rhat_full = jnp.max(potential_scale_reduction(zs))
            print(
                f"max rhat - 1 (full chain, bulk) = {rhat_full - 1}", flush=True
            )
            if target_ess is not None:
                # Chodera-style ESS-maximizing burn-in: a transient inflates
                # the autocorrelation of any window containing it, so the
                # argmax over discard fractions lands just past it. The ESS
                # stopping statistic uses the best window (largest defensible
                # sample); R-hat stays on the conservative latter half, whose
                # job is detecting the transient.
                fracs = (0.0, 0.125, 0.25, 0.375, 0.5)
                ess_by_frac = {
                    f: float(
                        jnp.min(
                            effective_sample_size(
                                zs[:, int(f * samples.shape[1]):]
                            )
                        )
                    )
                    for f in fracs
                }
                best_frac = max(ess_by_frac, key=ess_by_frac.get)
                min_ess = ess_by_frac[best_frac]
                print(
                    f"min bulk ESS = {min_ess:.0f} at burn-in fraction "
                    f"{best_frac:g} (target {target_ess:.0f}; "
                    f"full-chain {ess_by_frac[0.0]:.0f}, "
                    f"latter-half {ess_by_frac[0.5]:.0f})",
                    flush=True,
                )

            counter += 1
            if is_io_process():
                self._save_checkpoint(
                    output_file, samples, log_density, sigmas, reference
                )
                if divergent_hist is not None:
                    # per-chain, per-transition flags for the samples drawn
                    # in this run (restart-local), to localize divergences
                    np.save(f"{output_file}.divergent_chk.npy", divergent_hist)
            n_steps = samples.shape[1]

            keys = jax.random.split(rng_key, 1 + n_chains)
            rng_key = keys[0]
            sample_keys = keys[1:]

        return samples, log_density

    def _finalize_samples(self, samples, log_density, sigmas, reference, param_names):
        """Rescale to physical space and append the log posterior column."""
        samples = jnp.asarray(self._to_physical(samples, sigmas, reference))
        samples = jnp.vstack([samples.T, log_density[..., None].T]).T
        param_names.append("log_posterior")
        return samples, param_names
