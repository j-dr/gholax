import json
from collections import namedtuple
from datetime import datetime

import jax
import jax.scipy.special
import jax.numpy as jnp
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

    def _warmup_host(self, prefix):
        """Bundle for the warmup engine (see gholax.sampler.warmup)."""
        from .warmup import WarmupHost

        return WarmupHost(
            chains_per_device=self.chains_per_device,
            param_names=getattr(self, "_param_names", None),
            sample_transform=bool(
                getattr(getattr(self, "_prior", None), "transform", False)
            ),
            mesh=self.mesh,
            prefix=prefix,
        )

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

    def _seeding_host(self, map_chains=None):
        """Bundle for the seeding engine (see gholax.sampler.seeding).

        Built lazily per call: _init_chains sets _prior/_param_names, and
        Minimize sets self.mesh inside run(), so this must not be cached at
        construction time.
        """
        from .seeding import SeedingHost

        return SeedingHost(
            map_chains=self._map_chains if map_chains is None else map_chains,
            mesh=self.mesh,
            chains_per_device=self.chains_per_device,
            param_names=getattr(self, "_param_names", None),
            constrain=getattr(getattr(self, "_prior", None), "constrain", None),
        )

    def _seeder(self):
        """Seeder wired to this sampler's config, host, and fisher params."""
        from .seeding import FisherParams, Seeder, SeedingConfig

        cfg = getattr(self, "seeding_config", None) or SeedingConfig()
        wc = getattr(self, "warmup_config", None)
        fisher = None
        if wc is not None:
            fisher = FisherParams(
                wc.dense_rank or "auto", wc.fisher_cutoff, wc.fisher_reg
            )
        seeder = Seeder(cfg, self._seeding_host(), fisher_params=fisher)
        seeder.pathfinder_imm = getattr(self, "_pathfinder_imm", None)
        return seeder

    # The seeding stage lives in gholax.sampler.seeding.Seeder.  These
    # delegating shims keep the historical call surface (and the two
    # attributes callers read back) working.

    def _best_fit_position(self, jlp, position, n_adam=None,
                           n_lbfgs_restarts=None, n_starts=None,
                           start_scale=None, rng_key=None):
        return self._seeder().best_fit_position(
            jlp, position, n_adam, n_lbfgs_restarts, n_starts, start_scale,
            rng_key,
        )

    def _adam_multistart(self, jlp, position, n_adam=None, n_starts=None,
                         start_scale=None, rng_key=None):
        return self._seeder().adam_multistart(
            jlp, position, n_adam, n_starts, start_scale, rng_key
        )

    def _jittered_starts(self, jlp, position, n_starts, start_scale, rng_key):
        return self._seeder().jittered_starts(
            jlp, position, n_starts, start_scale, rng_key
        )

    def _adam_descent(self, jlp, position, n_adam=None):
        return self._seeder().adam_descent(jlp, position, n_adam)

    def _lbfgs_polish(self, jlp, position, value, n_lbfgs_restarts=None):
        return self._seeder().lbfgs_polish(
            jlp, position, value, n_lbfgs_restarts
        )

    def _pathfinder_init(self, jlp, x_map, n_chains, rng_key, output_file):
        seeder = self._seeder()
        positions = seeder.pathfinder_init(
            jlp, x_map, n_chains, rng_key, output_file
        )
        self._pathfinder_imm = seeder.pathfinder_imm
        return positions

    def _polish_and_pathfinder(self, log_posterior, position, n_chains,
                               pathfinder_key, output_file):
        seeder = self._seeder()
        x_opt = seeder.polish_and_pathfinder(
            log_posterior, position, n_chains, pathfinder_key, output_file
        )
        self._pathfinder_imm = seeder.pathfinder_imm
        self._pathfinder_positions = seeder.pathfinder_positions
        return x_opt

    def _hessian_mass_matrix(self, jlp, position):
        return self._seeder().hessian_mass_matrix(jlp, position)

    def _hessian_mass_matrix_dense(self, jlp, position):
        return self._seeder().hessian_mass_matrix_dense(jlp, position)

    def _mclmc_mass_matrix(self, log_posterior, initial_positions, rng_key,
                           n_tune_steps=None, n_steps=None, n_chains=None):
        return self._seeder().mclmc_mass_matrix(
            log_posterior, initial_positions, rng_key, n_tune_steps, n_steps,
            n_chains,
        )

    def _minimize_and_sample(
        self, log_posterior, initial_positions, n_chains, output_file,
        chi2_threshold=1, pathfinder_key=None,
    ):
        seeder = self._seeder()
        positions = seeder.minimize(
            log_posterior, initial_positions, n_chains, output_file,
            chi2_threshold=chi2_threshold, pathfinder_key=pathfinder_key,
        )
        self._pathfinder_imm = seeder.pathfinder_imm
        self._pathfinder_positions = seeder.pathfinder_positions
        return positions


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
