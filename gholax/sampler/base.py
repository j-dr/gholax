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
    """

    mesh = None

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
            model-axis collectives inside the sharded posterior).
        """
        from jax.sharding import PartitionSpec as P

        in_specs = tuple(P(CHAIN_AXIS) if a == 0 else P() for a in chain_axes)

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
        """Map fn over chains: pmap in legacy mode, _chain_map in mesh mode."""
        if self.mesh is None:
            return jax.pmap(
                fn, in_axes=tuple(0 if a == 0 else None for a in chain_axes)
            )
        return self._chain_map(fn, chain_axes)

    def _hessian_mass_matrix(self, jlp, position):
        """Estimate diagonal inverse mass matrix from the Hessian of the log posterior.

        Uses forward finite differences of the gradient to estimate the diagonal
        of the Hessian using only reverse-mode AD. Forward-mode (JVP) cannot be
        used because odeint defines a custom_vjp without a matching custom_jvp.
        Requires dim+1 gradient evaluations.

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
        eps = 1e-3
        dim = len(position)
        g0 = grad_fn(position)
        # Sequential loop: one gradient eval per parameter, keeping only the
        # i-th element each time to avoid allocating dim gradient arrays at once.
        diag_H = jnp.array([
            (grad_fn(position.at[i].set(position[i] + eps))[i] - g0[i]) / eps
            for i in range(dim)
        ])
        return 1.0 / jnp.clip(diag_H, 1e-6, 1e6)

    def _minimize_and_sample(
        self, log_posterior, initial_positions, n_devices, output_file,
        chi2_threshold=1,
    ):
        """Run L-BFGS from every chain's start, save the results, and move
        chains whose minimum is worse than chi2_threshold x the best one to
        the best position."""
        # minimize negative log posterior
        jnlp = jax.jit(lambda p: -log_posterior(p))
        vgrad = jax.value_and_grad(jnlp)
        solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

        minimize_map = self._map_chains(solver.run)
        print("Running minimization before sampling", flush=True)
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
            initial_positions[np.argmin(values)], n_devices
        ).reshape(n_devices, -1)
        initial_positions = jnp.where(
            chi2_ratio[:, None] > chi2_threshold,
            initial_positions_min,
            initial_positions,
        )
        return initial_positions

    def _make_pmap_inference_loop(self, collect_info=False):
        """Build the pmapped lax.scan inference loop.

        With collect_info=True each step also stacks the kernel's info
        (needed by MetropolisHastings for acceptance tracking).
        """

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, info = kernel(rng_key, state)
                return state, ([state, info] if collect_info else state)

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        return jax.pmap(
            inference_loop,
            in_axes=(0, None, 0, None),
            static_broadcasted_argnums=(1, 3),
        )

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
        n_devices,
        output_file,
        reinit_fn,
    ):
        """Run n_steps_incr batches until R-hat converges, checkpointing after
        each batch.

        Args:
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
            if counter == 0:
                states = pmap_inference_loop(
                    sample_keys, kernel, states, self.n_steps_incr
                )
            else:
                states, rng_key = reinit_fn(states, rng_key)
                states = pmap_inference_loop(
                    sample_keys, kernel, states, self.n_steps_incr
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

            rhat = jnp.mean(potential_scale_reduction(samples))

            print(f"n_samples = {samples.shape[1]}", flush=True)
            print(f"rhat - 1 = {rhat - 1}", flush=True)

            counter += 1
            if is_io_process():
                self._save_checkpoint(
                    output_file, samples, log_density, sigmas, reference
                )
            n_steps = samples.shape[1]

            keys = jax.random.split(rng_key, 1 + n_devices)
            rng_key = keys[0]
            sample_keys = keys[1:]

        return samples, log_density

    def _finalize_samples(self, samples, log_density, sigmas, reference, param_names):
        """Rescale to physical space and append the log posterior column."""
        samples = samples * sigmas[None, None, :] + reference[None, None, :]
        samples = jnp.vstack([samples.T, log_density[..., None].T]).T
        param_names.append("log_posterior")
        return samples, param_names
