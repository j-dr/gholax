import json
from collections import namedtuple
from datetime import datetime

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from blackjax.diagnostics import potential_scale_reduction

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
    """

    def _init_chains(self, model, jit_logpost=True):
        """Seed the rng, extract prior scaling, and draw per-device initial
        positions in normalized parameter space."""
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
            self._save_checkpoint(output_file, samples, log_density, sigmas, reference)
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
