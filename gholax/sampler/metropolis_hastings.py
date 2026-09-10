import os
from dataclasses import replace

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.diagnostics import potential_scale_reduction

from .base import BaseSampler
from .seeding import (
    SeedingConfig,
    proposal_covariance_from_metric,
)


def choleskyL_corr(M):
    r"""
    Gets the Cholesky lower triangular matrix :math:`L` (defined as :math:`M=LL^T`)
    for the matrix ``M``, in the form :math:`L = S L^\prime` where S is diagonal.

    Can be used to create an affine transformation that decorrelates a sample
    :math:`x=\{x_i\}` with covariance M, as :math:`x=Ly`,
    where :math:`L` is extracted from M and y has identity covariance.

    Returns a tuple of a matrix :math:`S` containing the square roots of the diagonal
    of the input matrix (the standard deviations, if a covariance is given),
    and the scale-free :math:`L^\prime=S^{-1}L`.
    (could just use Cholesky directly for proposal)
    """
    std_diag, corr = cov_to_std_and_corr(M)
    return jnp.diag(std_diag), jnp.linalg.cholesky(corr)


def cov_to_std_and_corr(cov):
    """
    Gets the standard deviations (as a 1D array
    and the correlation matrix of a covariance matrix.
    """
    std = jnp.sqrt(jnp.diag(cov))
    inv_std = 1 / std
    corr = inv_std[:, np.newaxis] * cov * inv_std[np.newaxis, :]
    corr = jnp.fill_diagonal(corr, 1.0, inplace=False)
    return std, corr


def get_cosmomc_proposal_generator(proposal_cov):
    """
    Create a CosmoMC-style proposal generator for MCMC sampling.
    
    Args:
        proposal_cov: Covariance matrix for the proposal distribution
    
    Returns:
        A proposal generator function compatible with blackjax
    """
    sigmas, L = choleskyL_corr(proposal_cov)
    L = jnp.dot(sigmas, L)
    n = 2
    f = 2 / 3

    def cosmomc_proposal_cdf(r, n, f):
        return f * (1 - jax.scipy.special.gammaincc(n / 2, n * r**2 / 2)) + (1 - f) * (
            1 - jnp.exp(-r)
        )

    r_grid = np.linspace(0, 10, 1000)
    r_prop_cdf = cosmomc_proposal_cdf(np.linspace(0, 10, 1000), n, f)
    # sgn_grid = jnp.array([-1,1])

    def proposal_generator_(rng_key, position, L, r_grid, r_prop_cdf):
        r_key, rot_key, sgn_key = jax.random.split(rng_key, 3)
        n = L.shape[0]
        X = jax.random.normal(rot_key, (n, n))
        Q, _ = jnp.linalg.qr(X, mode="complete")
        r_n = jax.random.uniform(r_key)
        r = jnp.interp(r_n, r_prop_cdf, r_grid)
        sgn = jax.random.choice(sgn_key, jnp.array([-1, 1]))
        delta = jnp.dot(L, Q[:, 0] * r * sgn * 2.4)

        return delta + position

    proposal_generator = lambda key, state: proposal_generator_(
        key, state, L, r_grid, r_prop_cdf
    )
    return proposal_generator


def get_gaussian_proposal_generator(proposal_cov):
    """
    Create a Gaussian proposal generator for MCMC sampling.
    
    Args:
        proposal_cov: Covariance matrix for the Gaussian proposal distribution
    
    Returns:
        A proposal generator function compatible with blackjax
    """
    L = jnp.linalg.cholesky(proposal_cov)

    def proposal_generator_(rng_key, position, L):
        n = L.shape[0]
        x = jax.random.normal(rng_key, (n,))
        delta = jnp.dot(L, x)

        return delta + position

    proposal_generator = lambda key, state: proposal_generator_(key, state, L)
    return proposal_generator


class MetropolisHastings(BaseSampler):
    def __init__(self, config):
        c = config["sampler"]["MetropolisHastings"]
        self.target_r_minus_one = c.get("target_r_minus_one", 0.1)
        self.min_cov_r_minus_one = c.get("min_cov_r_minus_one", 10)
        self.n_steps_checkpoint = c.get("n_steps_checkpoint", 1000)
        self.n_steps_incr = c.get("n_steps_incr", 100)
        self.n_steps_min = c.get("n_steps", 100000)
        self.random_start = c.get("random_start", True)
        self.covariance_filename = c.get("covariance_filename", None)
        self.init_covariance = c.get("init_covariance", "jacobian")
        if self.init_covariance == 'None':
            self.init_covariance = None

        
        self.update_covariance = c.get("update_covariance", True)
        self.proposal_distribution = c.get("proposal_distribution", "cosmomc")
        self.minimize_and_sample = c.get("minimize_and_sample", False)
        self.restart = c.get("restart", False)

        if self.init_covariance == "hessian":
            if not self.minimize_and_sample:
                print(
                    "init_covariance = hessian, so setting minimize_and_sample=True to ensure PSD hessian."
                )
                self.minimize_and_sample = True

        self.seeding_config = SeedingConfig.from_sampler_config(
            c, sampler="MetropolisHastings"
        )
        # The hessian coercion above also has to reach the seeding config: it
        # drives whether the metric branch re-polishes the MAP.
        if self.minimize_and_sample != self.seeding_config.minimize_and_sample:
            self.seeding_config = replace(
                self.seeding_config, minimize_and_sample=self.minimize_and_sample
            )
        # init_covariance doubles as the seeding ladder's mass_matrix_init for
        # every value the ladder knows; None/jacobian/from_file stay MH's own.
        if self.init_covariance not in (None, "jacobian", "from_file"):
            # "hessian" means a full covariance with correlations here, which
            # is the ladder's hessian_dense; the ladder's "hessian" is the
            # diagonal metric.
            self.seeding_config = replace(
                self.seeding_config,
                mass_matrix_init=(
                    "hessian_dense" if self.init_covariance == "hessian"
                    else self.init_covariance
                ),
            )

        if self.init_covariance == "from_file":
            self.proposal_covariance = np.genfromtxt(self.covariance_filename)
        else:
            self.proposal_covariance = None

    def run(self, model, output_file):
        if getattr(model.prior, "transform", False):
            raise NotImplementedError(
                "sample_transform is only supported by the NUTS and Minimize "
                "samplers; disable it for this sampler."
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

        if (self.restart) & (os.path.exists(f"{output_file}.proposal_cov.npy")):
            proposal_cov = np.load(f"{output_file}.proposal_cov.npy")
            samples = np.load(f"{output_file}.samples_chk.npy")
            log_density = np.load(f"{output_file}.logposterior_chk.npy")

            if samples.shape[0] == 1:
                initial_positions = (samples[0, :, :] - reference[None, :]) / sigmas[
                    None, :
                ]
            else:
                initial_positions = (samples[:, 0, :] - reference[None, :]) / sigmas[
                    None, :
                ]

        else:
            samples = None
            seeder = self._seeder()
            if self.minimize_and_sample:
                # n_chains, not n_devices: with chains_per_device > 1
                # the chain axis is n_devices * K.  MH never parses
                # chains_per_device, so this is a no-op today.
                initial_positions = seeder.minimize(
                    log_posterior,
                    initial_positions,
                    initial_positions.shape[0],
                    output_file,
                    chi2_threshold=2,
                )

            if self.proposal_covariance is None:
                if self.init_covariance is None:
                    proposal_sigmas = np.array(prior.get_proposal_sigmas()) / sigmas
                    proposal_cov = jnp.diag(proposal_sigmas**2)

                if self.init_covariance == "jacobian":
                    jac = jax.jacobian(jlp)
                    jac_init = jac(initial_positions[0])
                    jac_step = np.abs(1 / jac_init)
                    prior_step = 0.34 * jnp.ones_like(jac_step)
                    proposal_sigmas = jnp.min(jnp.array([prior_step, jac_step]), axis=0)
                    proposal_cov = jnp.diag(proposal_sigmas**2)
                    initial_positions = jnp.tile(
                        initial_positions[0], n_devices
                    ).reshape(n_devices, -1)

                elif self.init_covariance is not None:
                    # Every other value is a seeding mass_matrix_init: take
                    # the metric from the shared ladder and turn it into a
                    # proposal covariance with MH's prior-width cap.
                    rng_key, mm_key = jax.random.split(rng_key)
                    imm, _ = seeder.initial_metric(
                        jlp, initial_positions[0], initial_positions, mm_key
                    )
                    if imm is None:
                        raise ValueError(
                            f"init_covariance: {self.init_covariance} is not a "
                            "known seeding mass_matrix_init"
                        )
                    proposal_cov = proposal_covariance_from_metric(imm)
                    ev = jnp.linalg.eigvalsh(proposal_cov)
                    print(
                        f"Proposal covariance from {self.init_covariance} "
                        f"metric: eigenvalue range [{float(ev.min()):.3e}, "
                        f"{float(ev.max()):.3e}]",
                        flush=True,
                    )

            else:
                proposal_cov = self.proposal_covariance

        if self.proposal_distribution == "gaussian":
            print("Using Gaussian proposal distribution")
            assert (jnp.linalg.eigvalsh(proposal_cov) > 0).all()

            proposal_generator = get_gaussian_proposal_generator(proposal_cov)
            random_walk = blackjax.rmh(jlp, proposal_generator)

        elif self.proposal_distribution == "cosmomc":
            assert (jnp.linalg.eigvalsh(proposal_cov) > 0).all()
            proposal_generator = get_cosmomc_proposal_generator(proposal_cov)
            random_walk = blackjax.rmh(jlp, proposal_generator)

        #        random_walk = blackjax.rmh(jlp, proposal_generator)
        init_pmap = jax.pmap(random_walk.init, in_axes=(0))
        states = init_pmap(initial_positions)

        np.save(f"{output_file}.proposal_cov.npy", proposal_cov)
        np.save(
            f"{output_file}.samples_chk.npy",
            states.position * sigmas[None, None, :] + reference[None, None, :],
        )

        keys = jax.random.split(rng_key, 1 + n_devices)
        rng_key = keys[0]
        sample_keys = keys[1:]

        pmap_inference_loop = self._make_pmap_inference_loop(collect_info=True)

        print("Running inference loop", flush=True)
        rhat = 10000

        if samples is None:
            counter = 0
            n_steps = 0
            log_density = []
            is_accepted = []
            samples = []
        else:
            counter = 0
            n_steps = samples.shape[1]
            is_accepted = jnp.ones(n_steps, dtype=bool)

        while (rhat - 1 > self.target_r_minus_one) | (n_steps < self.n_steps_min):
            if counter == 0:
                states, info = pmap_inference_loop(
                    sample_keys, random_walk.step, states, self.n_steps_incr
                )
            else:
                states = init_pmap(states.position[:, -1, :])
                states, info = pmap_inference_loop(
                    sample_keys, random_walk.step, states, self.n_steps_incr
                )

            if counter == 0:
                is_accepted = info.is_accepted
                samples = states.position
                log_density = states.logdensity
            else:
                is_accepted = np.hstack([is_accepted, info.is_accepted])
                samples = np.hstack([samples, states.position])
                log_density = np.hstack([log_density, states.logdensity])

            n_accepted_steps = np.sum(is_accepted, axis=1)
            print(
                f"n_accepted, n_calls = {n_accepted_steps}, {is_accepted.shape[1]}",
                flush=True,
            )

            if (n_accepted_steps > n_steps + self.n_steps_checkpoint).all():
                samples_temp = jnp.zeros(
                    (n_devices, n_steps + self.n_steps_checkpoint, samples.shape[-1])
                )
                log_density_temp = jnp.zeros(
                    (n_devices, n_steps + self.n_steps_checkpoint)
                )
                for i in range(n_devices):
                    samples_temp = samples_temp.at[i].set(
                        samples[i][is_accepted[i]][: n_steps + self.n_steps_checkpoint]
                    )
                    log_density_temp = log_density_temp.at[i].set(
                        log_density[i][is_accepted[i]][
                            : n_steps + self.n_steps_checkpoint
                        ]
                    )

                rhat = jnp.max(potential_scale_reduction(samples_temp))
                print(
                    f"Acceptance fraction per chain = {n_accepted_steps / is_accepted.shape[1]}",
                    flush=True,
                )
                print(f"rhat - 1 = {rhat - 1}", flush=True)

                if self.update_covariance & (rhat - 1 < self.min_cov_r_minus_one):
                    proposal_cov = jnp.cov(
                        samples_temp.reshape(-1, samples_temp.shape[-1]).T
                    )
                    if self.proposal_distribution == "gaussian":
                        proposal_generator = get_gaussian_proposal_generator(
                            proposal_cov
                        )
                        random_walk = blackjax.rmh(jlp, proposal_generator)
                    elif self.proposal_distribution == "cosmomc":
                        proposal_generator = get_cosmomc_proposal_generator(
                            proposal_cov
                        )
                        random_walk = blackjax.rmh(jlp, proposal_generator)

                    init_pmap = jax.pmap(random_walk.init, in_axes=(0))
                np.save(f"{output_file}.proposal_cov.npy", proposal_cov)
                np.save(
                    f"{output_file}.samples_chk.npy",
                    samples_temp * sigmas[None, None, :] + reference[None, None, :],
                )
                np.save(f"{output_file}.logposterior_chk.npy", log_density_temp)

                n_steps = n_steps + self.n_steps_checkpoint

            counter += 1

            keys = jax.random.split(rng_key, 1 + n_devices)
            rng_key = keys[0]
            sample_keys = keys[1:]

        samples = samples_temp * sigmas[None, None, :] + reference[None, None, :]
        log_density = log_density_temp
        samples = jnp.vstack([samples.T, log_density[..., None].T]).T
        param_names.append("log_posterior")

        return samples, param_names