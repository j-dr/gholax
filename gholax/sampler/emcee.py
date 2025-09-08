from datetime import datetime

import emcee
import jax
import jax.numpy as jnp
import numpy as np


class Emcee(object):
    def __init__(self, config):
        c = config["sampler"]["Emcee"]
        self.nwalkers = c.get("nwalkers", "3d")
        self.vectorize = c.get("vectorize", True)
        self.n_steps_incr = c.get("n_steps_incr", 50)
        self.restart = c.get("restart", False)
        self.target_neff = c.get("target_neff", 2000)
        self.burn_in = c.get("burn_in", 100)
        self.random_start = c.get("random_start", True)

    def run(self, model, output_file):
        rng_key = jax.random.key(int(datetime.now().strftime("%Y%m%d%s")))
        param_names = model.prior.params
        prior = model.prior

        log_posterior = model.log_posterior

        jlp = jax.jit(log_posterior)
        if self.vectorize:
            jlp = jax.vmap(jlp)

        ndim = len(param_names)
        if type(self.nwalkers) is str:
            nwalkers = int(self.nwalkers.split("d")[0]) * ndim
        else:
            nwalkers = int(self.nwalkers)

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        except:
            rank = 0

        if self.restart:
            filename = f"{output_file}.{rank}.samples.h5"
            backend = emcee.backends.HDFBackend(filename)
            s = emcee.EnsembleSampler(
                nwalkers, ndim, jlp, vectorize=self.vectorize, backend=backend
            )
            samples = s.get_chain()
            tau = np.max(s.get_autocorr_time(tol=0))
            neff = len(samples) * nwalkers / tau
            state = s.get_last_sample()

        else:
            filename = f"{output_file}.{rank}.samples.h5"
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)
            s = emcee.EnsembleSampler(
                nwalkers, ndim, jlp, vectorize=self.vectorize, backend=backend
            )
            keys = jax.random.split(rng_key, nwalkers + 1)
            rng_key = keys[0]
            initial_keys = keys[1:]
            initial_positions = np.array(
                [
                    list(
                        prior.initial_position(
                            random_start=self.random_start, key=k, normalize=False
                        ).values()
                    )
                    for k in initial_keys
                ]
            )
            state = s.run_mcmc(initial_positions, self.burn_in)
            s.reset()
            neff = 0
            tau = np.inf

        while (neff < self.target_neff) | (neff / 50 < tau):
            state = s.run_mcmc(state, self.n_steps_incr, store=True)
            samples = s.get_chain()
            tau = np.max(s.get_autocorr_time(tol=0))
            neff = len(samples) * nwalkers / tau

            print(
                f"Acceptance fraction {np.mean(s.acceptance_fraction)} after {len(samples)} samples per walker.",
                flush=True,
            )
            print(f"Neff = {neff}, tau = {tau}", flush=True)

        samples = s.get_chain()
        logp = s.get_log_prob()
        samples = np.einsum("swp->wsp", samples)
        samples = jnp.vstack([samples.T, logp]).T
        param_names.extend(["log_posterior"])

        return samples, param_names
