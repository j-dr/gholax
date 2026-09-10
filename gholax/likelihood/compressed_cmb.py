from typing import Dict

import jax.numpy as jnp
import numpy as np

from .likelihood import Likelihood
from ..theory.cmb_compression import theta_star

# DESI DR2 Results II (arXiv:2503.14738) Appendix A, eqs. A1-A2: Gaussian
# prior on (theta_*, omega_b h^2, omega_bc h^2) from the CamSpec PR4 CMB
# likelihood, marginalized over late-time effects.
DESI_DR2_MEAN = np.array([0.01041, 0.02223, 0.14208])
DESI_DR2_COV = 1e-9 * np.array([
    [0.006621, 0.12444, -1.1929],
    [0.12444, 21.344, -94.001],
    [-1.1929, -94.001, 1488.4],
])


class CompressedCMBLikelihood(Likelihood):
    """Gaussian likelihood on early-Universe CMB compression
    (theta_*, omega_b h^2, omega_b h^2 + omega_c h^2).

    theta_* is computed analytically (gholax.theory.cmb_compression), so
    the likelihood is smooth everywhere in the prior box, unlike a flow
    trained on a finite chain. Config keys ``mean`` and ``cov`` override the
    DESI DR2 defaults; ``n_massive_neutrinos`` (default 3) the neutrino split.
    """

    def __init__(self, config):
        c = config["likelihood"]["CompressedCMBLikelihood"]
        self.mean = jnp.asarray(c.get("mean", DESI_DR2_MEAN), dtype=float)
        cov = np.asarray(c.get("cov", DESI_DR2_COV), dtype=float)
        self.cinv = jnp.asarray(np.linalg.inv(cov))
        self.n_int = int(c.get("n_int", 4096))
        # 3 degenerate species (ExpansionHistory); 1 reproduces CAMB/DESI chains
        self.n_massive = int(c.get("n_massive_neutrinos", 3))
        self.likelihood_pipeline = []
        super().__init__(c, config["likelihood"].get("params", {}))

    def observables(self, params: Dict) -> jnp.ndarray:
        """(theta_*, omega_b h^2, omega_bc h^2) at the given parameters."""
        _, p = self.setup_state_params(dict(params))
        th = theta_star(
            p["H0"], p["ombh2"], p["omch2"], p.get("w", -1.0),
            p.get("wa", 0.0), p.get("mnu", 0.06), self.n_int, self.n_massive,
        )
        return jnp.array([th, p["ombh2"], p["ombh2"] + p["omch2"]])

    def compute(self, params: Dict) -> jnp.ndarray:
        d = self.observables(params) - self.mean
        return -0.5 * d @ self.cinv @ d
