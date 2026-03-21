from ..util.likelihood_module import LikelihoodModule
from ..data_vector.two_point_spectrum import field_types
import jax.numpy as jnp
import numpy as np



class RealSpace2LoopMatterPowerSpectrum(LikelihoodModule):
    def __init__(self, zmin=0, zmax=2.0, nz=50, kmin=1e-3, kmax=3.95, nk=200, **config):
        self.nz = nz
        self.nk = nk
        self.z = jnp.linspace(zmin, zmax, nz)
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.kmin = kmin
        self.kmax = kmax
        self.p2loop_grid_file = config.get("p2loop_grid_file", None)

        self.p2loop_grid = np.loadtxt(self.p2loop_grid_file)
        self.output_requirements["p_mm"] = [
            "As",
            "ns",
            "H0",
            "w",
            "ombh2",
            "omch2",
            "mnu",
        ] #add counterterm names

        self.interpolation_order = config.get("interpolation_order", "cubic")


    def compute(self, state, params_values):
        """
        saves p_mm to state, where p_mm is the 2-loop matter power spectrum
        with shape (nk, nz) where nk is the number of k values and nz is the number of z values
        computed on the grid defined by self.k and self.z
        """
        #p2loop = some grid of k and z
        #state['p_mm'] = p2loop ()
        return state



