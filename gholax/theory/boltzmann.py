try:
    from classy import Class
except:
    pass
from .emulator import ScalarEmulator
from ..util.likelihood_module import LikelihoodModule
import jax.numpy as jnp
from jax import lax
from interpax import interp1d


class Boltzmann(LikelihoodModule):
    def __init__(self, **config):
        self.output_requirements = {}
        self.output_requirements["boltzmann_results"] = [
            "As",
            "ns",
            "H0",
            "w",
            "ombh2",
            "omch2",
            "mnu",
        ]
        self.nonlinear_pk = config.get("nonlinear_pk", "none")

    def compute(self, state, params_values):
        """Calculates stuff.

        Args:
            state dict: Inputs are contained here and outputs are written.
            params_values dict: Dictionary of parameter values.
        """
        boltz_params = {
            "output": "mPk",
            "P_k_max_h/Mpc": 20.0,
            "z_pk": "0.0,10",
            "A_s": params_values["As"] * 1e-9,
            "n_s": params_values["ns"],
            "h": params_values["H0"] / 100,
            "N_ur": 0.00641,
            "N_ncdm": 1,
            "deg_ncdm": 3,
            "Omega_Lambda": 0.0,
            "w0_fld": params_values["w"],
            "wa_fld": 0.0,
            "m_ncdm": params_values["mnu"] / 3,
            "tau_reio": 0.0568,
            "omega_b": params_values["ombh2"],
            "omega_cdm": params_values["omch2"],
            "non_linear": self.nonlinear_pk,
        }

        boltz = Class()
        boltz.set(boltz_params)
        boltz.compute()

        state["boltzmann_results"] = boltz

        return state
