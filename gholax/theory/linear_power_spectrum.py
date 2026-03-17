from .emulator import PijEmulator
from ..util.likelihood_module import LikelihoodModule
import jax.numpy as jnp
import numpy as np


class LinearPowerSpectrum(LikelihoodModule):
    """Compute the linear matter and CDM+baryon power spectra P_m(k,z) and P_cb(k,z).

    Supports emulator and Boltzmann solver backends.
    Writes 'Pm_lin_z', 'Pcb_lin_z', and 'k_lin' to the state.
    """

    def __init__(self, zmin=0.0, zmax=2.0, nz=125, **config):
        """Initialize the linear power spectrum module.

        Args:
            zmin: Minimum redshift.
            zmax: Maximum redshift.
            nz: Number of redshift bins.
            **config: Additional config (use_emulator, use_boltzmann,
                kmin, kmax, nk, emulator_file_name).
        """
        self.nz = nz
        self.kmin = config.get("kmin", 1e-3)
        self.kmax = config.get("kmax", 10)
        self.nk = config.get("nk", 200)
        self.use_emulator = bool(config.get("use_emulator", False))
        self.use_boltzmann = bool(config.get("use_boltzmann", True))

        self.output_requirements = {}
        if self.use_emulator:
            self.z = jnp.linspace(zmin, zmax, self.nz)
            self.k = jnp.logspace(jnp.log10(self.kmin), jnp.log10(self.kmax), self.nk)
            self.emulator_file_name = config["emulator_file_name"]
            self.emulator = PijEmulator(self.emulator_file_name)
            # these are the parameters that are checked in order to decide whether quantities need to be recomputed
            self.output_requirements["Pm_lin_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]
            self.output_requirements["Pcb_lin_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]

        elif self.use_boltzmann:
            self.z = np.linspace(zmin, zmax, self.nz)
            self.k = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nk)

            self.output_requirements["Pm_lin_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]
            self.output_requirements["Pcb_lin_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]

        else:
            raise (
                ValueError(
                    "Cannot currently compute differentiable growth predictions without emulator."
                )
            )

    def compute_emulator(self, state, params_values):
        """Compute P_m and P_cb using the neural network emulator."""
        from .spectral_equivalence import build_equiv_cparam_grid
        cparam_grid = build_equiv_cparam_grid(params_values, self.z, state)

        Plin = self.emulator.predict(cparam_grid)

        state["Pm_lin_z"] = Plin[:, 0, :]
        state["Pcb_lin_z"] = Plin[:, 1, :]
        state["k_lin"] = self.k

        return state

    def compute_boltzmann(self, state, params_values):
        """Compute P_m and P_cb from a CLASS Boltzmann solver result."""
        boltz = state["boltzmann_results"]
        h = params_values["H0"] / 100

        pk_cb = np.zeros((self.nz, self.nk))
        pk_m = np.zeros((self.nz, self.nk))

        for i, z in enumerate(self.z):
            for j, k in enumerate(self.k):
                pk_cb[i, j] = boltz.pk_cb(k * h, z) * h**3
                pk_m[i, j] = boltz.pk(k * h, z) * h**3

        state["Pm_lin_z"] = pk_m
        state["Pcb_lin_z"] = pk_cb
        state["k_lin"] = self.k

        return state

    def compute_analytic(state, params_values):
        """Compute P_lin analytically (not implemented)."""
        raise (NotImplementedError("Analytic Plin(k,z) calculation not implemented."))

    def compute(self, state, params_values):
        """Compute linear power spectra and write to state."""
        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        elif self.use_boltzmann:
            state = self.compute_boltzmann(state, params_values)
        else:
            state = self.compute_analytic(params_values)

        return state
