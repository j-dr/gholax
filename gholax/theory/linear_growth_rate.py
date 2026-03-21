from .emulator import ScalarEmulator
from ..util.likelihood_module import LikelihoodModule
import jax.numpy as jnp


class LinearGrowthRate(LikelihoodModule):
    """Compute the linear growth rate f(z) = d ln D / d ln a.

    Supports emulator and Boltzmann solver backends.
    Writes 'f_z' and 'z_pk' to the state.
    """

    def __init__(self, zmin=0.0, zmax=2.0, nz=125, **config):
        """Initialize the linear growth rate module.

        Args:
            zmin: Minimum redshift.
            zmax: Maximum redshift.
            nz: Number of redshift bins.
            **config: Additional config (use_emulator, use_boltzmann, emulator_file_name).
        """
        self.nz = nz
        self.z = jnp.linspace(zmin, zmax, self.nz)
        self.use_emulator = bool(config.get("use_emulator", True))
        self.use_boltzmann = bool(config.get("use_boltzmann", False))

        self.output_requirements = {}
        if self.use_emulator:
            self.emulator_file_name = config["emulator_file_name"]
            self.emulator = ScalarEmulator(self.emulator_file_name)
            ipo = getattr(self.emulator, 'input_param_order', None)
            params = ["As", "ns", "H0", "w", "ombh2", "omch2", "mnu"]
            if ipo is not None and "wa" in ipo:
                params.append("wa")
            self.output_requirements["f_z"] = params

        elif self.use_boltzmann:
            self.output_requirements["f_z"] = ["boltzmann_results"]

        else:
            raise (
                ValueError(
                    "Cannot currently compute differentiable growth predictions without emulator."
                )
            )

    def compute_emulator(self, state, params_values):
        """Compute f(z) using the neural network emulator."""
        from .spectral_equivalence import build_equiv_cparam_grid_custom_order
        order = getattr(
            self.emulator, 'input_param_order',
            ["As", "ns", "omch2", "ombh2", "H0", "w", "logmnu", "z"],
        )
        cparam_grid = build_equiv_cparam_grid_custom_order(
            params_values, self.z, state, order,
        )

        f_z = self.emulator.predict(cparam_grid)[:, 0]
        state["f_z"] = f_z
        state["z_pk"] = self.z

        return state

    def compute_boltzmann(self, state, params_values):
        """Compute f(z) from a CLASS Boltzmann solver result."""
        boltz = state["boltzmann_results"]
        f_z = jnp.array([boltz.scale_independent_growth_factor_f(z) for z in self.z])
        state["f_z"] = f_z[:, 0]
        state["z_pk"] = self.z

        return state

    def compute_analytic(state, params_values):
        """Compute f(z) analytically (not implemented)."""
        raise (NotImplementedError("Analytic f(z) calculation not implemented."))

    def compute(self, state, params_values):
        """Compute linear growth rate and write 'f_z', 'z_pk' to state."""
        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        elif self.use_boltzmann:
            state = self.compute_boltzmann(state, params_values)
        else:
            state = self.compute_analytic(params_values)

        return state
