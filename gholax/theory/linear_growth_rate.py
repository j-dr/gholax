from .emulator import ScalarEmulator
from ..util.likelihood_module import LikelihoodModule
import jax.numpy as jnp


class LinearGrowthRate(LikelihoodModule):
    def __init__(self, zmin=0.0, zmax=2.0, nz=125, **config):
        self.nz = nz
        self.z = jnp.linspace(zmin, zmax, self.nz)
        self.use_emulator = bool(config.get("use_emulator", True))
        self.use_boltzmann = bool(config.get("use_boltzmann", False))

        self.output_requirements = {}
        if self.use_emulator:
            self.emulator_file_name = config["emulator_file_name"]
            self.emulator = ScalarEmulator(self.emulator_file_name)
            # these are the parameters that are checked in order to decide whether quantities need to be recomputed
            self.output_requirements["f_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]

        elif self.use_boltzmann:
            self.output_requirements["f_z"] = [
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
        cosmo_params = jnp.array(
            [
                params_values["As"],
                params_values["ns"],
                params_values["omch2"],
                params_values["ombh2"],
                params_values["H0"],
                params_values["w"],
                jnp.log10(params_values["mnu"]),
            ]
        )

        cparam_grid = jnp.zeros((self.nz, len(cosmo_params) + 1))
        cparam_grid = cparam_grid.at[:, :-1].set(cosmo_params)
        cparam_grid = cparam_grid.at[:, -1].set(self.z)

        f_z = self.emulator.predict(cparam_grid)[:, 0]
        state["f_z"] = f_z
        state["z_pk"] = self.z

        return state

    def compute_boltzmann(self, state, params_values):
        boltz = state["boltzmann_results"]
        f_z = jnp.array([boltz.scale_independent_growth_factor_f(z) for z in self.z])
        state["f_z"] = f_z[:, 0]
        state["z_pk"] = self.z

        return state

    def compute_analytic(state, params_values):
        raise (NotImplementedError("Analytic f(z) calculation not implemented."))

    def compute(self, state, params_values):
        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        elif self.use_boltzmann:
            state = self.compute_boltzmann(state, params_values)
        else:
            state = self.compute_analytic(params_values)

        return state
