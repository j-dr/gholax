from .emulator import ScalarEmulator
from ..util.likelihood_module import LikelihoodModule
import jax.numpy as jnp
from jax import grad, jit
from jax.experimental.ode import odeint
from functools import partial

def dark_energy_density(a, w0=-1.0, wa=0.0):
    return a**(-3*(1 + w0 + wa)) * jnp.exp(3*wa*(a - 1))

def hubble_E(a, Omega_m=0.3, Omega_Lambda=0.7, Omega_k=0.0, Omega_r=0.0, 
             w0=-1.0, wa=0.0):
    Omega_DE_term = Omega_Lambda * dark_energy_density(a, w0, wa)
    
    E_squared = (
        Omega_r * a**(-4) + 
        Omega_m * a**(-3) + 
        Omega_k * a**(-2) + 
        Omega_DE_term
    )
    
    return jnp.sqrt(E_squared)

def hubble_E_z(z, Omega_m=0.3, Omega_Lambda=0.7, Omega_k=0.0, Omega_r=0.0,
               w0=-1.0, wa=0.0):
    a = 1 / (1 + z)
    return hubble_E(a, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)

def dE_da_auto(a, Omega_m=0.3, Omega_Lambda=0.7, Omega_k=0.0, Omega_r=0.0,
               w0=-1.0, wa=0.0):
    E_func = partial(hubble_E, 
                     Omega_m=Omega_m, 
                     Omega_Lambda=Omega_Lambda,
                     Omega_k=Omega_k,
                     Omega_r=Omega_r,
                     w0=w0,
                     wa=wa)
    return grad(E_func)(a)

def d2E_da2_auto(a, Omega_m=0.3, Omega_Lambda=0.7, Omega_k=0.0, Omega_r=0.0,
                 w0=-1.0, wa=0.0):
    dE_func = partial(dE_da_auto,
                     Omega_m=Omega_m,
                     Omega_Lambda=Omega_Lambda, 
                     Omega_k=Omega_k,
                     Omega_r=Omega_r,
                     w0=w0,
                     wa=wa)
    return grad(dE_func)(a)

def growth_ode_system(y, a, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa, f_nu):
    """"
    d^2D/da^2 + (3/a + dlnE/da) * dD/da - (3/2) * Omega_m/(a^5 * E^2) * D = 0
    """
    D, dD_da = y
    
    E = hubble_E(a, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)
    dE_da_val = dE_da_auto(a, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)
    
    d2D_da2 = (
        -(3/a + dE_da_val/E) * dD_da + 
        (3/2) * Omega_m * (1 - f_nu) / (a**5 * E**2) * D
    )
    
    return jnp.array([dD_da, d2D_da2])


class LinearGrowth(LikelihoodModule):
    def __init__(self, zmin=0.0, zmax=2.0, nz=125, **config):
        self.nz = nz
        self.z = jnp.linspace(zmin, zmax, self.nz)
        self.use_emulator = bool(config.get("use_emulator", True))
        self.use_boltzmann = bool(config.get("use_boltzmann", False))
        self.solve_ode = bool(config.get("solve_ode", False))
        self.n_points_ode = int(config.get("n_points_ode", 30))
        self.a_init_ode = float(config.get("a_init_ode", 1/(1+50)))
        self.z_ode = jnp.linspace(0, 15, 200)

        self.output_requirements = {}
        if self.use_emulator:
            self.emulator_file_name = config["emulator_file_name"]
            self.emulator = ScalarEmulator(self.emulator_file_name)
            # these are the parameters that are checked in order to decide whether quantities need to be recomputed
            self.output_requirements["sigma8_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]
            self.output_requirements["D_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]

        elif self.use_boltzmann:
            self.output_requirements["sigma8_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]
            self.output_requirements["D_z"] = [
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
                params_values["H0"],
                params_values["w"],
                params_values["ombh2"],
                params_values["omch2"],
                jnp.log10(params_values["mnu"]),
            ]
        )

        cparam_grid = jnp.zeros((self.nz, len(cosmo_params) + 1))
        cparam_grid = cparam_grid.at[:, :-1].set(cosmo_params)
        cparam_grid = cparam_grid.at[:, -1].set(self.z)

        sigma8_z = self.emulator.predict(cparam_grid)[:, 0]
        Dz = sigma8_z / sigma8_z[0]

        state["sigma8_z"] = sigma8_z
        state["D_z"] = Dz

        return state

    def compute_boltzmann(self, state, params_values):
        boltz = state["boltzmann_results"]
        Dz = jnp.array([boltz.scale_independent_growth_factor(z) for z in self.z])
        sigma8_z = jnp.array([boltz.sigma(8, z, h_units=True) for z in self.z])
        state["sigma8_z"] = sigma8_z
        state["D_z"] = Dz

        return state
    
    def compute_ode(self, state, params_values):
        h = params_values["H0"] / 100
        omega_nu = params_values["mnu"] / 93.14
        omegam = (params_values["omch2"] + params_values["ombh2"] + omega_nu) / h**2    
        fnu = omega_nu / (omegam * h**2)        
        omegal = 1 - omegam
        w0 = params_values["w"]

        D_init = self.a_init_ode
        dD_da_init = 1.0
        y0 = jnp.array([D_init, dD_da_init])
        
        a_points = jnp.logspace(jnp.log10(self.a_init_ode), 0, self.n_points_ode)
        
        ode_func = partial(growth_ode_system, 
                        Omega_m=omegam, 
                        Omega_Lambda=omegal,
                        Omega_k=0.,
                        Omega_r=0.,
                        w0=w0,
                        wa=0.0,
                        fnu=fnu)
        
        solution = odeint(ode_func, y0, a_points)
        D_solution = solution[:, 0]
        
        D_norm = D_solution[-1]
        D_normalized = D_solution / D_norm
        
        a_target = 1 / (1 + self.z_ode)
        a_target_array = jnp.atleast_1d(a_target)
        D_interp = jnp.interp(a_target_array, a_points, D_normalized)
        
        state["D_z"] = (self.z_ode, D_interp)

        return state


    def compute(self, state, params_values):
        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        elif self.use_boltzmann:
            state = self.compute_boltzmann(state, params_values)
                        
        if self.solve_ode:
            state = self.compute_ode(state, params_values)


        return state
