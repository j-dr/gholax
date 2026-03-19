from .emulator import ScalarEmulator
from ..util.likelihood_module import LikelihoodModule
import jax.numpy as jnp
from jax import grad, jit
from jax.experimental.ode import odeint
from functools import partial

# Physical constants for neutrino thermodynamics
_T_CMB0_K = 2.7255                   # K, Fixsen (2009)
_K_B_EV = 8.617333e-5                # eV/K
_T_NU0_EV = (4.0/11.0)**(1.0/3.0) * _K_B_EV * _T_CMB0_K  # neutrino temperature today in eV
_OMEGA_GAMMA_H2 = 2.473e-5           # photon omega*h^2 for T_CMB=2.7255 K
_OMEGA_NU_REL_H2_PER_SPECIES = (7.0/8.0) * (4.0/11.0)**(4.0/3.0) * _OMEGA_GAMMA_H2


def neutrino_density_ratio(y, nodes, weights):
    """
    Returns F(y) = (120/(7*pi^4)) * integral_0^inf sqrt(x^2+y^2)*x^2/(e^x+1) dx
    via Gauss-Laguerre quadrature (nodes and weights from scipy.special.roots_laguerre).

    y = m_nu_per_species * a / _T_NU0_EV

    Limits: F(0) = 1 (fully relativistic); F(y>>1) -> (45*zeta(3)/(7*pi^4))*y
    (non-relativistic, recovering omega_nu*h^2 = mnu/93.14).

    JAX-differentiable in y.
    """
    norm = 120.0 / (7.0 * jnp.pi**4)
    # Gauss-Laguerre integrates f(x)*e^{-x}: rewrite 1/(e^x+1) = e^{-x}/(1+e^{-x})
    integrand = jnp.sqrt(nodes**2 + y**2) * nodes**2 / (1.0 + jnp.exp(-nodes))
    return norm * jnp.dot(weights, integrand)


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

def d3E_da3_auto(a, Omega_m=0.3, Omega_Lambda=0.7, Omega_k=0.0, Omega_r=0.0,
                 w0=-1.0, wa=0.0):
    d2E_func = partial(d2E_da2_auto,
                     Omega_m=Omega_m,
                     Omega_Lambda=Omega_Lambda, 
                     Omega_k=Omega_k,
                     Omega_r=Omega_r,
                     w0=w0,
                     wa=wa)
    return grad(d2E_func)(a)

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

        self.compute_w0wa = bool(config.get("compute_w0wa", False))
        self.output_requirements = {}
        if self.use_emulator:
            self.emulator_file_name = config["emulator_file_name"]
            self.emulator = ScalarEmulator(self.emulator_file_name)
            # these are the parameters that are checked in order to decide whether quantities need to be recomputed
            if self.compute_w0wa:
                params = ["As", "ns", "H0", "w", "wa", "ombh2", "omch2", "mnu"]
            else:
                params = ["As", "ns", "H0", "w", "ombh2", "omch2", "mnu"]
            self.output_requirements["sigma8_z"] = params
            self.output_requirements["D_z"] = params

        elif self.use_boltzmann:
            self.output_requirements["sigma8_z"] = ["boltzmann_results"]
            self.output_requirements["D_z"] = ["boltzmann_results"]

        else:
            raise (
                ValueError(
                    "Cannot currently compute differentiable growth predictions without emulator."
                )
            )
        if self.solve_ode:
            self.output_requirements['D_ode'] = [
                "As",
                "ns",
                "H0",
                "w",
                "wa",
                "ombh2",
                "omch2",
                "mnu",
            ]

    def _compute_D_at_z(self, omegam, omegal, fnu, w0, wa):
        """Returns unnormalized D(z) interpolated onto self.z."""
        D_init = self.a_init_ode
        y0 = jnp.array([D_init, 1.0])
        a_points = jnp.logspace(jnp.log10(self.a_init_ode), 0, self.n_points_ode)
        ode_func = partial(growth_ode_system,
                           Omega_m=omegam, Omega_Lambda=omegal,
                           Omega_k=0., Omega_r=0.,
                           w0=w0, wa=wa, f_nu=fnu)
        solution = odeint(ode_func, y0, a_points)
        D_solution = solution[:, 0]
        a_target = 1.0 / (1.0 + self.z)
        return jnp.interp(a_target, a_points, D_solution)

    def compute_w0wa_emulator(self, state, params_values):
        # Run wCDM emulator with w=w0, wa=0
        cosmo_params = jnp.array([
            params_values["As"],
            params_values["ns"],
            params_values["H0"],
            params_values["w"],
            params_values["ombh2"],
            params_values["omch2"],
            jnp.log10(params_values["mnu"]),
        ])
        cparam_grid = jnp.zeros((self.nz, len(cosmo_params) + 1))
        cparam_grid = cparam_grid.at[:, :-1].set(cosmo_params)
        cparam_grid = cparam_grid.at[:, -1].set(self.z)
        sigma8_z = self.emulator.predict(cparam_grid)[:, 0]

        # Growth factor correction: unnormalized ODE ratio D_w0wa / D_wCDM
        h = params_values["H0"] / 100
        omega_nu = params_values["mnu"] / 93.14
        omegam = (params_values["omch2"] + params_values["ombh2"] + omega_nu) / h**2
        omegal = 1 - omegam
        fnu = omega_nu / (omegam * h**2)
        w0 = params_values["w"]
        wa = params_values["wa"]

        D_wcdm = self._compute_D_at_z(omegam, omegal, fnu, w0=w0, wa=0.0)
        D_w0wa = self._compute_D_at_z(omegam, omegal, fnu, w0=w0, wa=wa)
        sigma8_z = sigma8_z * (D_w0wa / D_wcdm)

        Dz = sigma8_z / sigma8_z[0]
        state["z_D"] = self.z
        state["sigma8_z"] = sigma8_z
        state["D_z"] = Dz
        return state

    def compute_emulator(self, state, params_values):
        from .spectral_equivalence import build_equiv_cparam_grid
        cparam_grid = build_equiv_cparam_grid(params_values, self.z, state)

        sigma8_z = self.emulator.predict(cparam_grid)[:, 0]
        Dz = sigma8_z / sigma8_z[0]
        
        state["z_D"] = self.z
        state["sigma8_z"] = sigma8_z
        state["D_z"] = Dz

        return state

    def compute_boltzmann(self, state, params_values):
        boltz = state["boltzmann_results"]
        Dz = jnp.array([boltz.scale_independent_growth_factor(z) for z in self.z])
        sigma8_z = jnp.array([boltz.sigma(8, z, h_units=True) for z in self.z])
        sigma12_z = jnp.array([boltz.sigma(12, z, h_units=False) for z in self.z])
        
        state["z_D"] = self.z
        state["sigma8_z"] = sigma8_z
        state["D_z"] = Dz
        state["sigma12_z"] = sigma12_z

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
                        wa=params_values.get("wa", 0.0),
                        f_nu=fnu)
        
        solution = odeint(ode_func, y0, a_points)
        D_solution = solution[:, 0]
        
        D_norm = D_solution[-1]
        D_normalized = D_solution / D_norm
        
        a_target = 1 / (1 + self.z_ode)
        a_target_array = jnp.atleast_1d(a_target)
        D_interp = jnp.interp(a_target_array, a_points, D_normalized)
        
        state["z_Dz_ode"] = self.z_ode
        state["Dz_ode"] = D_interp

        return state


    def compute(self, state, params_values):
        if self.use_emulator and self.compute_w0wa:
            state = self.compute_w0wa_emulator(state, params_values)
        elif self.use_emulator:
            state = self.compute_emulator(state, params_values)
        elif self.use_boltzmann:
            state = self.compute_boltzmann(state, params_values)
                        
        if self.solve_ode:
            state = self.compute_ode(state, params_values)


        return state