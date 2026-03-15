from .emulator import ScalarEmulator
from ..util.likelihood_module import LikelihoodModule
from .linear_growth import hubble_E_z, dE_da_auto, d2E_da2_auto, d3E_da3_auto
import jax.numpy as jnp
from scipy.special import roots_legendre

speed_of_light = 2.99792458e5

def cumulative_trapz(y, x):
    """
    Cumulative trapezoidal integral from x[0] to x[i].
    Pure JAX: differentiable w.r.t. y (and thus parameters inside y).
    """
    dx = x[1:] - x[:-1]
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return jnp.concatenate([jnp.array([0.0], dtype=y.dtype), jnp.cumsum(area)])


def comoving_distance_integral(z_targets, n_int=4096,\
                         Omega_m=0.3, Omega_Lambda=0.7, Omega_k=0.0, Omega_r=0.0,\
                         w0=-1.0, wa=0.0):
    """
    Returns (E(z_targets), chi(z_targets)) where chi is comoving distance in Mpc/h.

    Uses a uniform z-grid up to max(z_targets), cumulative trapezoid of 1/E(z).
    """
    z_targets = jnp.atleast_1d(z_targets)
    zmax = jnp.max(z_targets)

    z_grid = jnp.linspace(0.0, zmax, n_int)
    E_grid = hubble_E_z(z_grid, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)
    I_grid = cumulative_trapz(1.0 / E_grid, z_grid)  # \int_0^z dz'/E(z')
    chi_grid = (speed_of_light/ 100.) * I_grid  # Mpc since H0 is km/s/Mpc

    E_targets = hubble_E_z(z_targets, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)
    chi_targets = jnp.interp(z_targets, z_grid, chi_grid)

    dEda = dE_da_auto(1.0, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)
    d2Eda2 = d2E_da2_auto(1.0, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)
    d3Eda3 = d3E_da3_auto(1.0, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)
    dchidz = 1.0
    d2chidz2 = dEda
    d3chidz3 = -2*dEda + 2*dEda**2  - d2Eda2
    d4chidz4 = 6*dEda-12*dEda**2+6*dEda**3+6*d2Eda2-6*dEda*d2Eda2+d3Eda3

    chi_derivs = jnp.array([dchidz,d2chidz2,d3chidz3,d4chidz4])
    
    return chi_targets, E_targets, chi_derivs


class ExpansionHistory(LikelihoodModule):
    def __init__(self, zmin=0, zmax=2.0, nz=125, **config):
        self.z = jnp.linspace(zmin, zmax, nz)

        self.use_direct_integration = bool(config.get("use_direct_integration", True))
        self.use_emulator = bool(config.get("use_emulator", False))
        self.use_boltzmann = bool(config.get("use_boltzmann", False))
        self.integration_method = config.get("integration_method", "gl_quad")
        if self.integration_method == "gl_quad":
            self.nodes, self.weights = roots_legendre(nz)

        self.output_requirements = {}
        if self.use_emulator:
            self.output_requirements["chi_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]
            self.output_requirements["e_z"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]
            self.emulator_file_names = config["emulator_file_names"]
            self.emulators = {}
            self.emulators["chi_z"] = ScalarEmulator(self.emulator_file_names["chi_z"])
            self.emulators["e_z"] = ScalarEmulator(self.emulator_file_names["e_z"])

        elif self.use_boltzmann:
            self.output_requirements["chi_z"] = ["boltzmann_results"]
            self.output_requirements["e_z"] = ["boltzmann_results"]
            self.output_requirements["omegam"] = ["boltzmann_results"]
        else:
            self.output_requirements["chi_z"] = ["omch2", "ombh2", "H0", "mnu", "w"]
            self.output_requirements["e_z"] = ["omch2", "ombh2", "H0", "mnu", "w"]
            self.output_requirements["omegam"] = ["omch2", "ombh2", "H0", "mnu"]

    def T_AK(self, x):
        """The "T" function of Adachi & Kasai (2012), used below."""
        b1, b2, b3 = 2.64086441, 0.883044401, 0.0531249537
        c1, c2, c3 = 1.39186078, 0.512094674, 0.0394382061
        x3 = x**3
        x6 = x3 * x3
        x9 = x3 * x6
        tmp = 2 + b1 * x3 + b2 * x6 + b3 * x9
        tmp /= 1 + c1 * x3 + c2 * x6 + c3 * x9
        tmp *= x**0.5

        return tmp

    def chi_of_z(self, omegam):
        """The comoving distance to redshift z, in Mpc/h.
        Uses the Pade approximate of Adachi & Kasai (2012) to compute chi
        for a LCDM model, ignoring massive neutrinos."""
        s_ak = ((1 - omegam) / omegam) ** 0.3333333
        tmp = self.T_AK(s_ak) - self.T_AK(s_ak / (1 + self.z))
        tmp = jnp.multiply(tmp, 2997.925 / (s_ak * omegam) ** 0.5)
        return tmp

    def E_of_z(self, omegam):
        """The dimensionless Hubble parameter at zz."""
        Ez = (omegam * (1 + self.z) ** 3 + (1 - omegam)) ** 0.5
        return Ez

    def compute_pade(self, params_values):
        h = params_values["H0"] / 100
        omega_nu = params_values["mnu"] / 93.14
        omegam = (params_values["omch2"] + params_values["ombh2"] + omega_nu) / h**2

        chi_z = self.chi_of_z(omegam)
        e_z = self.E_of_z(omegam)

        return chi_z, e_z, omegam
    
    def compute_analytic(self, params_values):

        h = params_values['H0'] / 100.
        omega_nu = params_values["mnu"] / 93.14
        Omega_m = (params_values["omch2"] + params_values["ombh2"] + omega_nu) / h**2
        w0 = params_values["w"]
        
        # These are set to zero for now
        wa = 0.0
        Omega_r = 0.0

        Omega_Lambda = 1 - Omega_m
        Omega_k = 1 - Omega_m - Omega_Lambda
        
        n_int = getattr(self, "n_points_int", 1024)

        chi_z, e_z, chi_derivs = comoving_distance_integral(self.z, n_int, Omega_m, Omega_Lambda, Omega_k, Omega_r, w0, wa)
        
        return chi_z, e_z, Omega_m, chi_derivs
    
    def compute_emulator(self, params_values):
        cosmo_params = jnp.array(
            [
                params_values["As"] * 10**9,
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

        chi_z = self.emulators["chi_z"].predict(cparam_grid)
        e_z = self.emulators["e_z"].predict(cparam_grid)

        return chi_z, e_z

    def compute_boltzmann(self, params_values, state):
        speed_of_light = 2.99792458e5
        boltz = state["boltzmann_results"]
        h = params_values["H0"] / 100
        e_z = jnp.array([boltz.Hubble(z) * speed_of_light / h / 100 for z in self.z])
        chi_z = jnp.array([boltz.angular_distance(z) * (1.0 + z) * h for z in self.z])

        return chi_z[:, 0], e_z[:, 0]

    def compute(self, state, params_values):
        if self.use_direct_integration:
            chi_z, e_z, omegam, chi_derivs = self.compute_analytic(params_values)
            state['omegam'] = omegam
            state['chi_derivs'] = chi_derivs
        
        elif self.use_emulator:
            chi_z, e_z = self.compute_emulator(params_values)
            omega_nu = params_values["mnu"] / 93.14
            h = params_values["H0"] / 100
            state["omegam"] = (
                params_values["omch2"] + params_values["ombh2"] + omega_nu
            ) / h**2

                        
        elif self.use_boltzmann:
            chi_z, e_z = self.compute_boltzmann(params_values, state)
            omega_nu = params_values["mnu"] / 93.14
            h = params_values["H0"] / 100
            state["omegam"] = (
                params_values["omch2"] + params_values["ombh2"] + omega_nu
            ) / h**2

        else:
            chi_z, e_z, omegam = self.compute_pade(params_values)
            state["omegam"] = omegam
            
        state["chi_z"] = chi_z
        state["e_z"] = e_z

        if not self.use_direct_integration:
            state['chi_derivs'] = jnp.array([1,\
                                -1.5 * state['omegam'],\
                                -3*state['omegam'] + 27./4 * state['omegam']**2,\
                                -3*state['omegam'] + (81*state['omegam']**2)/2. - (405*state['omegam']**3)/8.,])

        if self.integration_method == "gl_quad":
            chi_min = jnp.min(chi_z) + 1e-5
            chi_max = jnp.max(chi_z)
            chi_z_proj = 0.5 * (chi_max - chi_min) * (self.nodes + 1) + chi_min
            gl_scaled_weights = 0.5 * (chi_max - chi_min) * self.weights
            state["gl_scaled_weights"] = gl_scaled_weights
        else:
            chi_z_proj = jnp.linspace(
                jnp.min(chi_z) + 1e-5, jnp.max(chi_z), chi_z.shape[0]
            )

        z_limber = jnp.interp(chi_z_proj, chi_z, self.z)
        e_z_limber = jnp.interp(z_limber, self.z, e_z)
        state["chi_z_limber"] = chi_z_proj
        state["z_limber"] = z_limber
        state["e_z_limber"] = e_z_limber

        return state