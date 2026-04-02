from .emulator import ScalarEmulator
from ..util.likelihood_module import LikelihoodModule
from .linear_growth import (
    growth_ode_system, dark_energy_density, neutrino_density_ratio,
    _T_NU0_EV, _OMEGA_GAMMA_H2, _OMEGA_NU_REL_H2_PER_SPECIES,
)
from .expansion_history import comoving_distance_integral_full, speed_of_light

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from functools import partial
from scipy.special import roots_laguerre


class SpectralEquivalence(LikelihoodModule):
    """Find equivalent wCDM cosmologies for w0wa models via distance + amplitude matching.

    For each redshift z_i, finds w_equiv such that chi_wCDM(z_i -> z_LSS; w_equiv)
    matches chi_w0wa(z_i -> z_LSS; w0, wa), then adjusts As to match sigma8(z).

    Reference: arXiv:2510.09503
    """

    def __init__(self, z, z_lss=1089.0, n_newton=5, n_int_chi=2048,
                 n_gl_chi=32, sigma8_emulator_file_name=None, **config):
        self.z = jnp.array(z)
        self.nz = len(z)
        self.z_lss = z_lss
        self.n_newton = n_newton
        self.n_int_chi = n_int_chi

        _gl_nu_nodes_np, _gl_nu_weights_np = roots_laguerre(32)
        self.gl_nu_nodes = jnp.array(_gl_nu_nodes_np)
        self.gl_nu_weights = jnp.array(_gl_nu_weights_np)

        # GL quadrature nodes/weights for chi integrals inside Newton loop
        from scipy.special import roots_legendre as _roots_legendre
        _gl_chi_nodes_np, _gl_chi_weights_np = _roots_legendre(n_gl_chi)
        self.gl_chi_nodes = jnp.array(_gl_chi_nodes_np)
        self.gl_chi_weights = jnp.array(_gl_chi_weights_np)

        self.sigma8_emu = None
        if sigma8_emulator_file_name is not None:
            self.sigma8_emu = ScalarEmulator(sigma8_emulator_file_name)

        # ODE grid for growth factor
        self.n_a_ode = 100
        self.a_init_ode = 1.0 / (1.0 + 200.0)

        self.output_requirements = {
            "w_equiv_z": ["As", "ns", "H0", "w", "wa", "ombh2", "omch2", "mnu"],
            "As_equiv_z": ["As", "ns", "H0", "w", "wa", "ombh2", "omch2", "mnu"],
            "z_equiv": ["As", "ns", "H0", "w", "wa", "ombh2", "omch2", "mnu"],
        }

    def _build_E_z_func(self, Omega_cb, Omega_Lambda, Omega_r_photon,
                        mnu_per_species, h, w0, wa):
        """Build a JAX-traceable E(z) function for a CPL dark energy model."""
        gl_nodes = self.gl_nu_nodes
        gl_weights = self.gl_nu_weights
        n_species = 3

        def E_z(z):
            a = 1.0 / (1.0 + z)
            y = mnu_per_species * a / _T_NU0_EV
            F_y = neutrino_density_ratio(y, gl_nodes, gl_weights)
            Omega_nu_a = n_species * (_OMEGA_NU_REL_H2_PER_SPECIES / h**2) * F_y / a**4
            Omega_de = Omega_Lambda * dark_energy_density(a, w0, wa)
            return jnp.sqrt(Omega_r_photon / a**4 + Omega_cb / a**3 + Omega_nu_a + Omega_de)

        return E_z

    def _chi_z_to_zlss(self, z_start, w_const, Omega_cb, Omega_Lambda,
                       Omega_r_photon, mnu_per_species, h):
        """Comoving distance from z_start to z_LSS for wCDM with constant w.

        Uses Gauss-Legendre quadrature in log(1+z) space for efficiency
        inside Newton iterations (avoids building a full cumulative grid).
        """
        E_z_func = self._build_E_z_func(
            Omega_cb, Omega_Lambda, Omega_r_photon, mnu_per_species, h, w_const, 0.0
        )

        # GL quadrature in u = log(1+z) space:
        # chi = (c/H0) * int_{u_start}^{u_LSS} exp(u) / E(exp(u)-1) du
        u_start = jnp.log1p(z_start)
        u_lss = jnp.log1p(self.z_lss)
        half_width = 0.5 * (u_lss - u_start)
        mid = 0.5 * (u_lss + u_start)
        u_nodes = half_width * self.gl_chi_nodes + mid
        z_nodes = jnp.expm1(u_nodes)
        E_vals = jax.vmap(E_z_func)(z_nodes)
        # Jacobian: dz/du = exp(u) = 1+z
        integrand = (1.0 + z_nodes) / E_vals
        chi = (speed_of_light / 100.0) * half_width * jnp.dot(self.gl_chi_weights, integrand)
        return chi

    def _compute_D_unnorm(self, Omega_m, Omega_Lambda, f_nu, w0, wa, z_arr):
        """Compute unnormalized growth factor D(z) via ODE."""
        a_points = jnp.logspace(jnp.log10(self.a_init_ode), 0.0, self.n_a_ode)
        y0 = jnp.array([self.a_init_ode, 1.0])
        ode_func = partial(growth_ode_system,
                           Omega_m=Omega_m, Omega_Lambda=Omega_Lambda,
                           Omega_k=0.0, Omega_r=0.0,
                           w0=w0, wa=wa, f_nu=f_nu)
        solution = odeint(ode_func, y0, a_points)
        D_solution = solution[:, 0]
        a_target = 1.0 / (1.0 + z_arr)
        return jnp.interp(a_target, a_points, D_solution)

    def compute(self, state, params_values):
        As = params_values["As"]
        ns = params_values["ns"]
        H0 = params_values["H0"]
        w0 = params_values["w"]
        wa = params_values["wa"]
        ombh2 = params_values["ombh2"]
        omch2 = params_values["omch2"]
        mnu = params_values["mnu"]

        h = H0 / 100.0
        n_species = 3
        mnu_per_species = mnu / n_species

        Omega_cb = (omch2 + ombh2) / h**2
        Omega_r_photon = _OMEGA_GAMMA_H2 / h**2
        y0_nu = mnu_per_species / _T_NU0_EV
        F_y0 = neutrino_density_ratio(y0_nu, self.gl_nu_nodes, self.gl_nu_weights)
        Omega_nu_0 = n_species * (_OMEGA_NU_REL_H2_PER_SPECIES / h**2) * F_y0
        Omega_m = Omega_cb + Omega_nu_0
        Omega_Lambda = 1.0 - Omega_cb - Omega_nu_0 - Omega_r_photon
        f_nu = Omega_nu_0 / Omega_m

        # 1. Compute chi_w0wa(0 -> z_LSS) and chi_w0wa(0 -> z_i)
        E_z_w0wa = self._build_E_z_func(
            Omega_cb, Omega_Lambda, Omega_r_photon, mnu_per_species, h, w0, wa
        )
        z_all = jnp.concatenate([self.z, jnp.array([self.z_lss])])
        chi_all, _ = comoving_distance_integral_full(z_all, E_z_w0wa, self.n_int_chi)
        chi_zi = chi_all[:-1]
        chi_zlss = chi_all[-1]
        chi_target = chi_zlss - chi_zi  # chi(z_i -> z_LSS) in w0wa

        # 2. Newton root-finding: for each z_i, find w s.t. chi_wCDM(z_i->z_LSS; w) = chi_target(z_i)
        def newton_one_z(z_i, chi_tgt):
            """Find w_equiv for one redshift via Newton's method."""
            w_curr = w0  # initial guess

            def newton_step(i, w):
                def chi_of_w(ww):
                    return self._chi_z_to_zlss(
                        z_i, ww, Omega_cb, Omega_Lambda,
                        Omega_r_photon, mnu_per_species, h
                    )
                residual = chi_of_w(w) - chi_tgt
                dchi_dw = jax.grad(chi_of_w)(w)
                return w - residual / dchi_dw

            w_equiv = jax.lax.fori_loop(0, self.n_newton, newton_step, w_curr)
            return w_equiv

        w_equiv_z = jax.vmap(newton_one_z)(self.z, chi_target)

        # 3. Amplitude matching: sigma8 correction
        # D_w0wa(z) / D_wCDM(z; w=w0) ratio
        D_w0wa = self._compute_D_unnorm(Omega_m, Omega_Lambda, f_nu, w0, wa, self.z)
        D_wcdm_w0 = self._compute_D_unnorm(Omega_m, Omega_Lambda, f_nu, w0, 0.0, self.z)
        growth_ratio = D_w0wa / D_wcdm_w0

        if self.sigma8_emu is not None:
            # sigma8_wCDM(z; w=w0) from emulator
            cosmo_params_w0 = jnp.array([
                As, ns, H0, w0, ombh2, omch2, jnp.log10(mnu),
            ])
            cparam_grid_w0 = jnp.zeros((self.nz, 8))
            cparam_grid_w0 = cparam_grid_w0.at[:, :-1].set(cosmo_params_w0)
            cparam_grid_w0 = cparam_grid_w0.at[:, -1].set(self.z)
            sigma8_wcdm_w0 = self.sigma8_emu.predict(cparam_grid_w0)[:, 0]

            # sigma8_w0wa(z) ~ sigma8_wCDM(z; w=w0) * growth_ratio
            sigma8_w0wa = sigma8_wcdm_w0 * growth_ratio

            # sigma8_wCDM(z; w_equiv) from emulator with z-dependent w
            cparam_grid_equiv = jnp.zeros((self.nz, 8))
            cparam_grid_equiv = cparam_grid_equiv.at[:, 0].set(As)
            cparam_grid_equiv = cparam_grid_equiv.at[:, 1].set(ns)
            cparam_grid_equiv = cparam_grid_equiv.at[:, 2].set(H0)
            cparam_grid_equiv = cparam_grid_equiv.at[:, 3].set(w_equiv_z)
            cparam_grid_equiv = cparam_grid_equiv.at[:, 4].set(ombh2)
            cparam_grid_equiv = cparam_grid_equiv.at[:, 5].set(omch2)
            cparam_grid_equiv = cparam_grid_equiv.at[:, 6].set(jnp.log10(mnu))
            cparam_grid_equiv = cparam_grid_equiv.at[:, 7].set(self.z)
            sigma8_wcdm_equiv = self.sigma8_emu.predict(cparam_grid_equiv)[:, 0]

            As_equiv_z = As * (sigma8_w0wa / sigma8_wcdm_equiv) ** 2
        else:
            # Without sigma8 emulator, use growth ODE ratio for amplitude correction
            # For each z_i, compute D_wCDM(z; w_equiv(z_i))
            def D_at_z_with_w(w_i, z_i):
                return self._compute_D_unnorm(Omega_m, Omega_Lambda, f_nu, w_i, 0.0,
                                              jnp.atleast_1d(z_i))[0]
            D_wcdm_equiv = jax.vmap(D_at_z_with_w)(w_equiv_z, self.z)
            As_equiv_z = As * (D_w0wa / D_wcdm_equiv) ** 2

        state["w_equiv_z"] = w_equiv_z
        state["As_equiv_z"] = As_equiv_z
        state["z_equiv"] = self.z

        return state


def build_equiv_cparam_grid(params_values, z, state, As_key="As", scale_As=1.0,
                            input_param_order=None):
    """Build cparam_grid, using z-dependent w/As if spectral equivalence is active.

    Args:
        params_values: Dict of parameter values.
        z: Array of redshifts for this module.
        state: Pipeline state dict.
        As_key: Parameter name for As (default "As").
        scale_As: Multiplicative factor for As (e.g. 1e9 for expansion history).

    Returns:
        cparam_grid: Array of shape (nz, 8) with [As, ns, H0, w, ombh2, omch2, log10(mnu), z].
    """
    # If the emulator accepts wa directly, bypass spectral equivalence
    if input_param_order is not None and "wa" in input_param_order:
        return build_equiv_cparam_grid_custom_order(
            params_values, z, state, input_param_order
        )

    nz = len(z)
    As_val = params_values[As_key] * scale_As

    if "w_equiv_z" not in state:
        cosmo_params = jnp.array([
            As_val,
            params_values["ns"],
            params_values["H0"],
            params_values["w"],
            params_values["ombh2"],
            params_values["omch2"],
            jnp.log10(params_values["mnu"]),
        ])
        cparam_grid = jnp.zeros((nz, len(cosmo_params) + 1))
        cparam_grid = cparam_grid.at[:, :-1].set(cosmo_params)
        cparam_grid = cparam_grid.at[:, -1].set(z)
        return cparam_grid

    # Spectral equivalence: interpolate w_equiv, As_equiv onto this module's z grid
    w_equiv = jnp.interp(z, state["z_equiv"], state["w_equiv_z"])
    As_equiv = jnp.interp(z, state["z_equiv"], state["As_equiv_z"])
    As_equiv = As_equiv * scale_As  # apply same scaling

    cparam_grid = jnp.zeros((nz, 8))
    cparam_grid = cparam_grid.at[:, 0].set(As_equiv)
    cparam_grid = cparam_grid.at[:, 1].set(params_values["ns"])
    cparam_grid = cparam_grid.at[:, 2].set(params_values["H0"])
    cparam_grid = cparam_grid.at[:, 3].set(w_equiv)
    cparam_grid = cparam_grid.at[:, 4].set(params_values["ombh2"])
    cparam_grid = cparam_grid.at[:, 5].set(params_values["omch2"])
    cparam_grid = cparam_grid.at[:, 6].set(jnp.log10(params_values["mnu"]))
    cparam_grid = cparam_grid.at[:, 7].set(z)
    return cparam_grid


def build_equiv_cparam_grid_custom_order(params_values, z, state, input_param_order):
    """Build cparam_grid for emulators with custom parameter ordering.

    Used by IA emulators and RedshiftSpaceBiasedTracerSpectra whose input_param_order
    may differ from the standard [As, ns, H0, w, ombh2, omch2, logmnu, z].

    Args:
        params_values: Dict of parameter values.
        z: Array of redshifts for this module.
        state: Pipeline state dict.
        input_param_order: List of parameter names (last must be 'z').

    Returns:
        cparam_grid: Array of shape (nz, len(input_param_order)).
    """
    nz = len(z)
    # Parameter name mapping for common aliases
    param_map = {"logmnu": lambda pv: jnp.log10(pv["mnu"])}

    cosmo_params = jnp.array([
        param_map[p](params_values) if p in param_map else params_values[p]
        for p in input_param_order[:-1]
    ])

    if "wa" in input_param_order or "w_equiv_z" not in state:
        cparam_grid = jnp.zeros((nz, len(cosmo_params) + 1))
        cparam_grid = cparam_grid.at[:, :-1].set(cosmo_params)
        cparam_grid = cparam_grid.at[:, -1].set(z)
        return cparam_grid

    # With spectral equivalence: substitute w and As with z-dependent equivalents
    w_equiv = jnp.interp(z, state["z_equiv"], state["w_equiv_z"])
    As_equiv = jnp.interp(z, state["z_equiv"], state["As_equiv_z"])

    cparam_grid = jnp.zeros((nz, len(input_param_order)))
    for col, p in enumerate(input_param_order):
        if p == "z":
            cparam_grid = cparam_grid.at[:, col].set(z)
        elif p == "w":
            cparam_grid = cparam_grid.at[:, col].set(w_equiv)
        elif p == "As":
            cparam_grid = cparam_grid.at[:, col].set(As_equiv)
        elif p == "logmnu":
            cparam_grid = cparam_grid.at[:, col].set(jnp.log10(params_values["mnu"]))
        else:
            cparam_grid = cparam_grid.at[:, col].set(params_values[p])

    return cparam_grid
