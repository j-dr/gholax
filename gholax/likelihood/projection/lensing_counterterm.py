import warnings

import jax.numpy as jnp
import numpy as np
from interpax import interp1d, interp2d
from jax import jacobian, vmap
from jax.lax import scan, switch, select_n
from jax.scipy.integrate import trapezoid
from jax.scipy.special import factorial
from scipy.special import roots_legendre
from ...util.likelihood_module import LikelihoodModule
from .limber import required_components


class LensingCounterterm(LikelihoodModule):
    def __init__(
        self,
        observed_data_vector,
        spectrum_types,
        spectrum_info,
        zmin_proj=None,
        zmax_proj=None,
        nz_proj=None,
        zmin_pk=None,
        zmax_pk=None,
        nz_pk=None,
        kmin=None,
        kmax=None,
        nk=None,
        k_cutoff=None,
        mean_model="dmo",
        n_ell=200,
        l_max=3001,
        **config,
    ):
        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.kmax_emu = kmax #config.get("kmax_emu", 4.0)
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(self.kmax_emu), nk)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(self.kmax_emu), nk)
        self.z_proj = jnp.linspace(zmin_proj, zmax_proj, nz_proj)
        self.z_pk = jnp.linspace(zmin_pk, zmax_pk, nz_pk)
        self.nz_proj = nz_proj
        self.nz_pk = nz_pk
        self.n_ell = n_ell
        self.l_max = l_max
        self.ell = jnp.logspace(1, jnp.log10(self.l_max), self.n_ell)
        self.hubble_radius = 2997.925
        self.k_cutoff = k_cutoff
        self.mean_model = mean_model
        self.magnification_x_ia = config.get("include_magnification_x_ia", False)
        self.lensing_counterterm_order = config.get("lensing_counterterm_order", 0)
        self.analytic_zchi_derivatives = config.get("analytic_zchi_derivatives", True)
        self.power_law_extrapolation = config.get("power_law_extrapolation", 0)
        self.integration_method = config.get("integration_method", "gl_quad")
        if self.integration_method == "gl_quad":
            nk_gl = config.get("nk_gl", 50)
            nodes, weights = roots_legendre(nk_gl)
            self.k_nodes = 0.5 * (self.kmax_emu - self.k_cutoff) * (nodes + 1) + self.k_cutoff
            self.quad_weights = 0.5 * (self.kmax_emu - self.k_cutoff) * weights
            
        if type(self.power_law_extrapolation) is not bool:
            try:
                self.power_law_extrapolation = float(self.power_law_extrapolation)
            except ValueError:
                raise ValueError(
                    "power_law_extrapolation must be a boolean or a float."
                )

        self.all_spectra = {}
        if not self.magnification_x_ia:
            required_components["c_dk"] = [
                (("w_d_dk", "w_k"), "p_gm", "zeff_w_d"),
                (("w_mag_dk", "w_k"), "p_mm", "z_limber"),
                (("w_d_dk", "w_ia"), "p_gi", "zeff_w_d"),
            ]

        self.lensing_kernels = ["w_mag", "w_mag_dk", "w_mag_dcmbk", "w_k", "w_cmbk"]

        self.output_requirements = {}
        self.lensing_counterterms = []
        for N in range(2, self.lensing_counterterm_order + 2):
            for o in range(N - 1):
                self.lensing_counterterms.append(f"sigma_{N}_{o}")
                self.output_requirements[f"sigma_{N}_{o}_emu"] = ["p_mm"]
                self.output_requirements[f"sigma_{N}_{o}_emu"].extend(
                    self.lensing_counterterms
                )

        self.all_spectra = {}
        self.indexed_params = {}

        for t in self.spectrum_types:
            self.output_requirements[f"{t}_w_lensing_ct"] = []
            for (w_i, w_j), p, _ in required_components[t]:
                if (w_i in self.lensing_kernels) & (w_j in self.lensing_kernels):
                    self.output_requirements[f"{t}_w_lensing_ct"].extend(
                        self.lensing_counterterms
                    )
                    for n in range(1, self.lensing_counterterm_order + 1):
                        self.output_requirements[f"{w_i}_{n}"] = [w_i]
                        self.output_requirements[f"{w_j}_{n}"] = [w_j]
                    if "mag" in w_i:
                        self.indexed_params[w_i] = []
                        for l in range(self.observed_data_vector.nz_d.shape[0]):
                            if l in self.spectrum_info[t]["bins0"]:
                                self.indexed_params[w_i].append(f"smag_{l}")
                            else:
                                self.indexed_params[w_i].append("NA")

        self.indexed_params["all"] = []
        for N in range(self.lensing_counterterm_order):
            self.indexed_params["all"].append([])
            for o in range(self.lensing_counterterm_order):
                if o <= N:
                    self.indexed_params["all"][N].append(f"sigma_{N + 2}_{o}")
                else:
                    self.indexed_params["all"][N].append("NA")

        for k in self.indexed_params:
            self.indexed_params[k] = np.array(self.indexed_params[k])
            if k != "all":
                self.indexed_params[k] = self.indexed_params[k][:, None]

        for o in self.output_requirements:
            self.output_requirements[o] = list(np.unique(self.output_requirements[o]))

    def sigma_N_o(self, state, lk):
        k = 10**lk
        if self.integration_method == "gl_quad":
            mask = jnp.ones_like(k)
        else:
            mask = (self.k_cutoff < k) * (k <= self.kmax_emu)

        dz = 0.01
        eps = 0.001
        D_grid = state["sigma8_z"] / state["sigma8_z"][0]

        if self.mean_model == "dmo":
            p_mm = state["p_11_real_space_bias_grid"]
        else:
            p_mm = state["p_mm"][0, ...]
        p_mm = p_mm + 1e-8

        log_p_mm_normalized = jnp.log(p_mm / D_grid**2)

        def compute_pk(z):
            D_z = interp1d(z, self.z_pk, D_grid, extrap=0)
            return (
                jnp.exp(
                    interp2d(
                        lk,
                        z,
                        self.logk,
                        self.z_pk,
                        log_p_mm_normalized,
                        extrap=self.power_law_extrapolation,
                    )
                )
                * D_z**2
            )

        max_points = self.lensing_counterterm_order + 1
        z_all = eps + jnp.arange(max_points) * dz
        pk_all = jnp.stack([compute_pk(z) for z in z_all])

        factorials = jnp.array(
            [factorial(o) for o in range(self.lensing_counterterm_order)]
        )

        from scipy.special import comb

        fd_coeffs = jnp.zeros((self.lensing_counterterm_order, max_points))
        for o in range(self.lensing_counterterm_order):
            for i in range(min(o + 1, max_points)):
                if o == 0:
                    fd_coeffs = fd_coeffs.at[0, 0].set(1.0)
                else:
                    fd_coeffs = fd_coeffs.at[o, i].set(
                        (-1) ** (o - i) * comb(o, i, exact=True)
                    )

        N_all = jnp.arange(2, 2 + self.lensing_counterterm_order)
        k_powers_all = jnp.array([k ** (-N) for N in N_all])
        
        if self.integration_method == "trapezoid":
            vint = vmap(lambda integrand: trapezoid(integrand, x=k))
        else:
            vint = vmap(lambda integrand: jnp.sum(integrand * self.quad_weights))

        def scan_body(carry, o):
            sigma_N_o_current = carry

            coeff_mask = (jnp.arange(max_points) < (o+1)).astype(jnp.float32)
            coeffs = fd_coeffs[o] * coeff_mask

            deriv_current = jnp.dot(coeffs, pk_all)

            norm_factor = (dz**o) * (self.hubble_radius**o)
            deriv_current = deriv_current / norm_factor

            integrand_base = deriv_current / factorials[o] * mask
            integrands = k_powers_all * integrand_base[jnp.newaxis, :]
            sigma_values = vint(integrands)

            valid_mask = (N_all >= (2 + o)).astype(jnp.float32)
            sigma_values = sigma_values * valid_mask
            sigma_N_o_current = sigma_N_o_current.at[:, o].set(sigma_values)

            return sigma_N_o_current, None

        sigma_N_o_emu = jnp.zeros(
            (self.lensing_counterterm_order, self.lensing_counterterm_order)
        )

        o_range = jnp.arange(self.lensing_counterterm_order)
        sigma_N_o_final, _ = scan(scan_body, sigma_N_o_emu, o_range)

        return sigma_N_o_final

    def w_n(self, w, state):
        def dzn(n):
            return select_n(
                n,
                1 / self.hubble_radius,
                1.5 * state["omegam"] / self.hubble_radius**2,
                3 * state["omegam"] / self.hubble_radius**3,
                3
                * state["omegam"]
                * (1 + 1.5 * state["omegam"])
                / self.hubble_radius**4,
                )
            
        def f_n(n, xs):
            w_i_n = switch(
                n,
                [
                    lambda n: jnp.ones_like(state[f"chi_inv_eff_{w}"]),
                    lambda n: 1 / self.hubble_radius - state[f"chi_inv_eff_{w}"],
                    lambda n: (
                        dzn(n - 1) - (n) * state[f"chi_inv_eff_{w}"] * dzn(n - 2)
                    )
                    / factorial(n), #n gets clamped to last value so for n>=2 we get this result
                ],
                n,
            )
            n = n + 1
            return n, w_i_n

        _, wn = scan(f_n, 0, jnp.arange(self.lensing_counterterm_order))

        return wn.T

    def w_n_autodiff(self, w, state):
        w_n = []
        for n in range(1, self.lensing_counterterm_order + 1):
            if n == 1:

                def dzdchi_n(chi):
                    return interp1d(chi, state["chi_z_limber"], state["z_limber"])

                w_n.append(1)

            elif n == 2:
                dzdchi_nm1 = dzdchi_n
                dzdchi_n = jacobian(dzdchi_n)
                w_n.append(
                    dzdchi_n(state["chi_z_limber"][0]) - state[f"chi_inv_eff_{w}"]
                )

            else:
                if n > 3:
                    warnings.warn(
                        "Potential loss of derivative accuracy for n>3 due to spline order."
                    )
                dzdchi_nm1 = dzdchi_n
                dzdchi_n = jacobian(dzdchi_nm1)
                w_n.append(
                    (
                        dzdchi_n(state["chi_z_limber"][0])
                        - (n - 1)
                        * state[f"chi_inv_eff_{w}"]
                        * dzdchi_nm1(state["chi_z_limber"][0])
                    )
                    / factorial(n - 1)
                )
        return jnp.array(w_n).T

    def compute(self, state, params_values):
        param_vec = jnp.array(list(params_values.values()))

        def f(carry, ell):
            return (carry + 1, (ell + 0.5) ** (carry + 1))

        if self.lensing_counterterm_order > 0:
            if self.integration_method == "gl_quad":
                log_kval = jnp.log10(self.k_nodes)
            else:
                log_kval = jnp.log10(self.k)
            sigma_N_o_emu = self.sigma_N_o(state, log_kval)
            state["sigma_N_o_emu"] = sigma_N_o_emu
            sigma_N_o = param_vec[self.param_indices["all"]]

            _, ell_N = scan(
                f,
                0,
                jnp.tile(self.ell, self.lensing_counterterm_order).reshape(
                    self.lensing_counterterm_order, -1
                ),
            )
            N, n, m, o = jnp.meshgrid(
                jnp.arange(self.lensing_counterterm_order),
                jnp.arange(self.lensing_counterterm_order),
                jnp.arange(self.lensing_counterterm_order),
                jnp.arange(self.lensing_counterterm_order),
                indexing="ij",
            )

        for t in self.spectrum_types:
            add_ct = False
            if self.lensing_counterterm_order > 0:
                for (w_i, w_j), _, _ in required_components[t]:
                    if (w_i in self.lensing_kernels) & (w_j in self.lensing_kernels):
                        w_i_n = self.w_n(w_i, state)
                        w_j_n = self.w_n(w_j, state)
                        c_l = state[t]

                        n_i = w_i_n.shape[0]
                        n_j = w_j_n.shape[0]
                        n_c_l = c_l.shape[0]
                        if n_i != n_c_l:
                            w_i_n = jnp.repeat(w_i_n, n_j, 0)
                        if n_j != n_c_l:
                            if n_i == n_c_l:
                                w_j_n = jnp.tile(w_j_n, (n_i // n_j, 1))
                            else:
                                w_j_n = jnp.tile(w_j_n, (n_i, 1))

                        if self.mean_model == "dmo":
                            c_l_uv = jnp.einsum(
                                "in,im,No,Nl->ilNnmo",
                                w_i_n,
                                w_j_n,
                                sigma_N_o_emu * sigma_N_o,
                                ell_N,
                            )
                        elif self.mean_model == "ct":
                            print(
                                "CT mean model should only be used for illustrative purposes.",
                                flush=True,
                            )
                            c_l_uv = jnp.einsum(
                                "in,im,No,Nl->ilNnmo",
                                w_i_n,
                                w_j_n,
                                sigma_N_o_emu * sigma_N_o,
                                ell_N,
                            )
                        elif self.mean_model == "zero":
                            c_l_uv = jnp.einsum(
                                "in,im,No,Nl->ilNnmo",
                                w_i_n,
                                w_j_n,
                                sigma_N_o_emu * (1 + sigma_N_o),
                                ell_N,
                            )

                        c_l_uv = (
                            jnp.sum(c_l_uv, axis=(2, 3, 4, 5), where=(N == (n + m + o)))
                            * (3 / 2 * state["omegam"] / self.hubble_radius**2) ** 2
                        )

                        if "mag" in w_i:
                            smag = param_vec[self.param_indices[w_i]][:, 0]
                            if smag.shape[0] != c_l_uv.shape[0]:
                                c_l_uv = (
                                    c_l_uv.reshape(smag.shape[0], -1, self.n_ell)
                                    * (5 * smag[:, None, None] - 2)
                                ).reshape(-1, self.n_ell)
                            else:
                                c_l_uv = c_l_uv * (5 * smag[:, None] - 2)

                        if "mag" in w_j:
                            smag = param_vec[self.param_indices[w_j]][:, 0]
                            if smag.shape[0] != c_l_uv.shape[0]:
                                c_l_uv = (
                                    c_l_uv.reshape(smag.shape[0], -1, self.n_ell)
                                    * (5 * smag[:, None, None] - 2)
                                ).reshape(-1, self.n_ell)
                            else:
                                c_l_uv = c_l_uv * (5 * smag[:, None] - 2)

                        state[f"{t}_w_lensing_ct"] = c_l + c_l_uv
                        add_ct = True

            # if we didn't add a counterterm, just relabel c_l for consistency
            if not add_ct:
                state[f"{t}_w_lensing_ct"] = state[t]

        return state
