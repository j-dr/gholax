import jax.numpy as jnp
import numpy as np
from interpax import interp2d
from jax.lax import scan
from jax.scipy.integrate import trapezoid
from ...theory.spline import spline_func_vec
from ...util.likelihood_module import LikelihoodModule

required_components = {
    "c_dd": [
        (("w_d", "w_d"), ("p_gg",0), "zeff_w_d"),
        (("w_mag", "w_d"), ("p_gm",1), "zeff_w_d"),
        (("w_d", "w_mag"), ("p_gm",0), "zeff_w_d"),
        (("w_mag", "w_mag"), ("p_mm",0), "z_limber"),
    ],
    "c_dk": [
        (("w_d_dk", "w_k"), ("p_gm",0), "zeff_w_d"),
        (("w_mag_dk", "w_k"), ("p_mm",0), "z_limber"),
        (("w_d_dk", "w_ia"), ("p_gi",0), "zeff_w_d"),
        (("w_mag_dk", "w_ia"), ("p_mi",1), "z_limber"),
    ],
    "c_dcmbk": [
        (("w_d_cmbk", "w_cmbk"), ("p_gm",0), "zeff_w_d"),
        (("w_mag_dcmbk", "w_cmbk"), ("p_mm",0), "z_limber"),
    ],
    "c_kk": [
        (("w_k", "w_k"), ("p_mm",0), "z_limber"),
        (("w_ia", "w_k"), ("p_mi",0), "z_limber"),
        (("w_k", "w_ia"), ("p_mi",1), "z_limber"),
        (("w_ia", "w_ia"), ("p_ii_ee",0), "z_limber"),
    ],
    "c_bb": [(("w_ia", "w_ia"), ("p_ii_bb",0), "z_limber")],
    "c_kcmbk": [
        (("w_k", "w_cmbk"), ("p_mm",0), "z_limber"),
        (("w_ia", "w_cmbk"), ("p_mi",0), "z_limber"),
    ],
    "c_cmbkcmbk": [(("w_cmbk", "w_cmbk"), ("p_mm",0), "z_limber")],
}


spec_enum = {
    "p_gg": 0,
    "p_gm": 1,
    "p_mm": 2,
    "p_mi": 3,
    "p_gi": 4,
    "p_ii_ee": 5,
    "p_ii_bb": 6,
}


class Limber(LikelihoodModule):
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
        n_ell=200,
        l_max=3001,
        **config,
    ):
        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.nk = nk
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.kmax = kmax
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.z_proj = jnp.linspace(zmin_proj, zmax_proj, nz_proj)
        self.z_pk = jnp.linspace(zmin_pk, zmax_pk, nz_pk)
        self.nz_proj = nz_proj
        self.nz_pk = nz_pk
        self.n_ell = n_ell
        self.l_max = l_max
        self.ell = jnp.logspace(1, jnp.log10(self.l_max), self.n_ell)
        self.k_cutoff = config.get("k_cutoff", None)
        self.integration_method = config.get("integration_method", "gl_quad")
        self.non_parametric_growth = config.get("non_parametric_growth", False)
        self.non_parametric_growth_spline_N = config.get("non_parametric_growth_spline_N", 3)
        self.non_parametric_growth_spline_Delta = config.get("non_parametric_growth_spline_Delta", 3)

        if self.k_cutoff == "None":
            self.k_cutoff = None

        self.no_ia = config.get("no_ia", False)
        self.magnification_x_ia = config.get("include_magnification_x_ia", False)
        self.interpolation_order = config.get("interpolation_order", "cubic")

        self.all_spectra = {}

        if (not self.magnification_x_ia) & (not self.no_ia):
            required_components["c_dk"] = [
                (("w_d_dk", "w_k"), ("p_gm",0), "zeff_w_d"),
                (("w_mag_dk", "w_k"), ("p_mm",0), "z_limber"),
                (("w_d_dk", "w_ia"), ("p_gi",0), "zeff_w_d"),
            ]

        if self.no_ia:
            required_components["c_dk"] = [
                (("w_d_dk", "w_k"), ("p_gm",0), "zeff_w_d"),
                (("w_mag_dk", "w_k"), ("p_mm",0), "z_limber"),
            ]

            required_components["c_kk"] = [
                (("w_k", "w_k"), ("p_mm",0), "z_limber"),
            ]

            required_components["c_bb"] = []

        if "c_dd" in self.spectrum_info:
            if self.spectrum_info["c_dd"]["use_cross"]:
                required_components["c_dd"] = [
                    (("w_d", "w_d"), ("p_gg",0), "zeff_w_d"),
                    (("w_mag", "w_d"), ("p_gm",1), "zeff_w_d"),
                    (("w_d", "w_mag"), ("p_gm",0), "zeff_w_d"),
                    (("w_mag", "w_mag"), ("p_mm",0), "z_limber"),
                ]

        self.output_requirements = {}
        self.all_spectra = {}

        for t in self.spectrum_types:
            self.all_spectra[t] = []
            self.output_requirements[t] = []

            for (w_i, w_j), (p, td_), zfield_ in required_components[t]:
                self.output_requirements[f"{w_i}_{w_j}"] = [w_i, w_j, p]
                self.output_requirements[t].extend([w_i, w_j, p])
                
        if self.non_parametric_growth:
            np_growth_params = [f'A_growth_{i}' for i in range(self.non_parametric_growth_spline_N)]
            self.indexed_params = {}
            for t in self.spectrum_types:
                self.output_requirements[t].extend(np_growth_params)
                self.indexed_params[t] = np.array([np_growth_params])

        for o in self.output_requirements:
            self.output_requirements[o] = list(np.unique(self.output_requirements[o]))
            


    def compute_c_l(self, state, spec_type):
        chi_z_proj = state["chi_z_limber"]
        k = (self.ell[:, np.newaxis] + 0.5) / chi_z_proj[None, :]
        log_kval = jnp.log10((k))
        
        def compute_component_c_l_chi(carry, x):
            w_i, w_j, s, zfield = x

            p_ij = interp2d(
                log_kval.reshape(-1),
                zfield.reshape(-1),
                self.logk,
                self.z_pk,
                s,
                extrap=0.0,
                method=self.interpolation_order,
            ).reshape(self.n_ell, self.nz_proj)

            if self.k_cutoff is not None:
                mask = k < self.k_cutoff
            else:
                mask = jnp.ones_like(k)

            c_l_chi = w_i * w_j * p_ij / state["chi_z_limber"][None, :] ** 2 * mask     
            if self.non_parametric_growth:
                A_growth_chi = state['A_growth_chi']
                c_l_chi = c_l_chi * (A_growth_chi[None, :] ** 2)

            return carry, c_l_chi

        for (w_i_, w_j_), (spectrum_, td_), zfield_ in required_components[spec_type]:
            w_i = state[w_i_]
            w_j = state[w_j_]
            s = state[spectrum_]
            zfield = state[zfield_]

            n_i = w_i.shape[0]

            if w_j.ndim < 3:
                n_j = w_j.shape[0]
            else:
                n_j = w_j.shape[1]

            if self.observed_data_vector.spectrum_info[spec_type]["use_cross"]:
                
                if w_i.ndim == 3:
                    w_i = w_i.reshape(n_i * n_j, -1)
                elif w_i.shape[0] != (n_i * n_j):
                    w_i = jnp.repeat(w_i, n_j, 0)

                if w_j.ndim == 3:
                    w_j = w_j.reshape(n_i * n_j, -1)
                elif w_j.shape[0] != (n_i * n_j):
                    w_j = jnp.tile(w_j, (n_i, 1))

                # if w_i_==w_j_ or n_i!=n_j then figure out 
                # how to reshape based on whether s.shape matches n_i or n_j
                if s.ndim == 4:
                    s = s.reshape(n_i * n_j, self.nk, self.nz_pk)
                elif (w_i_ == w_j_) | (n_i != n_j):
                    if n_i == s.shape[0]:
                        s = jnp.repeat(s, n_j, 0)
                    elif n_j == s.shape[0]:
                        s = jnp.tile(s, (n_i, 1, 1))
                    elif s.shape[0] == 1:
                        s = jnp.tile(s, (n_i * n_j, 1, 1))
                else:
                    if td_ == 0:
                        s = jnp.repeat(s, n_j, 0)
                    elif td_ == 1:
                        s = jnp.tile(s, (n_i, 1, 1))                        

                if zfield.shape[0] == self.nz_proj:
                    zfield = jnp.tile(zfield[None, :], (n_i * n_j, self.n_ell, 1))
                elif zfield.shape[0] == n_i:
                    zfield = jnp.repeat(
                        zfield, n_j * self.n_ell * self.nz_proj, 0
                    ).reshape(n_i * n_j, self.n_ell, self.nz_proj)
                elif zfield.shape[0] == n_j:
                    zfield = jnp.tile(
                        zfield[:, None, None], (n_i, self.n_ell, self.nz_proj)
                    )

            else:
                if n_i != n_j:
                    if w_i.shape[0] == 1:
                        w_i = jnp.tile(w_i, (n_i * n_j, 1))
                    if w_j.shape[0] == 1:
                        w_j = jnp.tile(w_j, (n_i * n_j, 1))
                    if s.shape[0] == 1:
                        s = jnp.tile(s, (n_i * n_j, 1, 1))
                    if zfield.shape[0] == self.nz_proj:
                        zfield = jnp.tile(zfield[None, :], (n_i * n_j, self.n_ell, 1))
                else:
                    if zfield.shape[0] == self.nz_proj:
                        zfield = jnp.tile(zfield[None, :], (n_i, self.n_ell, 1))
                    else:
                        zfield = jnp.tile(
                            zfield[:, None, None], (1, self.n_ell, self.nz_proj)
                        )
                    if s.shape[0] == 1:
                        s = jnp.tile(s, (n_i, 1, 1))

            xs = [w_i, w_j, s, zfield]
            _, c_l_chi_w_i_w_j = scan(compute_component_c_l_chi, None, xs)
            state[f"{w_i_}_{w_j_}"] = c_l_chi_w_i_w_j
            if spec_type in state:
                state[spec_type] += c_l_chi_w_i_w_j
            else:
                state[spec_type] = c_l_chi_w_i_w_j

        if self.integration_method == "trapezoid":
            state[spec_type] = trapezoid(
                state[spec_type], x=state["chi_z_limber"], axis=-1
            )
        elif self.integration_method == "gl_quad":
            state[spec_type] = jnp.sum(
                state[spec_type] * state["gl_scaled_weights"][None, :], axis=-1
            )

        return state

    def compute(self, state, params_values):
        if self.non_parametric_growth:
            param_vec = jnp.array(list(params_values.values()))
            a_growth_chi = spline_func_vec(
                state['z_limber'],
                param_vec[self.param_indices[self.spectrum_types[0]]].T,
                0.0,
                Delta=self.non_parametric_growth_spline_Delta,
            )[0]
            state['A_growth_chi'] = a_growth_chi

        for t in self.spectrum_types:
            state = self.compute_c_l(state, t)

        return state
