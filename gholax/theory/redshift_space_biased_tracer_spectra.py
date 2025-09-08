import jax
import jax.numpy as jnp
import numpy as np
from interpax import interp1d, interp2d
from jax.lax import scan

from ..util.likelihood_module import LikelihoodModule
from .emulator import MultiSpectrumEmulator

try:
    from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
except:
    pass


class RedshiftSpaceBiasedTracerSpectra(LikelihoodModule):
    def __init__(
        self,
        zeff,
        hz_fid,
        chiz_fid,
        zmin=0,
        zmax=2.3,
        nz=50,
        kmin=1e-3,
        kmax=0.601,
        nk=200,
        **config,
    ):
        self.nk = nk
        self.k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.logk = np.linspace(np.log10(kmin), np.log10(kmax), nk)

        self.nz = nz
        self.zmin = zmin
        self.zmax = zmax
        self.z = np.linspace(zmin, zmax, nz)

        self.hz_fid = np.array(hz_fid)
        self.chiz_fid = np.array(chiz_fid)
        self.zeff = zeff
        self.nzeff = len(self.zeff)

        self.use_emulator = bool(config.get("use_emulator", False))
        self.n_ell = 3
        self.save_noap_spectra = config.get("save_noap_spectra", False)
        self.n_ell_noap = 4

        if self.save_noap_spectra:
            ell_max = 2 * (self.n_ell_noap - 1)
        else:
            ell_max = 2 * (self.n_ell - 1)

        self.ngauss = int(
            np.ceil((ell_max + 1) / 2) + 2
        )  # extra 2 because P(k,mu) is not a polynomial in mu.

        self.nus, self.ws = np.polynomial.legendre.leggauss(2 * self.ngauss)
        self.nus = jnp.array(self.nus)
        self.ws = jnp.array(self.ws)
        self.L = jnp.array(
            [
                np.polynomial.legendre.Legendre((0,) * li * 2 + (1,))(self.nus)
                for li in range(self.n_ell_noap)
            ]
        )

        self.output_requirements = {}

        if self.use_emulator:
            self.emulator_file_names = config["emulator_file_names"]
            self.emulators = {}
            self.input_param_order = [
                "As",
                "ns",
                "omch2",
                "ombh2",
                "H0",
                "w",
                "logmnu",
                "z",
            ]
            self.output_requirements["p_ij_ell_redshift_space_bias_grid"] = [
                "As",
                "ns",
                "omch2",
                "ombh2",
                "H0",
                "w",
                "logmnu",
                "e_z",
                "chi_z",
            ]

            for ell in range(self.n_ell_noap):
                self.emulators[ell] = MultiSpectrumEmulator(
                    self.emulator_file_names[ell], self.input_param_order
                )
            self.n_spec = self.emulators[0].n_spec
        else:
            self.output_requirements["p_ij_ell_redshift_space_bias_grid"] = [
                "f_z",
                "Pcb_lin_z",
                "e_z",
                "chi_z",
            ]
        self.kIR = config.get("kIR", 0.2)
        self.interpolation_order = config.get("interpolation_order", "cubic")

    def apply_ap(self, state):
        h_z = state["e_z_limber"]
        chi_z = state["chi_z_limber"]
        h_z_eff = jnp.interp(self.zeff, state["z_limber"], h_z)
        chi_z_eff = jnp.interp(self.zeff, state["z_limber"], chi_z)
        apar, aperp = self.hz_fid / h_z_eff, self.chiz_fid / chi_z_eff

        pkell_true = interp1d(
            self.zeff,
            self.z,
            state["p_ij_ell_no_ap_redshift_space_bias_grid"].T.reshape(self.nz, -1),
            extrap=False,
            method="linear",
        ).reshape(self.nzeff, self.nk, self.n_ell_noap, 13)

        pknu_true = jnp.sum(
            self.L.T[:, None, None, :, None] * pkell_true[None, ...], axis=3
        )
        pknu_true = jnp.einsum("nzks->nksz", pknu_true)

        fac = jnp.sqrt(1 + self.nus[None, :] ** 2 * ((aperp / apar)[:, None] ** 2 - 1))
        k_apfac = fac / aperp[:, None]
        nu_true = jnp.tile(
            self.nus[None, :] * (aperp / apar)[:, None] / fac, (self.nk, 1, 1)
        ).T
        vol_fac = apar * aperp**2
        ktrue = k_apfac[:, :, None] * self.k[None, None, :]
        nu_true = jnp.einsum("nzk->nkz", nu_true)
        ktrue = jnp.einsum("znk->nkz", ktrue)

        def calc_pknu_obs(i, j):
            return (
                interp2d(
                    nu_true[:, :, j].reshape(-1),
                    ktrue[:, :, j].reshape(-1),
                    self.nus,
                    self.k,
                    pknu_true[:, :, i, j],
                    extrap=True,
                )
                .reshape(2 * self.ngauss, self.nk)
                .T
            )

        pknu_obs = jax.vmap(
            jax.vmap(calc_pknu_obs, in_axes=(None, 0)), in_axes=(0, None)
        )(jnp.arange(13), jnp.arange(self.nzeff)).T
        pknu_obs_w_sct = jnp.zeros((2 * self.ngauss, self.nk, self.nzeff, 19))
        p_ij_ell = jnp.zeros((19, self.n_ell, self.nk, self.nzeff))

        pknu_obs_w_sct = pknu_obs_w_sct.at[..., :-7].set(pknu_obs[..., :-1])
        pknu_obs_w_sct = pknu_obs_w_sct.at[..., -7].set(ktrue**2 * pknu_obs[..., -1])
        pknu_obs_w_sct = pknu_obs_w_sct.at[..., -6].set(
            ktrue**2 * nu_true**2 * pknu_obs[..., -1]
        )
        pknu_obs_w_sct = pknu_obs_w_sct.at[..., -5].set(
            ktrue**2 * nu_true**4 * pknu_obs[..., -1]
        )
        pknu_obs_w_sct = pknu_obs_w_sct.at[..., -4].set(
            ktrue**2 * nu_true**6 * pknu_obs[..., -1]
        )

        pknu_obs_w_sct = pknu_obs_w_sct.at[..., -3].set(1)
        pknu_obs_w_sct = pknu_obs_w_sct.at[..., -2].set(ktrue**2 * nu_true**2)
        pknu_obs_w_sct = pknu_obs_w_sct.at[..., -1].set(ktrue**4 * nu_true**4)

        pknu_obs_w_sct = jnp.einsum("nkzs->nskz", pknu_obs_w_sct)

        p_ij_ell = p_ij_ell.at[:, 0, :, :].set(
            0.5
            * jnp.sum(
                (self.ws * self.L[0])[:, None, None, None] * pknu_obs_w_sct, axis=0
            )
            / vol_fac[None, None, :]
        )
        p_ij_ell = p_ij_ell.at[:, 1, :, :].set(
            2.5
            * jnp.sum(
                (self.ws * self.L[1])[:, None, None, None] * pknu_obs_w_sct, axis=0
            )
            / vol_fac[None, None, :]
        )
        p_ij_ell = p_ij_ell.at[:, 2, :, :].set(
            4.5
            * jnp.sum(
                (self.ws * self.L[2])[:, None, None, None] * pknu_obs_w_sct, axis=0
            )
            / vol_fac[None, None, :]
        )

        state["p_ij_ell_redshift_space_bias_grid"] = p_ij_ell

        return state

    def compute_emulator(self, state, params_values):
        cosmo_params = jnp.array(
            [params_values[p] for p in self.input_param_order[:-1]]
        )
        cparam_grid = jnp.zeros((self.nz, len(cosmo_params) + 1))
        cparam_grid = cparam_grid.at[:, :-1].set(cosmo_params)
        cparam_grid = cparam_grid.at[:, -1].set(self.z)

        logk_emu = jnp.log10(self.emulators[0].k)
        state["p_ij_ell_redshift_space_bias_grid"] = jnp.zeros(
            (self.n_spec, self.n_ell, self.nk, self.nz)
        )

        p_ij_l = []
        for l in range(self.n_ell_noap):
            p_ij_l.append(self.emulators[l].predict(cparam_grid))

        p_ij_l = jnp.array(p_ij_l).reshape(
            self.n_ell_noap * self.nz * self.n_spec, self.emulators[0].nk
        )

        def interp_spectrum(j):
            p = interp1d(
                self.logk,
                logk_emu,
                p_ij_l[j].T,
                extrap=0,
                method=self.interpolation_order,
            )
            return p

        # Process all spectra at once
        p = jax.vmap(interp_spectrum)(jnp.arange(p_ij_l.shape[0])).reshape(
            self.n_ell_noap, self.nz, self.n_spec, self.nk
        )
        state["p_ij_ell_no_ap_redshift_space_bias_grid"] = jnp.einsum("lzjk->jlkz", p)

        state = self.apply_ap(state)

        return state

    def compute_analytic(self, state, params_values):
        h_z = state["e_z_limber"]
        chi_z = state["chi_z_limber"]
        f_z = state["f_z"]
        k_lin = np.array(state["k_lin"])
        n_spec = 19

        h_z_eff = np.array(np.interp(self.zeff, state["z_limber"], h_z))
        chi_z_eff = np.array(np.interp(self.zeff, state["z_limber"], chi_z))
        f_z_eff = np.array(np.interp(self.zeff, state["z_pk"], f_z))

        if not self.save_noap_spectra:
            p_cb = np.array(
                [
                    np.interp(self.z, state["z_pk"], state["Pcb_lin_z"][:, i])
                    for i in range(k_lin.shape[0])
                ]
            ).T

            p_ij_ell = np.zeros((n_spec, self.n_ell, self.nk, self.nz))

            for i, z in enumerate(self.zeff):
                apar, aperp = (
                    self.hz_fid[i] / h_z_eff[i],
                    self.chiz_fid[i] / chi_z_eff[i],
                )
                model_lpt = LPT_RSD(
                    k_lin,
                    p_cb[i],
                    kIR=self.kIR,
                    use_Pzel=False,
                    cutoff=10,
                    extrap_min=-4,
                    extrap_max=3,
                    N=2000,
                    threads=1,
                    jn=5,
                )
                model_lpt.make_pltable(
                    np.float64(f_z_eff[i]),
                    kv=self.k,
                    apar=apar,
                    aperp=aperp,
                    ngauss=self.ngauss,
                )
                p_ij_ell[:, 0, :, i] = model_lpt.p0ktable.T
                p_ij_ell[:, 1, :, i] = model_lpt.p2ktable.T
                p_ij_ell[:, 2, :, i] = model_lpt.p4ktable.T
                state["p_ij_ell_redshift_space_bias_grid"] = p_ij_ell

        else:
            p_cb = state["Pcb_lin_z"]
            nus, ws = np.polynomial.legendre.leggauss(2 * self.ngauss)
            p_ij_ell_noap = np.zeros((self.n_ell_noap, self.nk, 13, self.nz))
            pknu_true = np.zeros((len(nus), self.nk, 13, self.nz))

            for i, z in enumerate(self.z):
                model_lpt = LPT_RSD(
                    k_lin,
                    p_cb[i],
                    kIR=self.kIR,
                    use_Pzel=False,
                    cutoff=10,
                    extrap_min=-4,
                    extrap_max=3,
                    N=2000,
                    threads=1,
                    jn=7,
                )
                model_lpt.make_pltable(
                    np.float64(f_z[i]),
                    kv=self.k,
                    apar=1,
                    aperp=1,
                    ngauss=self.ngauss,
                    nmax=10,
                )
                pknu_true[..., i] = model_lpt.pknutable[..., :-6]
                pknu_true[..., -1, i] /= self.k**2

            for li in range(self.n_ell_noap):
                Li = np.polynomial.legendre.Legendre((0,) * li * 2 + (1,))(nus)
                ell = 2 * li
                p_ij_ell_noap[li, :, :, :] = (
                    (2 * ell + 1)
                    / 2
                    * np.sum((ws * Li)[:, None, None, None] * pknu_true, axis=0)
                )

            state["p_ij_ell_no_ap_redshift_space_bias_grid"] = np.einsum(
                "lksz->slkz", p_ij_ell_noap
            )

            p_ij_ell = np.zeros((n_spec, self.n_ell, self.nk, self.nzeff))

            for i, z in enumerate(self.zeff):
                apar, aperp = (
                    self.hz_fid[i] / h_z_eff[i],
                    self.chiz_fid[i] / chi_z_eff[i],
                )
                fac = np.sqrt(1 + nus**2 * ((aperp / apar) ** 2 - 1))
                k_apfac = fac / aperp
                nu_true = jnp.tile(nus * aperp / apar / fac, (self.nk, 1)).T
                vol_fac = apar * aperp**2
                ktrue = k_apfac[:, None] * self.k[None, :]
                pknu_true_i = (
                    interp1d(
                        z,
                        self.z,
                        pknu_true.T.reshape(self.nz, -1),
                        extrap=False,
                        method="linear",
                    )
                    .reshape(13, self.nk, len(nus))
                    .T
                )

                pknu_obs = jnp.array(
                    [
                        interp2d(
                            nu_true.reshape(-1),
                            ktrue.reshape(-1),
                            nus,
                            self.k,
                            pknu_true_i[:, :, j],
                            extrap=True,
                        )
                        .reshape(2 * self.ngauss, self.nk)
                        .T
                        for j in range(pknu_true_i.shape[-1])
                    ]
                ).T
                pknu_obs_w_sct = np.zeros((2 * self.ngauss, self.nk, n_spec))
                pknu_obs_w_sct[..., :-7] = pknu_obs[:, :, :-1]
                pknu_obs_w_sct[..., -7] = ktrue**2 * pknu_obs[:, :, -1]
                pknu_obs_w_sct[..., -6] = ktrue**2 * nu_true**2 * pknu_obs[:, :, -1]
                pknu_obs_w_sct[..., -5] = ktrue**2 * nu_true**4 * pknu_obs[:, :, -1]
                pknu_obs_w_sct[..., -4] = ktrue**2 * nu_true**6 * pknu_obs[:, :, -1]

                pknu_obs_w_sct[..., -3] = 1
                pknu_obs_w_sct[..., -2] = ktrue**2 * nu_true**2
                pknu_obs_w_sct[..., -1] = ktrue**4 * nu_true**4

                L0 = np.polynomial.legendre.Legendre((1,))(nus)
                L2 = np.polynomial.legendre.Legendre((0,) * 2 + (1,))(nus)
                L4 = np.polynomial.legendre.Legendre((0,) * 4 + (1,))(nus)

                p_ij_ell[:, 0, :, i] = (
                    0.5
                    * np.sum((ws * L0)[:, None, None] * pknu_obs_w_sct, axis=0).T
                    / vol_fac
                )
                p_ij_ell[:, 1, :, i] = (
                    2.5
                    * np.sum((ws * L2)[:, None, None] * pknu_obs_w_sct, axis=0).T
                    / vol_fac
                )
                p_ij_ell[:, 2, :, i] = (
                    4.5
                    * np.sum((ws * L4)[:, None, None] * pknu_obs_w_sct, axis=0).T
                    / vol_fac
                )
                state["p_ij_ell_redshift_space_bias_grid"] = p_ij_ell

        return state

    def compute(self, state, params_values):
        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        else:
            state = self.compute_analytic(state, params_values)

        return state


class RedshiftSpaceBiasExpansion(LikelihoodModule):
    def __init__(
        self,
        observed_data_vector,
        spectrum_types,
        spectrum_info,
        zeff,
        kmin=1e-3,
        kmax=0.5,
        nk=200,
        **config,
    ):
        """Combine 3d basis spectra into p_gg_ell
        for each tracer and redshift that is requested.

        Args:
            observed_data_vector (DataVector): Measurement object containing tracer information.
            kmin (_type_, optional): _description_. Defaults to 1e-3.
            kmax (float, optional): _description_. Defaults to 3.95.
            nk (int, optional): _description_. Defaults to 200.
        """

        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.z = zeff
        self.nz = len(self.z)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.bias_model = config.get("bias_model", "lpt")
        self.fractional_b1_counterterm = config.get("fractional_b1_counterterm", True)
        self.scale_by_s8z = config.get("scale_by_s8z", True)

        self.output_requirements = {}
        self.sigma8_fid = 0.81

        self.spectrum_params = {
            "p_gg_ell": [
                "b_1_{i}",
                "b_2_{i}",
                "b_s_{i}",
                "b_3_{i}",
                "alpha_0_{i}",
                "alpha_2_{i}",
                "alpha_4_{i}",
                "alpha_6_{i}",
                "sn_{i}",
                "sn_2_{i}",
                "sn_4_{i}",
            ],
        }

        self.spectrum_basis = {
            "p_gg_ell": "p_ij_ell_redshift_space_bias_grid",
        }

        # figure out what spectra we need to calculate,
        # and what parameters are required. bins0
        # are always the density/lens bins
        self.all_spectra = {}
        self.output_requirements = {}
        self.indexed_params = {}
        self.dbins = []
        self.compute_p_gg_cross = False

        self.output_requirements["p_gg_ell"] = []
        self.all_spectra["p_gg_ell"] = []

        if self.scale_by_s8z:
            self.output_requirements["p_gg_ell"].append("sigma8_z")

        for i, j in self.spectrum_info["p_gg_ell"]["bin_pairs"]:
            if i not in self.dbins:
                self.dbins.append(i)
            if j not in self.dbins:
                self.dbins.append(j)
            if self.spectrum_info["p_gg_ell"]["use_cross"]:
                self.all_spectra["p_gg_ell"].append((i, j))
                self.output_requirements["p_gg_ell"].extend(
                    [p.format(i=i) for p in self.spectrum_params["p_gg_ell"]]
                )
                self.output_requirements["p_gg_ell"].extend(
                    [p.format(i=j) for p in self.spectrum_params["p_gg_ell"]]
                )
                self.output_requirements["p_gg_ell"].append(
                    self.spectrum_basis["p_gg_ell"]
                )
                self.compute_p_gg_cross = True
            else:
                if i not in self.dbins:
                    self.dbins.append(i)
                if (i,) not in self.all_spectra["p_gg_ell"]:
                    self.all_spectra["p_gg_ell"].append((i,))
                    self.output_requirements["p_gg_ell"].extend(
                        [p.format(i=i) for p in self.spectrum_params["p_gg_ell"]]
                    )
                    self.output_requirements["p_gg_ell"].append(
                        self.spectrum_basis["p_gg_ell"]
                    )

        self.dbins = jnp.unique(jnp.array(self.dbins))
        self.n_dbins = self.dbins.shape[0]

        # set up params for indexing
        self.indexed_params = {}
        self.indexed_params["p_gg_ell"] = []
        for i in range(self.observed_data_vector.nz_d.shape[0]):
            if (
                i in self.dbins
            ):  # self.observed_data_vector.spectrum_info[spec_type]["bins0"]:
                if self.spectrum_info["p_gg_ell"]["use_cross"]:
                    for j in self.dbins:
                        pars = [p.format(i=i) for p in self.spectrum_params["p_gg_ell"]]
                        pars.extend(
                            [p.format(i=j) for p in self.spectrum_params["p_gg_ell"]]
                        )
                        self.indexed_params["p_gg_ell"].append(pars)
                else:
                    self.indexed_params["p_gg_ell"].append(
                        [p.format(i=i) for p in self.spectrum_params["p_gg_ell"]]
                    )
            else:
                self.indexed_params["p_gg_ell"].append(
                    ["NA" for p in self.spectrum_params["p_gg_ell"]]
                )

        for s in self.indexed_params:
            self.indexed_params["p_gg_ell"] = np.array(self.indexed_params["p_gg_ell"])

    def compute_p_gg_ell(self, state, bias_params, p_ij, s8, f):
        if not self.scale_by_s8z:
            s8 = 1

        if self.bias_model == "lpt":
            p = combine_lpt_redshift_space_spectra(
                p_ij.T,
                bias_params,
                fracb1_counterterm=self.fractional_b1_counterterm,
                s8=s8,
                f=f,
                b1e=True,
            )

        return p

    def compute_cross_p_gg_ell(self, state, bias_params, p_ij, s8, f):
        if not self.scale_by_s8z:
            s8 = 1

        if self.bias_model == "lpt":
            p = combine_lpt_redshift_space_gg_cross_spectra(
                p_ij.T,
                bias_params,
                fracb1_counterterm=self.fractional_b1_counterterm,
                s8=s8,
                f=f,
                b1e=True,
            )

        return p

    def compute(self, state, params_values):
        param_vec = jnp.array(list(params_values.values()))

        for s in self.all_spectra:
            if (s == "p_gg_ell") & self.compute_p_gg_cross:
                computer = getattr(self, "compute_cross_p_gg_ell")
            else:
                computer = getattr(self, f"compute_{s}")

                bias_params = param_vec[self.param_indices[s]]
                s8z = (
                    jnp.interp(self.z, state["z_pk"], state["sigma8_z"])
                    / self.sigma8_fid
                )
                fz = jnp.interp(self.z, state["z_pk"], state["f_z"])

            def f(carry, xs):
                bias_params, p_ij, s8, f = xs
                p = computer(state, bias_params, p_ij, s8, f)
                carry += 1
                return carry, p

            _, state[s] = scan(
                f,
                0,
                [bias_params, state["p_ij_ell_redshift_space_bias_grid"].T, s8z, fz],
            )

        return state


def combine_lpt_redshift_space_spectra(
    spectra, bias_params, fracb1_counterterm=False, s8=None, f=None, b1e=False
):
    """
    Combine LPT-based redshift space power spectra with bias parameters.
    
    Args:
        spectra: basis spectra components, shape (n_spec, nk, nz)
        bias_params: List of bias parameters [b1, b2, bs, b3, alpha0p, alpha2p, alpha4p, alpha6p, sn, sn2, sn4]
        fracb1_counterterm: Flag for fractional b1 counterterm treatment (default: False)
        s8: Optional sigma8 normalization factor
        f: Optional growth rate parameter
        b1e: Optional flag for b1 evolution (default: False)
    
    Returns:
        Combined redshift space power spectrum
    """
    b1, b2, bs, b3, alpha0p, alpha2p, alpha4p, alpha6p, sn, sn2, sn4 = bias_params

    if s8 is not None:
        b1 = b1 / s8
        b2 = b2 / s8**2
        bs = bs / s8**2
        b3 = b3 / s8**3
        if not fracb1_counterterm:
            alpha0 = alpha0p / s8**2
            alpha2 = alpha2p / s8**2
            alpha4 = alpha4p / s8**2
            alpha6 = alpha6p / s8**2

    if b1e:
        b1 = b1 - 1

    if fracb1_counterterm:
        alpha0 = (1 + b1) ** 2 * alpha0p / 0.2**2
        alpha2 = f * (1 + b1) * (alpha0p + alpha2p) / 0.2**2
        alpha4 = f * (f * alpha2p + (1 + b1) * alpha4p) / 0.2**2
        alpha6 = f**2 * alpha6p / 0.2**2

    bias_monomials = jnp.array(
        [
            1,
            b1,
            b1**2,
            b2,
            b1 * b2,
            b2**2,
            bs,
            b1 * bs,
            b2 * bs,
            bs**2,
            b3,
            b1 * b3,
            alpha0,
            alpha2,
            alpha4,
            alpha6,
            sn,
            sn2,
            sn4,
        ]
    )
    p_ell = jnp.sum(spectra * bias_monomials[:, None, None], axis=0)

    return p_ell


def combine_lpt_redshift_space_gg_cross_spectra(
    spectra, bias_params, fracb1_counterterm=False, s8=None, f=None, b1e=False
):
    """
    Combine LPT-based redshift space galaxy-galaxy cross spectra with bias parameters.
    
    Args:
        spectra: basis spectra components, shape (n_spec, nk, nz)
        bias_params: List of bias parameters
        fracb1_counterterm: Flag for fractional b1 counterterm treatment (default: False)
        s8: Optional sigma8 normalization factor
        f: Optional growth rate parameter
        b1e: Optional flag for b1 evolution (default: False)
    
    Returns:
        NotImplementedError: This function is not yet implemented
    """
    raise (NotImplementedError)
