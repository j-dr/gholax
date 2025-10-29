from ..util.likelihood_module import LikelihoodModule
from .emulator import NNPowerSpectrumInterpolator, PijEmulator
from .spline import spline_func_vec
from ..data_vector.two_point_spectrum import field_types
from jax.scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d as scipy_interp1d
from interpax import interp1d, Interpolator2D
from functools import partial
from jax.lax import cond, scan
import jax.numpy as jnp
import numpy as np
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT

# from aemulus_heft.heft_emu import HEFTEmulator
from spinosaurus.cleft_fftw import CLEFT


class RealSpaceBiasedTracerSpectra(LikelihoodModule):
    def __init__(self, zmin=0, zmax=2.0, nz=50, kmin=1e-3, kmax=3.95, nk=200, **config):
        self.nz = nz
        self.nk = nk
        self.z = jnp.linspace(zmin, zmax, nz)
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.kmin = kmin
        self.kmax = kmax

        self.use_emulator = bool(config.get("use_emulator", False))
        self.output_requirements = {}
        self.bias_model = config.get("bias_model", "heft")

        self.contamination_ratio_file = config.get("contamination_ratio_file", None)
        if self.contamination_ratio_file is not None:
            self.contamination_ratio = np.load(self.contamination_ratio_file)
            nk = np.sum(self.contamination_ratio["z"] == 0)
            nz = len(self.contamination_ratio) // nk
            kidx = self.contamination_ratio["k"][0, :].searchsorted(self.k[-1])
            self.contamination_ratio = self.contamination_ratio.reshape(-1, nk)[
                :, :kidx
            ]
            self.contamination_interp = Interpolator2D(
                self.contamination_ratio["z"][:, 0],
                self.contamination_ratio["k"][0, :],
                self.contamination_ratio["pk"][:, :],
                extrap=True,
                method="linear",
            )

        if self.use_emulator:
            self.emulator_file_names = config["emulator_file_names"]
            self.emulator = PijEmulator(self.emulator_file_names)
            self.output_requirements["p_ij_real_space_bias_grid"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]
            self.output_requirements["p_11_real_space_bias_grid"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]

        else:
            self.output_requirements["p_ij_real_space_bias_grid"] = [
                "Pm_lin_z",
                "Pcb_lin_z",
            ]
            self.output_requirements["p_11_real_space_bias_grid"] = [
                "Pm_lin_z",
            ]

        self.kIR = config.get("kIR", 0.2)
        self.save_p_11_separately = False
        self.save_p_ij = True
        self.interpolation_order = config.get("interpolation_order", "cubic")

        if (self.bias_model == "heft") & (not self.use_emulator):
            self.kecleft = True

        else:
            self.kecleft = False

        if self.kecleft:
            self.nspec = 15
        else:
            self.nspec = 19

    def compute_cleft_analytic(self, state, params_values):
        spec_grid = jnp.zeros((self.nz, self.nspec, self.nk))

        s_m_map = {1: 0, 3: 1, 6: 3, 10: 6, 15: 10}
        s_cb_map = {
            2: 0,
            4: 1,
            5: 2,
            7: 3,
            8: 4,
            9: 5,
            11: 6,
            12: 7,
            13: 8,
            14: 9,
            16: 10,
            17: 11,
            18: 12,
        }

        if self.kecleft:
            for i, z in enumerate(self.z):
                # can't just rescale by D(z) with neutrinos.
                pk_m_lin = state["Pm_lin_z"][i, :]
                pk_cb_lin = state["Pcb_lin_z"][i, :]
                pk_cb_m_lin = np.sqrt(pk_m_lin * pk_cb_lin)

                # matter power spectrum
                # kmin, kmax nk all hard coded to agree with what was used for aemulus_heft
                cleftobj = RKECLEFT(state["k_lin"], pk_m_lin)
                cleftobj.make_ptable(D=1, kmin=1e-3, kmax=10, nk=300)
                cleftpk_m_m = scipy_interp1d(
                    cleftobj.pktable.T[0, :],
                    cleftobj.pktable.T[1:, :],
                    kind="cubic",
                    axis=1,
                )(self.k)

                # cb spectra
                cleftobj = RKECLEFT(state["k_lin"], pk_cb_lin)
                cleftobj.make_ptable(D=1, kmin=1e-3, kmax=10, nk=300)
                cleftpk_cb_cb = scipy_interp1d(
                    cleftobj.pktable.T[0, :],
                    cleftobj.pktable.T[1:, :],
                    kind="cubic",
                    axis=1,
                )(self.k)

                # cb x matter power spectra
                cleftobj = RKECLEFT(state["k_lin"], pk_cb_m_lin)
                cleftobj.make_ptable(D=1, kmin=1e-3, kmax=10, nk=300)
                cleftpk_cb_m = scipy_interp1d(
                    cleftobj.pktable.T[0, :],
                    cleftobj.pktable.T[1:, :],
                    kind="cubic",
                    axis=1,
                )(self.k)

                for s in np.arange(self.nspec):
                    if s == 0:
                        spec_grid = spec_grid.at[i, 0, ...].set(cleftpk_m_m[0])
                    elif s in [1, 3, 6, 10]:
                        spec_grid = spec_grid.at[i, s, ...].set(
                            cleftpk_cb_m[s_m_map[s]]
                        )
                    else:
                        spec_grid = spec_grid.at[i, s, ...].set(
                            cleftpk_cb_cb[s_cb_map[s]]
                        )

        else:
            for i, z in enumerate(self.z):
                pk_m_lin = state["Pm_lin_z"][i, :]
                pk_cb_lin = state["Pcb_lin_z"][i, :]
                pk_cb_m_lin = np.sqrt(pk_m_lin * pk_cb_lin)

                # matter power spectrum
                cleftobj = CLEFT(state["k_lin"], pk_m_lin, kIR=self.kIR)
                cleftobj.make_ptable(kmin=self.kmin, kmax=self.kmax, nk=300)
                cleftpk_m_m = scipy_interp1d(
                    cleftobj.pktable.T[0, :],
                    cleftobj.pktable.T[1:, :],
                    kind="cubic",
                    axis=1,
                    fill_value="extrapolate",
                )(self.k)

                # cb spectra
                cleftobj = CLEFT(state["k_lin"], pk_cb_lin, kIR=self.kIR)
                cleftobj.make_ptable(kmin=self.kmin, kmax=self.kmax, nk=300)
                cleftpk_cb_cb = scipy_interp1d(
                    cleftobj.pktable.T[0, :],
                    cleftobj.pktable.T[1:, :],
                    kind="cubic",
                    axis=1,
                    fill_value="extrapolate",
                )(self.k)

                # cb x matter power spectra
                cleftobj = CLEFT(state["k_lin"], pk_cb_m_lin, kIR=self.kIR)
                cleftobj.make_ptable(kmin=self.kmin, kmax=self.kmax, nk=300)
                cleftpk_cb_m = scipy_interp1d(
                    cleftobj.pktable.T[0, :],
                    cleftobj.pktable.T[1:, :],
                    kind="cubic",
                    axis=1,
                    fill_value="extrapolate",
                )(self.k)

                for s in np.arange(self.nspec):
                    if s == 0:
                        spec_grid = spec_grid.at[i, 0, ...].set(cleftpk_m_m[0])
                    elif s in [1, 3, 6, 10, 15]:
                        spec_grid = spec_grid.at[i, s, ...].set(
                            cleftpk_cb_m[s_m_map[s]]
                        )
                    else:
                        spec_grid = spec_grid.at[i, s, ...].set(
                            cleftpk_cb_cb[s_cb_map[s]]
                        )

        spec_grid = spec_grid.at[:, 3, :].set(spec_grid[:, 3, :] / 2)
        spec_grid = spec_grid.at[:, 4, :].set(spec_grid[:, 4, :] / 2)
        spec_grid = spec_grid.at[:, 9, :].set(spec_grid[:, 9, :] / 0.25)
        spec_grid = spec_grid.at[:, 10, :].set(spec_grid[:, 10, :] / 2)
        spec_grid = spec_grid.at[:, 11, :].set(spec_grid[:, 11, :] / 2)
        spec_grid = spec_grid.at[:, 12, :].set(spec_grid[:, 12, :] / 2)

        if not self.kecleft:
            spec_grid = spec_grid.at[:, 15, :].set(spec_grid[:, 15, :] / 2)
            spec_grid = spec_grid.at[:, 16, :].set(spec_grid[:, 16, :] / 2)
            spec_grid = spec_grid.at[:, 17, :].set(spec_grid[:, 17, :] / 2)

        if self.bias_model == "heft":
            state["eft_spectrum_grid"] = spec_grid
        else:
            state["p_ij_real_space_bias_grid"] = spec_grid

        return state

    def compute_heft(self, state, params_values):
        return state

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
        pk_ij = self.emulator.predict(cparam_grid).T
        n_spec = len(self.emulator.pij_emus)

        logk_emu = jnp.log10(self.emulator.pij_emus[0].k)
        if self.save_p_ij:
            state["p_ij_real_space_bias_grid"] = jnp.zeros((n_spec, self.nk, self.nz))
            for i in range(n_spec):
                p = interp1d(
                    self.logk,
                    logk_emu,
                    pk_ij[:, i, :],
                    extrap=0,
                    method=self.interpolation_order,
                )

                state["p_ij_real_space_bias_grid"] = (
                    state["p_ij_real_space_bias_grid"].at[i, ...].set(p)
                )

        if self.save_p_11_separately:
            state["p_11_real_space_bias_grid"] = interp1d(
                self.logk,
                logk_emu,
                pk_ij[:, 0, :],
                extrap=0,
                method=self.interpolation_order,
            )

        if self.contamination_ratio_file is not None:
            print("Calculating contamination", flush=True)
            zs = self.z.reshape(-1, 1)
            k = jnp.tile(self.k, (len(zs), 1))
            xs = (zs, k)

            def interp_scan(carry, x):
                return carry, self.contamination_interp(x[0], x[1])

            _, p11_cont = scan(interp_scan, 0, xs)
            # print(p11_cont, flush=True)
            if self.save_p_ij:
                state["p_ij_real_space_bias_grid"] = (
                    state["p_ij_real_space_bias_grid"]
                    .at[0, ...]
                    .set(state["p_ij_real_space_bias_grid"][0, ...] * p11_cont.T)
                )
                state["p_ij_real_space_bias_grid"] = (
                    state["p_ij_real_space_bias_grid"]
                    .at[1, ...]
                    .set(state["p_ij_real_space_bias_grid"][1, ...] * p11_cont.T)
                )
                state["p_ij_real_space_bias_grid"] = (
                    state["p_ij_real_space_bias_grid"]
                    .at[2, ...]
                    .set(state["p_ij_real_space_bias_grid"][2, ...] * p11_cont.T)
                )

            if self.save_p_11_separately:
                state["p_11_real_space_bias_grid"] = (
                    state["p_11_real_space_bias_grid"]
                    .at[...]
                    .set(state["p_11_real_space_bias_grid"][...] * p11_cont.T)
                )

        return state

    def compute_p11_boltz_analytic(self, state, params_values):
        boltz = state["boltzmann_results"]
        h = params_values["H0"] / 100

        pk_m = np.zeros((self.nz, self.nk))

        for i, z in enumerate(self.z):
            for j, k in enumerate(self.k):
                pk_m[i, j] = boltz.pk(k * h, z) * h**3

        state["p_11_real_space_bias_grid"] = pk_m

        return state

    def compute_p11_boltz_analytic(self, state, params_values):
        boltz = state["boltzmann_results"]
        h = params_values["H0"] / 100

        pk_m = np.zeros((self.nz, self.nk))

        for i, z in enumerate(self.z):
            for j, k in enumerate(self.k):
                pk_m[i, j] = boltz.pk(k * h, z) * h**3

        state["p_11_real_space_bias_grid"] = pk_m

        return state

    def compute(self, state, params_values):
        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        else:
            if self.bias_model == "heft":
                state = self.compute_cleft_analytic(state, params_values)
                state = self.compute_heft(state, params_values)

            elif self.bias_model == "cleft":
                state = self.compute_cleft_analytic(state, params_values)

            else:
                raise ValueError("Not implemented")
        return state


class RealSpaceMatterPowerSpectrum(RealSpaceBiasedTracerSpectra):
    def __init__(self, zmin=0, zmax=2, nz=50, kmin=0.001, kmax=3.95, nk=200, **config):
        super().__init__(zmin, zmax, nz, kmin, kmax, nk, **config)
        self.save_p_11_separately = True
        self.save_p_ij = False
        self.use_boltzmann = config.get("use_boltzmann", False)

        if self.use_emulator:
            self.output_requirements["p_11_real_space_bias_grid"] = [
                "As",
                "ns",
                "H0",
                "w",
                "ombh2",
                "omch2",
                "mnu",
            ]
        else:
            self.output_requirements["p_11_real_space_bias_grid"] = [
                "Pm_lin_z",
            ]

    def compute(self, state, params_values):
        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        else:
            if self.use_boltzmann:
                state = self.compute_p11_boltz_analytic(state, params_values)
            else:
                raise (ValueError("Analytic HEFT/CLEFT calculation not implemented"))

        return state


class RealSpaceBiasExpansion(LikelihoodModule):
    def __init__(
        self,
        observed_data_vector,
        spectrum_types,
        spectrum_info,
        zmin=0.0,
        zmax=2.0,
        nz=50,
        kmin=1e-3,
        kmax=3.95,
        nk=200,
        k_cutoff=2.0,
        **config,
    ):
        """Combine 3d basis spectra into p_mm, p_gm and p_gg 
        for each tracer that is requested.

        Args:
            observed_data_vector (DataVector): Measurement object containing tracer information.
            zmin (float, optional): Minimum redshift of outputs. Defaults to 0.0.
            zmax (float, optional): Maximum redshift of outputs. Defaults to 2.0.
            nz (int, optional): number of redshifts. Defaults to 50.
            kmin (_type_, optional): Minimum k value in h/Mpc. Defaults to 1e-3.
            kmax (float, optional): Maximum k value in h/Mpc. Defaults to 3.95.
            nk (int, optional): Number of k value. Defaults to 200.
        """

        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.z = jnp.linspace(zmin, zmax, nz)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.bias_model = config.get("bias_model", "heft")
        self.fractional_b1_counterterm = config.get("fractional_b1_counterterm", True)
        self.scale_by_s8z = config.get("scale_by_s8z", True)
        self.baryon_ct_s8z_scaling = config.get("baryon_ct_s8z_scaling", False)
        self.baryon_z_evolution_model = config.get(
            "baryon_z_evolution_model", "constant_bias"
        )
        self.spline_N = config.get("spline_N", 1)
        self.spline_Delta = config.get("spline_Delta", 2.0)
        self.spline_min = config.get("spline_min", 0.0)
        self.p_mm_ct_order = config.get("p_mm_ct_order", 1)
        self.p_mm_uv_behavior = config.get("p_mm_uv_behavior", "dmo")
        self.p_mm_ct_pade = config.get("p_mm_ct_pade", True)
        self.k_cutoff = k_cutoff

        self.output_requirements = {}
        if self.scale_by_s8z:
            self.sigma8_fid = 0.81

        self.required_spectra = {
            "c_dd": ["p_gg", "p_gm", "p_mm"],
            "c_dk": ["p_gm", "p_mm"],
            "c_kk": ["p_mm"],
            "c_bb": [],
            "c_dcmbk": ["p_gm", "p_mm"],
            "c_cmbkcmbk": ["p_gm", "p_mm"],
        }

        self.spectrum_params = {
            "p_mm": ["b_kma"],
            "p_gm": ["b_1_{i}", "b_2_{i}", "b_s_{i}", "b_3_{i}", "b_kx_{i}", "sn_{i}"],
            "p_gg": ["b_1_{i}", "b_2_{i}", "b_s_{i}", "b_3_{i}", "b_ka_{i}", "sn_{i}"],
        }

        if self.p_mm_ct_pade:
            self.spectrum_params["p_mm"].append("r_pade")

        if self.p_mm_ct_order > 1:
            for i in range(2, self.p_mm_ct_order + 1):
                self.spectrum_params["p_mm"].append(f"b_kma{int(i * 2)}")

        self.spectrum_basis = {
            "p_mm": "p_11_real_space_bias_grid",
            "p_gm": "p_ij_real_space_bias_grid",
            "p_gg": "p_ij_real_space_bias_grid",
        }

        # figure out what spectra we need to calculate,
        # and what parameters are required. bins0
        # are always the density/lens bins
        self.all_spectra = {}
        self.output_requirements = {}
        self.indexed_params = {}
        self.dbins = []
        self.compute_p_gg_cross = False

        for spec_type in self.spectrum_types:
            if spec_type in self.required_spectra.keys():
                spectra = self.required_spectra[spec_type]
                for s in spectra:
                    self.output_requirements[s] = []

                    if s not in self.all_spectra:
                        if s == "p_mm":
                            self.all_spectra[s] = [(0,)]
                            self.output_requirements["p_mm"] = [
                                "p_11_real_space_bias_grid",
                            ]
                            if self.baryon_z_evolution_model == "spline":
                                pars = [
                                    f"{p}_{n}"
                                    for p in self.spectrum_params["p_mm"]
                                    for n in range(self.spline_N)
                                ]
                            else:
                                pars = self.spectrum_params["p_mm"]

                            self.output_requirements["p_mm"].extend(pars)
                            continue  # p_mm doesn't have any tracer specific parameters
                        else:
                            self.all_spectra[s] = []

                    if self.scale_by_s8z:
                        self.output_requirements[s].append("sigma8_z")

                    for i, j in self.spectrum_info[spec_type]["bin_pairs"]:
                        if (i not in self.dbins) & ("d" in field_types[spec_type][0]):
                            self.dbins.append(i)
                        if (j not in self.dbins) & ("d" in field_types[spec_type][1]):
                            self.dbins.append(j)
                        if (s == "p_gg") & (self.spectrum_info[spec_type]["use_cross"]):
                            self.all_spectra[s].append((i, j))
                            self.output_requirements[f"p_gg"].extend(
                                [p.format(i=i) for p in self.spectrum_params[s]]
                            )
                            self.output_requirements[f"p_gg"].extend(
                                [p.format(i=j) for p in self.spectrum_params[s]]
                            )
                            self.output_requirements[f"p_gg"].append(
                                self.spectrum_basis[s]
                            )
                            self.compute_p_gg_cross = True
                        else:
                            if (i not in self.dbins) & (
                                "d" in field_types[spec_type][0]
                            ):
                                self.dbins.append(i)
                            if ((i,) not in self.all_spectra[s]) & (
                                "d" in field_types[spec_type][0]
                            ):
                                self.all_spectra[s].append((i,))
                                self.output_requirements[s].extend(
                                    [p.format(i=i) for p in self.spectrum_params[s]]
                                )
                                self.output_requirements[s].append(
                                    self.spectrum_basis[s]
                                )

        self.dbins = jnp.unique(jnp.array(self.dbins))
        self.n_dbins = self.dbins.shape[0]

        # set up params for indexing
        self.indexed_params = {}
        for s in self.all_spectra:
            self.indexed_params[s] = []
            if s != "p_mm":
                for i in range(self.observed_data_vector.nz_d.shape[0]):
                    if (
                        i in self.dbins
                    ):  # self.observed_data_vector.spectrum_info[spec_type]["bins0"]:
                        if (s == "p_gg") & self.spectrum_info["c_dd"]["use_cross"]:
                            for j in self.dbins:
                                pars = [p.format(i=i) for p in self.spectrum_params[s]]
                                pars.extend(
                                    [p.format(i=j) for p in self.spectrum_params[s]]
                                )
                                self.indexed_params[s].append(pars)
                        else:
                            self.indexed_params[s].append(
                                [p.format(i=i) for p in self.spectrum_params[s]]
                            )
                    else:
                        self.indexed_params[s].append(
                            ["NA" for p in self.spectrum_params[s]]
                        )
            else:
                if self.baryon_z_evolution_model == "spline":
                    pars = [
                        f"{p}_{n}"
                        for p in self.spectrum_params["p_mm"]
                        for n in range(self.spline_N)
                    ]
                else:
                    pars = self.spectrum_params["p_mm"]
                self.indexed_params[s].append(pars)

        for s in self.indexed_params:
            self.indexed_params[s] = np.array(self.indexed_params[s])

    def set_bs(self, param_vec, param_indices, z_evolution_model):
        if z_evolution_model == "constant_bias":
            b_vec = param_vec[param_indices]

        elif z_evolution_model == "spline":
            param_indices = param_indices.reshape(-1, self.spline_N).T
            b_vec = spline_func_vec(
                self.z,
                param_vec[param_indices],
                self.spline_min,
                Delta=self.spline_Delta,
            )

        return b_vec

    def compute_p_mm(self, state, bias_params):
        p_11 = state["p_11_real_space_bias_grid"]
        b_kma = bias_params[0]

        if self.p_mm_ct_pade:
            r_pade_n = bias_params[1]
            r_pade_d = bias_params[1]
            counter = 1
        else:
            r_pade_n = 1
            r_pade_d = 0
            counter = 0

        if self.p_mm_uv_behavior == "dmo":
            mask = self.k < self.k_cutoff
        else:
            mask = np.ones_like(self.k)
            
        if self.baryon_ct_s8z_scaling:
            s8z = state["sigma8_z"] / self.sigma8_fid
        else:
            s8z = jnp.ones(state["sigma8_z"].shape[0])

        k_b = 0.4 / np.sqrt(
            0.1
        )  # kmax / sqrt(eps) s.t. b_kma = 1 gives eps fractional contribution at kmax
        ct_n = b_kma * (self.k[:, None] / k_b) ** 2
        ct_d = 1 + (r_pade_d * self.k[:, None] / k_b) ** 2

        for i in range(2, self.p_mm_ct_order + 1):
            bk2i = bias_params[counter + i - 1]
            ct_n = ct_n + bk2i * (r_pade_n * self.k[:, None] / k_b) ** (2 * i)
            ct_d = ct_d + bk2i * (r_pade_d * self.k[:, None] / k_b) ** (2 * i)

        p = p_11 * (1 - (ct_n / ct_d * mask[:, None])/s8z**2)

        return p

    def compute_p_gm(self, state, bias_params):
        p_ij = state["p_ij_real_space_bias_grid"]
        if self.scale_by_s8z:
            s8z = state["sigma8_z"] / self.sigma8_fid
        else:
            s8z = jnp.ones(state["sigma8_z"].shape[0])

        if self.bias_model == "heft":
            p = combine_real_space_spectra(
                10**self.logk,
                p_ij,
                bias_params,
                cross=True,
                fracb1_counterterm=self.fractional_b1_counterterm,
                s8z=s8z,
                b1e=True,
            )

        return p

    def compute_p_gg(self, state, bias_params):
        p_ij = state["p_ij_real_space_bias_grid"]
        if self.scale_by_s8z:
            s8z = state["sigma8_z"] / self.sigma8_fid
        else:
            s8z = jnp.ones(state["sigma8_z"].shape[0])

        if self.bias_model == "heft":
            p = combine_real_space_spectra(
                10**self.logk,
                p_ij,
                bias_params,
                cross=False,
                fracb1_counterterm=self.fractional_b1_counterterm,
                s8z=s8z,
                b1e=True,
            )

        return p

    def compute_cross_p_gg(self, state, bias_params):
        p_ij = state["p_ij_real_space_bias_grid"]
        if self.scale_by_s8z:
            s8z = state["sigma8_z"] / self.sigma8_fid
        else:
            s8z = jnp.ones(state["sigma8_z"].shape[0])

        if self.bias_model == "heft":
            p = combine_real_space_gg_cross_spectra(
                10**self.logk,
                p_ij,
                bias_params,
                fracb1_counterterm=self.fractional_b1_counterterm,
                s8z=s8z,
                b1e=True,
            )

        return p

    def compute(self, state, params_values):
        param_vec = jnp.array(list(params_values.values()))

        for s in self.all_spectra:
            if (s == "p_gg") & self.compute_p_gg_cross:
                computer = getattr(self, f"compute_cross_p_gg")
            else:
                computer = getattr(self, f"compute_{s}")

            if s == "p_mm":
                if self.baryon_z_evolution_model == "spline":
                    bias_params = self.set_bs(
                        param_vec, self.param_indices[s], self.baryon_z_evolution_model
                    ).reshape(1, -1, len(self.z))
                else:
                    bias_params = self.set_bs(
                        param_vec, self.param_indices[s], self.baryon_z_evolution_model
                    )
            else:
                bias_params = param_vec[self.param_indices[s]]

            def f(carry, bias_params_i):
                p = computer(state, bias_params_i)
                carry += 1
                return carry, p

            _, state[s] = scan(f, 0, bias_params)

        return state


def combine_real_space_spectra(
    k, spectra, bias_params, fracb1_counterterm=False, cross=False, s8z=None, b1e=False
):
    """
    Combine real space power spectra with bias parameters.
    
    Args:
        k: Wavenumber array
        spectra: basis spectra components, shape (n_spec, n_k, n_z)
        bias_params: List of bias parameters [b1, b2, bs, b3, bk2, sn]
        fracb1_counterterm: Flag for fractional b1 counterterm treatment (default: False)
        cross: Flag for cross-spectrum computation (default: False)
        s8z: Optional sigma8(z) normalization factor
        b1e: Optional flag for eulerian bias (default: False)
    
    Returns:
        Combined real space power spectrum
    """
    pkvec = jnp.zeros((19, spectra.shape[1], spectra.shape[2]))
    pkvec = pkvec.at[:15, ...].set(spectra)

    b1, b2, bs, b3, bk2, sn = bias_params
    # normalize counterterm at kmax=0.4
    if s8z is not None:
        b1 = b1 / s8z
        b2 = b2 / s8z**2
        bs = bs / s8z**2
        if not fracb1_counterterm:
            bk2 = bk2 / s8z

        zero = jnp.zeros_like(b1)
    else:
        zero = 0.0

    if b1e:
        b1 = b1 - 1

    if fracb1_counterterm:
        bk2 = 0.5 * (1 + b1) * bk2 / 0.4**2

    # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
    bterms_hh = [
        zero,
        zero,
        zero + 1.0,
        zero,
        2 * b1,
        b1**2,
        zero,
        b2,
        b2 * b1,
        0.25 * b2**2,
        zero,
        2 * bs,
        2 * bs * b1,
        bs * b2,
        bs**2,
        2 * bk2,
        2 * bk2 * b1,
        bk2 * b2,
        2 * bk2 * bs,
    ]

    # hm correlations only have one kind of <1,delta_i> correlation
    bterms_hm = [
        zero,
        zero + 1.0,
        zero,
        b1,
        zero,
        zero,
        b2 / 2,
        zero,
        zero,
        zero,
        bs,
        zero,
        zero,
        zero,
        zero,
        bk2,
        zero,
        zero,
        zero,
    ]

    # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
    if cross:
        nabla_idx = (1, 3, 6, 10)
    else:
        nabla_idx = (2, 4, 7, 11)

    # Higher derivative terms
    for i, n in enumerate(nabla_idx):
        pkvec = pkvec.at[15 + i].set(-(k[:, None] ** 2) * pkvec[n])

    if cross:
        bterms_hm = jnp.array(bterms_hm)
        p = jnp.einsum("bz, bkz->kz", bterms_hm, pkvec)

    else:
        bterms_hh = jnp.array(bterms_hh)
        p = jnp.einsum("bz, bkz->kz", bterms_hh, pkvec) + sn

    return p


def combine_real_space_gg_cross_spectra(
    k, spectra, bias_params, fracb1_counterterm=False, s8z=None, b1e=False
):
    """
    Combine real space galaxy-galaxy cross power spectra with bias parameters.
    
    Args:
        k: Wavenumber array
        spectra: Basis spectra, shape (n_spec, n_k, n_z)
        bias_params: List of bias parameters [b1_a, b2_a, bs_a, b3_a, bk2_a, sn_a, b1_b, b2_b, bs_b, b3_b, bk2_b, sn_b]
        fracb1_counterterm: Flag for fractional b1 counterterm treatment (default: False)
        s8z: Optional sigma8(z) normalization factor
        b1e: Optional flag for eulerian b1 (default: False)
    
    Returns:
        Combined real space galaxy-galaxy cross power spectrum
    """
    pkvec = jnp.zeros((19, spectra.shape[1], spectra.shape[2]))
    pkvec = pkvec.at[:15, ...].set(spectra)

    b1_a, b2_a, bs_a, b3_a, bk2_a, sn_a, b1_b, b2_b, bs_b, b3_b, bk2_b, sn_b = (
        bias_params
    )

    # normalize counterterm at kmax=0.4
    if s8z is not None:
        b1_a = b1_a / s8z
        b2_a = b2_a / s8z**2
        bs_a = bs_a / s8z**2
        if not fracb1_counterterm:
            bk2_a = bk2_a / s8z

        b1_b = b1_b / s8z
        b2_b = b2_b / s8z**2
        bs_b = bs_b / s8z**2
        if not fracb1_counterterm:
            bk2_b = bk2_b / s8z

        zero = jnp.zeros_like(b1_a)
    else:
        zero = 0.0

    if b1e:
        b1_a = b1_a - 1
        b1_b = b1_b - 1

    if fracb1_counterterm:
        bk2_a = 0.5 * (1 + b1_a) * bk2_a / 0.4**2
        bk2_b = 0.5 * (1 + b1_b) * bk2_b / 0.4**2

    # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
    bterms_hh = [
        zero,
        zero,
        zero + 1.0,
        zero,
        b1_a + b1_b,
        b1_a * b1_b,
        zero,
        0.5 * (b2_a + b2_b),
        0.5 * (b2_a * b1_b + b2_b * b1_a),
        0.25 * b2_a * b2_b,
        zero,
        bs_a + bs_b,
        bs_a * b1_b + bs_b * b1_a,
        0.5 * (bs_a * b2_b + bs_b * b2_a),
        bs_a * bs_b,
        bk2_a + bk2_b,
        bk2_a * b1_b + bk2_b * b1_a,
        0.5 * (bk2_a * b2_a + bk2_b * b2_b),
        (bk2_a * bs_b + bk2_b + bs_a),
    ]

    nabla_idx = (2, 4, 7, 11)

    # Higher derivative terms
    for i, n in enumerate(nabla_idx):
        pkvec = pkvec.at[15 + i].set(-(k[:, None] ** 2) * pkvec[n])

    bterms_hh = jnp.array(bterms_hh)
    p = jnp.einsum("bz, bkz->kz", bterms_hh, pkvec)
    add_sn = lambda p, sn_a: p + sn_a
    noadd_sn = lambda p, sn_a: p
    p = cond(sn_a == sn_b, add_sn, noadd_sn, p, sn_a)

    return p
