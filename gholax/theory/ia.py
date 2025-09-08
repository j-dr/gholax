import jax.numpy as jnp
import numpy as np
from interpax import interp1d
from jax.lax import scan
from spinosaurus.density_shape_correlators_fftw import DensityShapeCorrelators
from spinosaurus.shape_shape_correlators_fftw import ShapeShapeCorrelators

from ..util.likelihood_module import LikelihoodModule
from .emulator import MultiSpectrumEmulator
from .spline import spline_func_vec


class DensityShapeIA(LikelihoodModule):
    def __init__(self, zmin=0, zmax=2.0, nz=50, kmin=1e-3, kmax=3.95, nk=200, **config):
        self.nz = nz
        self.nk = nk
        self.z = jnp.linspace(zmin, zmax, nz)
        self.kmin = kmin
        self.kmax = kmax
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)

        self.use_emulator = bool(config.get("use_emulator", False))
        self.interpolation_order = config.get("interpolation_order", "cubic")
        self.no_ia = config.get("no_ia", False)
        self.heft_nla = config.get("heft_nla", False)

        self.output_requirements = {}
        if self.use_emulator:
            self.emulator = MultiSpectrumEmulator(
                config["emulator_file_names"],
            )                
            self.input_param_order = self.emulator.input_param_order
            
            if self.input_param_order[-1] != 'z':
                self.input_param_order.remove('z')
                self.input_param_order.append('z')
                self.emulator = MultiSpectrumEmulator(
                    config['emulator_file_names'], input_param_order=self.input_param_order
                )

            self.output_requirements["p_ij_real_space_density_shape_grid"] = self.input_param_order[:-1]

        else:
            self.output_requirements["p_ij_real_space_density_shape_grid"] = [
                "Pcb_lin_z",
                "Pm_lin_z",
            ]

        if self.heft_nla:
            self.output_requirements["p_ij_real_space_density_shape_grid"].append(
                "p_ij_real_space_bias_grid"
            )

        self.kIR = config.get("kIR", 0.2)

    def compute_emulator(self, state, params_values):
        cosmo_params = jnp.array(
            [params_values[p] for p in self.input_param_order[:-1]]
        )
        cparam_grid = jnp.zeros((self.nz, len(cosmo_params) + 1))
        cparam_grid = cparam_grid.at[:, :-1].set(cosmo_params)
        cparam_grid = cparam_grid.at[:, -1].set(self.z)
        n_spec = self.emulator.n_spec

        # assume same k values for density shape and shape shape.
        logk_emu = jnp.log10(self.emulator.k)
        state["p_ij_real_space_density_shape_grid"] = jnp.zeros(
            (n_spec, self.nk, self.nz)
        )

        pk_ij = self.emulator.predict(cparam_grid).T

        for i in range(n_spec):
            p = interp1d(
                self.logk,
                logk_emu,
                pk_ij[:, i, :],
                extrap=0,
                method=self.interpolation_order,
            )
            state["p_ij_real_space_density_shape_grid"] = (
                state["p_ij_real_space_density_shape_grid"].at[i, ...].set(p)
            )

        return state

    def compute_analytic(self, state):
        k_lin = np.array(state["k_lin"])
        p_cb = state["Pcb_lin_z"]

        n_spec = 21
        state["p_ij_real_space_density_shape_grid"] = jnp.zeros(
            (n_spec, self.nk, self.nz)
        )

        for i, z in enumerate(self.z):
            ptmod = DensityShapeCorrelators(k_lin, p_cb[i, :], kIR=self.kIR)
            ptmod.make_gdtable(kmin=self.kmin, kmax=self.kmax, nk=self.nk)
            pk_ij = ptmod.pktable_gd.T[1:, :].T
            for j in range(n_spec):
                p = interp1d(
                    self.logk,
                    np.log10(ptmod.pktable_gd.T[0, :]),
                    pk_ij[:, j],
                    extrap=0,
                    method=self.interpolation_order,
                )
                state["p_ij_real_space_density_shape_grid"] = (
                    state["p_ij_real_space_density_shape_grid"].at[j, :, i].set(p)
                )

        return state

    def compute(self, state, params_values):
        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        else:
            state = self.compute_analytic(state)

        if self.heft_nla:
            state["p_ij_real_space_density_shape_grid"] = (
                state["p_ij_real_space_density_shape_grid"]
                .at[0, ...]
                .set(2 / 3 * state["p_ij_real_space_bias_grid"][2, ...])
            )
            state["p_ij_real_space_density_shape_grid"] = (
                state["p_ij_real_space_density_shape_grid"]
                .at[4, ...]
                .set(2 / 3 * state["p_ij_real_space_bias_grid"][4, ...])
            )

        return state


class ShapeShapeIA(LikelihoodModule):
    def __init__(self, zmin=0, zmax=2.0, nz=50, kmin=1e-3, kmax=3.95, nk=200, **config):
        self.nz = nz
        self.nk = nk
        self.kmin = kmin
        self.kmax = kmax
        self.z = jnp.linspace(zmin, zmax, nz)
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)

        self.use_emulator = bool(config.get("use_emulator", False))
        self.interpolation_order = config.get("interpolation_order", "cubic")
        self.no_ia = config.get("no_ia", False)
        self.heft_nla = config.get("heft_nla", False)

        self.output_requirements = {}
        
        if self.use_emulator:
            self.emulators = []
            for m in range(3):
                if m==0:
                    self.emulators.append( MultiSpectrumEmulator(
                        config['emulator_file_names'][m],
                    ))
                    self.input_param_order = self.emulators[0].input_param_order
                    
                    if self.input_param_order[-1] != 'z':
                        self.input_param_order.remove('z')
                        self.input_param_order.append('z')
                        self.emulators[0] = MultiSpectrumEmulator(
                            config['emulator_file_names'][m], input_param_order=self.input_param_order
                        )

                else:
                    self.emulators.append( MultiSpectrumEmulator(
                        config['emulator_file_names'][m], self.input_param_order
                    ))
                    
            self.output_requirements["p_mij_real_space_shape_shape_grid"] = self.input_param_order[:-1]

        else:
            try:
                self.output_requirements["p_mij_real_space_shape_shape_grid"] = [
                    "Pcb_lin_z"
                ]
            except Exception as e:
                print(e)
                raise ImportError(
                    "Must install spinosaurus to compute shape-shape contributions analytically."
                )

        if self.heft_nla:
            self.output_requirements["p_mij_real_space_shape_shape_grid"].append(
                "p_ij_real_space_bias_grid"
            )

        self.kIR = config.get("kIR", 0.2)

    def compute_emulator(self, state, params_values):
        cosmo_params = jnp.array(
            [params_values[p] for p in self.input_param_order[:-1]]
        )
        cparam_grid = jnp.zeros((self.nz, len(cosmo_params) + 1))
        cparam_grid = cparam_grid.at[:, :-1].set(cosmo_params)
        cparam_grid = cparam_grid.at[:, -1].set(self.z)
        n_spec = self.emulators[0].n_spec

        # assume same k values same for all m.
        logk_emu = jnp.log10(self.emulators[0].k)
        state["p_mij_real_space_shape_shape_grid"] = jnp.zeros(
            (3, n_spec, self.nk, self.nz)
        )

        for m in [0, 1, 2]:
            pk_ij = self.emulators[m].predict(cparam_grid).T

            for i in range(n_spec):
                p = interp1d(
                    self.logk,
                    logk_emu,
                    pk_ij[:, i, :],
                    extrap=0,
                    method=self.interpolation_order,
                )

                state["p_mij_real_space_shape_shape_grid"] = (
                    state["p_mij_real_space_shape_shape_grid"].at[m, i, ...].set(p)
                )

        return state

    def compute_analytic(self, state):
        k_lin = np.array(state["k_lin"])
        p_cb = state["Pcb_lin_z"]

        n_spec = 13
        state["p_mij_real_space_shape_shape_grid"] = jnp.zeros(
            (3, n_spec, self.nk, self.nz)
        )

        for i, z in enumerate(self.z):
            ptmod = ShapeShapeCorrelators(k_lin, p_cb[i, :], kIR=self.kIR)

            for m in [0, 1, 2]:
                ptmod.make_ggtable(m, kmin=self.kmin, kmax=self.kmax, nk=self.nk)
                pk_ij = ptmod.pktables_gg[m].T[1:, :].T
                for j in range(n_spec):
                    p = interp1d(
                        self.logk,
                        np.log10(ptmod.pktables_gg[m].T[0, :]),
                        pk_ij[:, j],
                        extrap=0,
                        method=self.interpolation_order,
                    )
                    state["p_mij_real_space_shape_shape_grid"] = (
                        state["p_mij_real_space_shape_shape_grid"].at[m, j, :, i].set(p)
                    )
        return state

    def compute(self, state, params_values):
        if self.no_ia:
            return state

        if self.use_emulator:
            state = self.compute_emulator(state, params_values)
        else:
            state = self.compute_analytic(state)

        if self.heft_nla:
            state["p_mij_real_space_shape_shape_grid"] = (
                state["p_mij_real_space_shape_shape_grid"]
                .at[0, 0, ...]
                .set(2 / 3 * state["p_ij_real_space_bias_grid"][2, ...])
            )

        return state


class RealSpaceIAExpansion(LikelihoodModule):
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
        **config,
    ):
        """Combine 3d basis spectra into p_mm, p_gm and p_gg splines
        for each tracer that is requested.

        Args:
            observed_data_vector (DataVector): Measurement object containing tracer information.
            zmin (float, optional): Minimum redshift of spline. Defaults to 0.0.
            zmax (float, optional): Maximum redshift of spline. Defaults to 2.0.
            nz (int, optional): _description_. Defaults to 50.
            kmin (_type_, optional): _description_. Defaults to 1e-3.
            kmax (float, optional): _description_. Defaults to 3.95.
            nk (int, optional): _description_. Defaults to 200.
        """

        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.nz = nz
        self.nk = nk
        self.z = jnp.linspace(zmin, zmax, nz)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.z_evolution_model = config.get("z_evolution_model", "spline")
        self.spline_N = config.get("spline_N", 3)
        self.spline_Delta = config.get("spline_Delta", 0.6)
        self.spline_min = config.get("spline_min", 0.0)
        self.scale_by_s8z = config.get("scale_by_s8z", True)
        self.magnification_x_ia = config.get("include_magnification_x_ia", False)
        self.interpolation_order = config.get("interpolation_order", "cubic")
        self.save_spherical_harmonic_spectra = config.get(
            "save_spherical_harmonic_spectra", False
        )
        self.no_ia = config.get("no_ia", False)

        self.output_requirements = {}
        if self.scale_by_s8z:
            self.sigma8_fid = 0.81

        if self.save_spherical_harmonic_spectra:
            self.required_spectra = {
                "c_dk": ["p_gi"],
                "c_kk": ["p_ii_ee", "p_mi", "p_ii_22_0", "p_ii_22_1", "p_ii_22_2"],
                "c_bb": ["p_ii_bb", "p_ii_22_1"],
            }
        else:
            self.required_spectra = {
                "c_dk": ["p_gi"],
                "c_kk": ["p_ii_ee", "p_mi"],
                "c_bb": ["p_ii_bb"],
            }

        if self.magnification_x_ia:
            if self.z_evolution_model == "maiar":
                raise (
                    ValueError(
                        "Cannot make magnification x IA predictions using MAIAR model"
                    )
                )
            self.required_spectra["c_dk"].append("p_mi")

        self.spectrum_params = {
            "p_ii_ee": [
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
            ],
            "p_ii_bb": [
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
            ],
            "p_ii_22_0": [
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
            ],
            "p_ii_22_1": [
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
            ],
            "p_ii_22_2": [
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
            ],
            "p_mi": [
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
            ],
            "p_gi": [
                ["b_1", "b_2", "b_s", "b_3"],
                ["c_s", "c_ds", "c_s2", "c_L2", "c_3", "c_dt", "alpha_s"],
            ],
        }

        self.spectrum_basis = {
            "p_ii_ee": "p_mij_real_space_shape_shape_grid",
            "p_ii_22_0": "p_mij_real_space_shape_shape_grid",
            "p_ii_22_1": "p_mij_real_space_shape_shape_grid",
            "p_ii_22_2": "p_mij_real_space_shape_shape_grid",
            "p_ii_bb": "p_mij_real_space_shape_shape_grid",
            "p_mi": "p_ij_real_space_density_shape_grid",
            "p_gi": "p_ij_real_space_density_shape_grid",
        }

        # figure out what spectra we need to calculate,
        # and what parameters are required.
        self.all_spectra = {}
        self.output_requirements = {}
        self.sbins = []
        self.dbins = []

        for spec_type in self.spectrum_types:
            if spec_type in self.required_spectra.keys():
                spectra = self.required_spectra[spec_type]
                for s in spectra:
                    if s not in self.all_spectra:
                        self.all_spectra[s] = []
                        self.output_requirements[s] = []

                    for i, j in self.spectrum_info[spec_type]["bin_pairs"]:
                        if j not in self.sbins:
                            self.sbins.append(j)
                        if spec_type == "c_dk":
                            self.dbins.append(i)
                        pars = self.required_spectrum_params(spec_type, s, i, j)
                        for s_idx in pars:
                            if s_idx not in self.all_spectra[s]:
                                self.all_spectra[s].append(s_idx)
                                if len(pars[s_idx]) == 0:
                                    continue
                                self.output_requirements[s].extend(pars[s_idx][0])
                                self.output_requirements[s].extend(pars[s_idx][1])

                                self.output_requirements[s].append(
                                    self.spectrum_basis[s]
                                )
                                if self.scale_by_s8z:
                                    self.output_requirements[s].append("sigma8_z")

        self.dbins = jnp.unique(jnp.array(self.dbins))
        self.sbins = jnp.unique(jnp.array(self.sbins))
        self.n_dbins = self.dbins.shape[0]
        self.n_sbins = self.sbins.shape[0]
        if self.n_dbins > 0:
            self.n_dbins_tot = self.observed_data_vector.nz_d.shape[0]
        else:
            self.n_dbins_tot = 0

        self.n_sbins_tot = self.observed_data_vector.nz_s.shape[0]

        self.indexed_params = {}
        # set up params for indexing
        for s in self.all_spectra:
            self.indexed_params[s] = [[], []]
            if s == "p_mi":
                for i in range(self.observed_data_vector.nz_s.shape[0]):
                    pars = self.required_spectrum_params("c_kk", "p_mi", i, 0)[(i,)]
                    if len(pars) > 0:
                        self.indexed_params[s][1].append(pars[0])
                    else:
                        self.indexed_params[s][1].append([])

            elif s == "p_gi":
                for i in range(self.observed_data_vector.nz_d.shape[0]):
                    for j in range(self.observed_data_vector.nz_s.shape[0]):
                        pars = self.required_spectrum_params("c_dk", "p_gi", i, j)[
                            (i, j)
                        ]
                        if len(pars) > 0:
                            self.indexed_params[s][0].append(pars[0])
                            self.indexed_params[s][1].append(pars[1])
                        else:
                            self.indexed_params[s][0].append([])
                            self.indexed_params[s][1].append([])

            else:
                for i in range(self.observed_data_vector.nz_s.shape[0]):
                    for j in range(self.observed_data_vector.nz_s.shape[0]):
                        pars = self.required_spectrum_params("c_kk", "p_ii_ee", i, j)[
                            (i, j)
                        ]
                        if len(pars) > 0:
                            self.indexed_params[s][0].append(pars[0])
                            self.indexed_params[s][1].append(pars[1])
                        else:
                            self.indexed_params[s][0].append([])
                            self.indexed_params[s][1].append([])

            self.indexed_params[s][0] = np.array(self.indexed_params[s][0])
            self.indexed_params[s][1] = np.array(self.indexed_params[s][1])

    def required_spectrum_params(self, c_type, p_type, i, j):
        pars = {}
        if self.no_ia:
            if p_type == "p_mi":
                s_idx = (i,)
                pars[s_idx] = []
                if c_type == "c_kk":
                    s_idx = (j,)
                    pars[s_idx] = []
            else:
                s_idx = (i, j)
                pars[s_idx] = []

            return pars

        if self.z_evolution_model == "maiar":
            s_idx = (i, j)
            pars[s_idx] = []
            if p_type == "p_gi":
                pars[s_idx].append(
                    [
                        f"{p}_{i}" if ((i in self.dbins) & (j in self.sbins)) else "NA"
                        for p in self.spectrum_params[p_type][0]
                    ]
                )

                pars[s_idx].append(
                    [
                        f"{p}_{i}_{j}"
                        if ((i in self.dbins) & (j in self.sbins))
                        else "NA"
                        for p in self.spectrum_params[p_type][1]
                    ]
                )
            else:
                pars[s_idx].append(
                    [
                        (
                            f"{p}_{i}_{j}"
                            if ((i in self.sbins) & (j in self.sbins))
                            else "NA"
                        )
                        for p in self.spectrum_params[p_type][0]
                    ]
                )
                pars[s_idx].append(
                    [
                        (
                            f"{p}_{i}_{j}"
                            if ((i in self.sbins) & (j in self.sbins))
                            else "NA"
                        )
                        for p in self.spectrum_params[p_type][1]
                    ]
                )

            return pars

        elif self.z_evolution_model == "per_source_bin_spline":
            if (p_type == "p_mi") & (c_type == "c_kk"):
                s_idx = (i,)
                pars[s_idx] = [
                    [
                        f"{p}_{i}_{n}" if i in self.sbins else "NA"
                        for p in self.spectrum_params[p_type][0]
                        for n in range(self.spline_N)
                    ],
                    [],
                ]
                s_idx = (j,)
                pars[s_idx] = [
                    [],
                    [
                        f"{p}_{j}_{n}" if j in self.sbins else "NA"
                        for p in self.spectrum_params[p_type][0]
                        for n in range(self.spline_N)
                    ],
                ]

            else:
                s_idx = (i, j)
                pars[s_idx] = []
                if p_type == "p_gi":
                    pars[s_idx].append(
                        [
                            (
                                f"{p}_{i}"
                                if ((i in self.dbins) & (j in self.sbins))
                                else "NA"
                            )
                            for p in self.spectrum_params[p_type][0]
                        ]
                    )
                else:
                    pars[s_idx].append(
                        [
                            f"{p}_{i}_{n}" if (i in self.sbins) else "NA"
                            for p in self.spectrum_params[p_type][0]
                            for n in range(self.spline_N)
                        ]
                    )
                pars[s_idx].append(
                    [
                        f"{p}_{j}_{n}" if (j in self.sbins) else "NA"
                        for p in self.spectrum_params[p_type][1]
                        for n in range(self.spline_N)
                    ]
                )
        elif self.z_evolution_model == "spline":
            if (p_type == "p_mi") & (c_type == "c_kk"):
                s_idx = (i,)
                pars[s_idx] = [
                    [
                        f"{p}_{n}"
                        for p in self.spectrum_params[p_type][0]
                        for n in range(self.spline_N)
                    ],
                    [],
                ]
                if i != j:
                    s_idx = (j,)
                    pars[s_idx] = [
                        [],
                        [
                            f"{p}_{n}"
                            for p in self.spectrum_params[p_type][0]
                            for n in range(self.spline_N)
                        ],
                    ]

            else:
                s_idx = (i, j)
                pars[s_idx] = []
                if p_type == "p_gi":
                    pars[s_idx].append(
                        [
                            f"{p}_{i}" if i in self.dbins else "NA"
                            for p in self.spectrum_params[p_type][0]
                        ]
                    )
                else:
                    pars[s_idx].append(
                        [
                            f"{p}_{n}"
                            for p in self.spectrum_params[p_type][0]
                            for n in range(self.spline_N)
                        ]
                    )
                pars[s_idx].append(
                    [
                        f"{p}_{n}"
                        for p in self.spectrum_params[p_type][1]
                        for n in range(self.spline_N)
                    ]
                )

        elif self.z_evolution_model == "powerlaw":
            assert self.spline_N == 2, (
                "Powerlaw model interprets two spline nodes as powerlaw amplitude and slope."
            )

            if (p_type == "p_mi") & (c_type == "c_kk"):
                s_idx = (i,)
                pars[s_idx] = [
                    [
                        f"{p}_{n}"
                        for p in self.spectrum_params[p_type][0]
                        for n in range(self.spline_N)
                    ],
                    [],
                ]
                if i != j:
                    s_idx = (j,)
                    pars[s_idx] = [
                        [],
                        [
                            f"{p}_{n}"
                            for p in self.spectrum_params[p_type][0]
                            for n in range(self.spline_N)
                        ],
                    ]

            else:
                s_idx = (i, j)
                pars[s_idx] = []
                if p_type == "p_gi":
                    pars[s_idx].append(
                        [
                            f"{p}_{i}" if i in self.dbins else "NA"
                            for p in self.spectrum_params[p_type][0]
                        ]
                    )
                else:
                    pars[s_idx].append(
                        [
                            f"{p}_{n}"
                            for p in self.spectrum_params[p_type][0]
                            for n in range(self.spline_N)
                        ]
                    )
                pars[s_idx].append(
                    [
                        f"{p}_{n}"
                        for p in self.spectrum_params[p_type][1]
                        for n in range(self.spline_N)
                    ]
                )

        return pars

    def set_cs(self, param_vec, param_indices, zeff, s8z, z_evolution_model):
        if (z_evolution_model == "maiar") | (z_evolution_model == "scalar_bias"):
            c_vec = param_vec[param_indices]
            s8z_i = jnp.interp(zeff, self.z, s8z) / self.sigma8_fid

        elif (z_evolution_model == "spline") | (
            z_evolution_model == "per_source_bin_spline"
        ):
            param_indices = param_indices.reshape(-1, self.spline_N).T
            c_vec = spline_func_vec(
                self.z,
                param_vec[param_indices],
                self.spline_min,
                Delta=self.spline_Delta,
            )
            s8z_i = s8z / self.sigma8_fid

        elif z_evolution_model == "powerlaw":
            zeff = 0.62
            param_indices = param_indices.reshape(-1, 2).T
            c_vec = (
                param_indices[:, 0] * ((1 + self.z) / (1 + zeff)) ** param_indices[:, 1]
            )

        return c_vec, s8z_i

    def get_cs(self, p_type, i, j=None):
        if self.no_ia:
            return []

        if self.z_evolution_model == "maiar":
            c_vec = [f"{p}_{i}_{j}" for p in self.spectrum_params[p_type]][1]

        elif self.z_evolution_model == "spline":
            c_vec = [
                f"{p}_{i}_{n}"
                for n in range(self.spline_N)
                for p in self.spectrum_params[p_type][1]
            ]

        return c_vec

    def compute_p_ii_ee(self, state, xs):
        if self.no_ia:
            return jnp.zeros((self.logk.shape[0], self.z.shape[0]))

        cvec_i = xs[0]
        cvec_j = xs[1]
        s8z_i = xs[2]
        s8z_j = xs[3]
        p_mij = state["p_mij_real_space_shape_shape_grid"]

        pii0 = combine_shape_shape_spectra(
            self.k, p_mij[0, ...], cvec_i, cvec_j, s8z_i=s8z_i, s8z_j=s8z_j
        )

        pii2 = combine_shape_shape_spectra(
            self.k, p_mij[2, ...], cvec_i, cvec_j, s8z_i=s8z_i, s8z_j=s8z_j
        )

        p = 0.125 * (3 * pii0 + pii2)

        return p

    def compute_p_ii_22_0(self, state, xs):
        if self.no_ia:
            return jnp.zeros((self.logk.shape[0], self.z.shape[0]))

        cvec_i = xs[0]
        cvec_j = xs[1]
        s8z_i = xs[2]
        s8z_j = xs[3]
        p_mij = state["p_mij_real_space_shape_shape_grid"]

        pii0 = combine_shape_shape_spectra(
            self.k, p_mij[0, ...], cvec_i, cvec_j, s8z_i=s8z_i, s8z_j=s8z_j
        )

        return pii0

    def compute_p_ii_22_1(self, state, xs):
        if self.no_ia:
            return jnp.zeros((self.logk.shape[0], self.z.shape[0]))

        cvec_i = xs[0]
        cvec_j = xs[1]
        s8z_i = xs[2]
        s8z_j = xs[3]
        p_mij = state["p_mij_real_space_shape_shape_grid"]

        pii1 = combine_shape_shape_spectra(
            self.k, p_mij[1, ...], cvec_i, cvec_j, s8z_i=s8z_i, s8z_j=s8z_j
        )

        return pii1

    def compute_p_ii_22_2(self, state, xs):
        if self.no_ia:
            return jnp.zeros((self.logk.shape[0], self.z.shape[0]))

        cvec_i = xs[0]
        cvec_j = xs[1]
        s8z_i = xs[2]
        s8z_j = xs[3]
        p_mij = state["p_mij_real_space_shape_shape_grid"]

        pii2 = combine_shape_shape_spectra(
            self.k, p_mij[2, ...], cvec_i, cvec_j, s8z_i=s8z_i, s8z_j=s8z_j
        )

        return pii2

    def compute_p_ii_bb(self, state, xs):
        if self.no_ia:
            return jnp.zeros((self.logk.shape[0], self.z.shape[0]))

        cvec_i = xs[0]
        cvec_j = xs[1]
        s8z_i = xs[2]
        s8z_j = xs[3]
        p_mij = state["p_mij_real_space_shape_shape_grid"]

        pii1 = combine_shape_shape_spectra(
            self.k, p_mij[1, ...], cvec_i, cvec_j, s8z_i=s8z_i, s8z_j=s8z_j
        )

        return pii1

    def compute_p_mi(self, state, xs):
        if self.no_ia:
            return jnp.zeros((self.logk.shape[0], self.z.shape[0]))

        cvec = xs[0]
        s8z = xs[1]
        bvec = [0] * len(self.spectrum_params["p_gi"][0])
        bvec[0] = 1
        p_ij = state["p_ij_real_space_density_shape_grid"]
        s8z = state["sigma8_z"] / self.sigma8_fid

        pmi = combine_density_shape_spectra(
            self.k, p_ij[0, ...], jnp.array(bvec), cvec, s8z_d=None, s8z_s=s8z, b1e=True
        )

        return 3 * pmi / 4

    def compute_p_gi(self, state, xs):
        if self.no_ia:
            return jnp.zeros((self.logk.shape[0], self.z.shape[0]))

        bvec = xs[0]
        cvec = xs[1]
        s8z_d = xs[2]
        s8z_s = xs[3]
        p_ij = state["p_ij_real_space_density_shape_grid"]

        pgi = combine_density_shape_spectra(
            self.k, p_ij[0, ...], bvec, cvec, s8z_d=s8z_d, s8z_s=s8z_s, b1e=True
        )

        return 3 * pgi / 4

    def compute(self, state, params_values):
        param_vec = jnp.array(list(params_values.values()))
        n_sbins_tot = self.n_sbins_tot
        n_dbins_tot = self.n_dbins_tot

        for s in self.all_spectra:
            computer = getattr(self, f"compute_{s}")

            def fspec(carry, xs):
                p = computer(state, xs)
                return carry, p

            if s == "p_mi":

                def f(carry, x):
                    p_idx, s8z = x
                    return carry, self.set_cs(
                        param_vec, p_idx, None, s8z, self.z_evolution_model
                    )

                _, xs = scan(
                    f,
                    0,
                    (
                        self.param_indices[s][1],
                        jnp.tile(state["sigma8_z"], n_sbins_tot).reshape(
                            n_sbins_tot, -1
                        ),
                    ),
                )
                _, p = scan(fspec, None, xs)
                state[s] = p

            elif s == "p_gi":

                def f_i(carry, x):
                    p_idx, zeff, s8z = x
                    return carry, self.set_cs(
                        param_vec, p_idx, zeff, s8z, "scalar_bias"
                    )

                def f_j(carry, x):
                    p_idx, zeff, s8z = x
                    return carry, self.set_cs(
                        param_vec, p_idx, zeff, s8z, self.z_evolution_model
                    )

                _, (bias_params_i, s8z_i) = scan(
                    f_i,
                    0,
                    (
                        self.param_indices[s][0],
                        jnp.repeat(state["zeff_w_d"][:, None], n_sbins_tot),
                        jnp.tile(
                            state["sigma8_z"], len(self.param_indices[s][0])
                        ).reshape(len(self.param_indices[s][0]), -1),
                    ),
                )
                _, (bias_params_j, s8z_j) = scan(
                    f_j,
                    0,
                    (
                        self.param_indices[s][1],
                        jnp.repeat(state["zeff_w_d"][:, None], n_sbins_tot),
                        jnp.tile(
                            state["sigma8_z"], len(self.param_indices[s][0])
                        ).reshape(len(self.param_indices[s][0]), -1),
                    ),
                )

                xs = [bias_params_i, bias_params_j, s8z_i, s8z_j]
                _, p = scan(fspec, None, xs)
                state[s] = p.reshape(n_dbins_tot, n_sbins_tot, self.nk, self.nz)

            elif s in ["p_ii_ee", "p_ii_bb", "p_ii_22_0", "p_ii_22_1", "p_ii_22_2"]:

                def f_i(carry, x):
                    p_idx, zeff, s8z = x
                    return carry, self.set_cs(
                        param_vec, p_idx, zeff, s8z, self.z_evolution_model
                    )

                _, (bias_params_i, s8z_i) = scan(
                    f_i,
                    0,
                    (
                        self.param_indices[s][0],
                        jnp.repeat(state["zeff_w_ia"][:, None], n_sbins_tot),
                        jnp.tile(
                            state["sigma8_z"], len(self.param_indices[s][0])
                        ).reshape(len(self.param_indices[s][0]), -1),
                    ),
                )
                _, (bias_params_j, s8z_j) = scan(
                    f_i,
                    0,
                    (
                        self.param_indices[s][1],
                        jnp.repeat(state["zeff_w_ia"][:, None], n_sbins_tot),
                        jnp.tile(
                            state["sigma8_z"], len(self.param_indices[s][0])
                        ).reshape(len(self.param_indices[s][1]), -1),
                    ),
                )

                xs = [bias_params_i, bias_params_j, s8z_i, s8z_j]
                _, p = scan(fspec, None, xs)
                state[s] = p.reshape(n_sbins_tot, n_sbins_tot, self.nk, self.nz)

        return state


def combine_density_shape_spectra(
    k, spectra, bvec_d, bvec_ia, s8z_d=None, s8z_s=None, b1e=False
):
    """
    Combine density and intrinsic alignment shape spectra with bias parameters.
    
    Args:
        k: Wavenumber array
        spectra: Dictionary of power spectra components
        bvec_d: Density bias parameters [b1, b2, bs, b3]
        bvec_ia: Intrinsic alignment bias parameters [c_s, c_ds, c_s2, c_L2, c_3, c_dt, alpha_s]
        s8z_d: Optional density sigma8(z) normalization factor
        s8z_s: Optional shape sigma8(z) normalization factor
        b1e: Optional flag for b1 evolution (default: False)
    
    Returns:
        Dictionary of combined density-shape power spectra
    """
    b1, b2, bs, b3 = bvec_d  # absorb alpha_d into alpha_s for now.
    c_s, c_ds, c_s2, c_L2, c_3, c_dt, alpha_s = bvec_ia

    if s8z_d is not None:
        b1 = b1 / s8z_d
        b2 = b2 / s8z_d**2
        b3 = b3 / s8z_d**3
        # alpha_d = alpha_d / s8z_d**2

    if s8z_s is not None:
        c_s = c_s / s8z_s
        c_ds = c_ds / s8z_s**2
        c_s2 = c_s2 / s8z_s**2
        c_L2 = c_L2 / s8z_s**2
        c_3 = c_3 / s8z_s**3
        c_dt = c_dt / s8z_s**3
        alpha_s = alpha_s / s8z_s**2

    if b1e:
        b1 = b1 - 1

    # The table is listed in order (1, Oab), (delta, Oab), (s2, Oab)
    bias_poly = jnp.array(
        [
            c_s,
            c_ds,
            c_s2,
            c_L2,
            b1 * c_s,
            b1 * c_ds,
            b1 * c_s2,
            b1 * c_L2,
            b2 * c_s,
            b2 * c_ds,
            b2 * c_s2,
            b2 * c_L2,
            bs * c_s,
            bs * c_ds,
            bs * c_s2,
            bs * c_L2,
            c_3,
            b1 * c_3 + b3 * c_s,
            c_dt,
            b1 * c_dt,
            alpha_s,
        ]
    )

    if len(bias_poly.shape) == 1:
        bias_poly = bias_poly.reshape(-1, 1)

    return jnp.sum(bias_poly[:, None, :] * spectra, axis=0)


def combine_shape_shape_spectra(
    k, spectra, shape_bvec1, shape_bvec2, Pshot=0, s8z_i=None, s8z_j=None
):
    """
    Combine shape-shape spectra with intrinsic alignment bias parameters.
    
    Args:
        k: Wavenumber array
        spectra: Dictionary of power spectra components
        shape_bvec1: Shape bias parameters for first sample
        shape_bvec2: Shape bias parameters for second sample
        Pshot: Shot noise contribution (default: 0)
        s8z_i: Optional sigma8(z) normalization for first sample
        s8z_j: Optional sigma8(z) normalization for second sample
    
    Returns:
        Combined shape-shape power spectrum
    """
    # Here we have to specify spectra for a specific helicity

    c_s, c_ds, c_s2, c_L2, c_3, c_dt, alpha_s1 = shape_bvec1
    b_s, b_ds, b_s2, b_L2, b_3, b_dt, alpha_s2 = shape_bvec2

    if s8z_i is not None:
        c_s /= s8z_i
        c_ds /= s8z_i**2
        c_s2 /= s8z_i**2
        c_L2 /= s8z_i**2
        c_3 /= s8z_i**3
        c_dt /= s8z_i**3
        alpha_s1 /= s8z_i**2

    if s8z_j is not None:
        b_s /= s8z_j
        b_ds /= s8z_j**2
        b_s2 /= s8z_j**2
        b_L2 /= s8z_j**2
        b_3 /= s8z_j**3
        b_dt /= s8z_j**3
        alpha_s2 /= s8z_j**2

    # The table is listed in order (s_ab, Ocd), (delta sab, Ocd), (s^2_ab, Ocd)
    bias_poly = jnp.array(
        [
            c_s * b_s,
            c_s * b_ds + b_s * c_ds,
            c_s * b_s2 + b_s * c_s2,
            c_s * b_L2 + b_s * c_L2,
            c_ds * b_ds,
            c_ds * b_s2 + c_s2 * b_ds,
            c_ds * b_L2 + c_L2 * b_ds,
            c_s2 * b_s2,
            c_s2 * b_L2 + c_L2 * b_s2,
            c_L2 * b_L2,
            c_s * b_3 + c_3 * b_s,
            c_s * b_dt + c_dt * b_s,
            alpha_s1 + alpha_s2,
        ]
    )

    if len(bias_poly.shape) == 1:
        bias_poly = bias_poly.reshape(-1, 1)

    p = jnp.sum(bias_poly[:, None, :] * spectra, axis=0)

    #    add_sn = lambda p, sn_a: p + sn_a
    # noadd_sn = lambda p, sn_a: p
    #    p = cond(jnp.all(eps_s1==eps_s2), add_sn, noadd_sn, p, 2 * eps_s1)

    return p
