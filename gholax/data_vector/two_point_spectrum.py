from .data_vector import DataVector
from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.integrate import trapezoid
from glob import glob
import jax.numpy as jnp
import numpy as np
import warnings
import yaml


datavector_requires = {
    "c_cmbkcmbk": [],
    "c_dcmbk": ["nz_d"],
    "c_kk": ["nz_s"],
    "c_bb": ["nz_s"],
    "c_dk": ["nz_d", "nz_s"],
    "c_dd": ["nz_d"],
}

covariance_field_types = {
    "c_cmbkcmbk": ["cmbk", "cmbk"],
    "c_dcmbk": ["d", "cmbk"],
    "c_kk": ["k", "k"],
    "c_bb": ["b", "b"],
    "c_dk": ["d", "k"],
    "c_dd": ["d", "d"],
}

field_types = {
    "c_cmbkcmbk": ["cmbk", "cmbk"],
    "c_dcmbk": ["d", "cmbk"],
    "c_kk": ["gamma_e", "gamma_e"],
    "c_bb": ["gamma_b", "gamma_b"],
    "c_dk": ["d", "gamma_e"],
    "c_dd": ["d", "d"],
}


class TwoPointSpectrum(DataVector):
    def __init__(
        self,
        data_vector_info_filename,
        spectrum_info,
        covariance_info=None,
        scale_cuts=None,
        zeff_weighting=True,
        dummy_cov=False,
        generate_data_vector=False,
        zmin=0,
        zmax=2.0,
        nz=125,
    ):
        self.data_vector_info_filename = data_vector_info_filename
        self.spectrum_info = spectrum_info
        self.scale_cuts = scale_cuts
        self.spectrum_types = list(self.spectrum_info.keys())
        self.zeff_weighting = zeff_weighting
        self.dummy_cov = dummy_cov
        self.generate_data_vector = generate_data_vector
        self.covariance_info = covariance_info
        self.z = jnp.linspace(zmin, zmax, nz)

    def load_data(self):
        self.load_data_vector()
        self.load_requirements()
        self.setup_scale_cuts()

        if not self.dummy_cov:
            self.load_covariance_matrix()
        else:
            self.cinv = None

    #        print(self.cW, flush=True)

    def generate_data(self):
        required_spectra = []
        for si in self.spectrum_info:
            for sj in self.spectrum_info:
                c0 = f"c_{covariance_field_types[si][0]}{covariance_field_types[sj][0]}"
                if c0 not in field_types:
                    c0 = f"c_{covariance_field_types[sj][0]}{covariance_field_types[si][0]}"

                c1 = f"c_{covariance_field_types[si][1]}{covariance_field_types[sj][1]}"
                if c1 not in field_types:
                    c1 = f"c_{covariance_field_types[sj][1]}{covariance_field_types[si][1]}"

                c2 = f"c_{covariance_field_types[si][0]}{covariance_field_types[sj][1]}"
                if c2 not in field_types:
                    c2 = f"c_{covariance_field_types[sj][1]}{covariance_field_types[si][0]}"

                c3 = f"c_{covariance_field_types[si][1]}{covariance_field_types[sj][0]}"
                if c3 not in field_types:
                    c3 = f"c_{covariance_field_types[sj][0]}{covariance_field_types[si][1]}"

                required_spectra.extend([c0, c1, c2, c3])
        required_spectra = np.unique(required_spectra)

        bin_pairs = {}
        n_bins = 0
        for t in required_spectra:
            bin_pairs[t] = []
            for i in self.spectrum_info[t]["bins0"]:
                use_cross = self.spectrum_info[t].get("use_cross", True)
                if use_cross:
                    for j in self.spectrum_info[t]["bins1"]:
                        if (
                            covariance_field_types[t][0] != covariance_field_types[t][1]
                        ) | (j >= i):
                            bin_pairs[t].append((i, j))
                            n_bins += 1
                else:
                    j = i
                    bin_pairs[t].append((i, j))
                    n_bins += 1

        ells = np.arange(3 * 2048, dtype="int32")  # Array of multipoles
        weights = np.zeros(len(ells))  # Array of weights
        bpws = np.zeros_like(ells) - 1  # Array of bandpower indices
        ell_start = 25
        delta_ell = 30

        i = 0
        counter = ell_start
        delta_ell = int(3 * np.sqrt(ell_start))
        bpw_widths = [delta_ell]

        while counter + delta_ell < ells.shape[0]:
            bpws[counter : counter + delta_ell] = i
            weights[counter : counter + delta_ell] = 1 / float(delta_ell)
            counter = counter + delta_ell
            delta_ell = int(3 * np.sqrt(ells[counter]))
            bpw_widths.append(delta_ell)
            i += 1

        self.delta_ell = np.array(bpw_widths)[:-1]
        ell_eff = np.bincount(bpws + 1, weights=ells * (2 * ells + 1)) / np.bincount(
            bpws + 1, weights=(2 * ells + 1)
        )
        ell_eff = ell_eff[1:]
        window = np.zeros((len(ell_eff), len(ells)))
        self.ell_eff = ell_eff
        for i, ell in enumerate(ell_eff):
            window[i, bpws == i] = 1 / float(self.delta_ell[i])

        dt = np.dtype(
            [
                ("spectrum_type", "U10"),
                ("zbin0", int),
                ("zbin1", int),
                ("separation", float),
                ("value", float),
            ]
        )
        spectra = np.zeros(n_bins * len(ell_eff), dtype=dt)
        self.cW = {}
        counter = 0
        for t in required_spectra:
            self.cW[t] = {}
            for bin_pair in bin_pairs[t]:
                spectra[counter : counter + ell_eff.shape[0]]["spectrum_type"] = t
                spectra[counter : counter + ell_eff.shape[0]]["zbin0"] = bin_pair[0]
                spectra[counter : counter + ell_eff.shape[0]]["zbin1"] = bin_pair[1]
                spectra[counter : counter + ell_eff.shape[0]]["separation"] = ell_eff

                self.cW[t][f"{bin_pair[0]}_{bin_pair[1]}"] = window
                counter += ell_eff.shape[0]
        #            print(self.cW[t], flush=True)

        return spectra

    def process_spectrum_info(self, spectra):
        self.spectra = []
        for t in self.spectrum_types:
            n_bins0_tot = len(
                np.unique(spectra[spectra["spectrum_type"] == t]["zbin0"])
            )
            n_bins1_tot = len(
                np.unique(spectra[spectra["spectrum_type"] == t]["zbin1"])
            )
            if "use_cross" not in self.spectrum_info[t]:
                self.spectrum_info[t]["use_cross"] = True

            if "bins0" not in self.spectrum_info[t]:
                warnings.warn(
                    f"bins0 not specified for spectrum type {t}, using all bin0 in file",
                    UserWarning,
                )
                self.spectrum_info[t]["bins0"] = np.unique(
                    spectra[spectra["spectrum_type"] == t]["zbin0"]
                )

            if "bins1" not in self.spectrum_info[t]:
                warnings.warn(
                    f"bins1 not specified for spectrum type {t}, using all bin1 in file",
                    UserWarning,
                )
                self.spectrum_info[t]["bins1"] = np.unique(
                    spectra[spectra["spectrum_type"] == t]["zbin1"]
                )

            if not self.spectrum_info[t]["use_cross"]:
                assert np.all(
                    self.spectrum_info[t]["bins0"] == self.spectrum_info[t]["bins1"]
                )

            # get rid of bins we don't want
            if self.spectrum_info[t]["use_cross"]:
                idx = (
                    (spectra["spectrum_type"] == t)
                    & (np.in1d(spectra["zbin0"], self.spectrum_info[t]["bins0"]))
                    & (np.in1d(spectra["zbin1"], self.spectrum_info[t]["bins1"]))
                )
                self.spectra.append(spectra[idx])
            else:
                for ii, i in enumerate(self.spectrum_info[t]["bins0"]):
                    if ii == 0:
                        idx = (
                            (spectra["spectrum_type"] == t)
                            & (spectra["zbin0"] == i)
                            & (spectra["zbin1"] == i)
                        )
                    else:
                        idx |= (
                            (spectra["spectrum_type"] == t)
                            & (spectra["zbin0"] == i)
                            & (spectra["zbin1"] == i)
                        )

                self.spectra.append(spectra[idx])

            self.spectrum_info[t]["bin_pairs"] = []
            for i in self.spectrum_info[t]["bins0"]:
                if self.spectrum_info[t]["use_cross"]:
                    for j in self.spectrum_info[t]["bins1"]:
                        idx = (
                            (spectra["spectrum_type"] == t)
                            & (spectra["zbin0"] == i)
                            & (spectra["zbin1"] == j)
                        )
                        if np.sum(idx) > 0:
                            self.spectrum_info[t]["bin_pairs"].append((i, j))
                else:
                    j = i
                    idx = (
                        (spectra["spectrum_type"] == t)
                        & (spectra["zbin0"] == i)
                        & (spectra["zbin1"] == j)
                    )
                    if np.sum(idx) > 0:
                        self.spectrum_info[t]["bin_pairs"].append((i, j))

            idx = spectra["spectrum_type"] == t
            z0 = spectra["zbin0"][idx][0]
            z1 = spectra["zbin1"][idx][0]
            idx &= (spectra["zbin0"] == z0) & (spectra["zbin1"] == z1)
            ndv_per_bin = np.sum(idx)
            sep_unmasked = spectra[idx]["separation"]

            self.spectrum_info[t].update(
                {
                    "n_dv_per_bin": ndv_per_bin,
                    "separation": sep_unmasked,
                    "n_bins0_tot": n_bins0_tot,
                    "n_bins1_tot": n_bins1_tot,
                }
            )

        self.spectra = np.hstack(self.spectra)
        self.spectrum_values = jnp.array(self.spectra["value"])
        self.n_dv = len(self.spectra)

    def load_data_vector(self):
        """Loads the required data."""

        with open(self.data_vector_info_filename, "r") as fp:
            self.data_vector_info = yaml.load(fp, Loader=yaml.SafeLoader)

        if not self.generate_data_vector:
            # must specify a file with the actual correlation functions
            # in it. Should have the following columns: datavector type
            # (e.g. p0), redshift bin number for each sample being
            # correlated, separation values (e.g. k, ell, etc.), and
            # actual values for the spectra/correlation functions
            dt = np.dtype(
                [
                    ("spectrum_type", "U10"),
                    ("zbin0", int),
                    ("zbin1", int),
                    ("separation", float),
                    ("value", float),
                ]
            )

            spectra = np.genfromtxt(
                self.data_vector_info["spectra_filename"], names=True, dtype=dt
            )
        else:
            spectra = self.generate_data()

        self.process_spectrum_info(spectra)

    def load_requirements(self):
        requirements = []
        for t in self.spectrum_info:
            requirements = datavector_requires[t]
            if self.zeff_weighting:
                if t == "c_dk":
                    requirements.append("nz_d_dk")
                elif t == "c_dcmbk":
                    requirements.append("nz_d_dcmbk")
            for r in requirements:
                nz_ = np.genfromtxt(self.data_vector_info[r], dtype=None, names=None)
                nbins = nz_.shape[1] - 1
                nz = jnp.zeros((nbins, len(self.z)))

                for i in range(nbins):
                    idx = nz_[:, i + 1] > 0
                    nz = nz.at[i, :].set(
                        RegularGridInterpolator(
                            [(nz_[idx, 0])],
                            np.atleast_2d(nz_[idx, i + 1]).T,
                            fill_value=0,
                        )(self.z)[:, 0]
                    )

                nz = nz / trapezoid(nz, x=self.z, axis=-1)[:, None]
                self.spectrum_info[t][r] = nz
                setattr(self, r, nz)

        if ("cell_windows" in self.data_vector_info.keys()) & (
            not self.generate_data_vector
        ):
            window_matrix_files = self.data_vector_info["cell_windows"]

            self.cW = {}
            for k in list(window_matrix_files.keys()):
                if k not in self.spectrum_types:
                    continue

                self.cW[k] = {}

                if isinstance(window_matrix_files[k], dict):
                    # want per bin windows here
                    for ij in list(window_matrix_files[k].keys()):
                        i = int(ij.split("_")[0])
                        j = int(ij.split("_")[1])

                        self.cW[k][ij] = np.loadtxt(window_matrix_files[k][ij])
                else:
                    basefile = window_matrix_files[k]
                    all_window_files = glob(basefile)
                    spbasefile = basefile.split("*")
                    ijs = [
                        f.split(spbasefile[0])[-1].split(spbasefile[1])[0]
                        for f in all_window_files
                    ]
                    for i, ij in enumerate(ijs):
                        self.cW[k][ij] = np.loadtxt(all_window_files[i])

    def setup_scale_cuts(self):
        # make scale cut mask
        if self.scale_cuts is not None:
            for t in self.spectrum_info:
                if t in self.scale_cuts:
                    scale_cut_dict = self.scale_cuts[t]
                    scale_cut_mask = {}
                    sep_unmasked = self.spectrum_info[t]["separation"]
                    for ii, i in enumerate(self.spectrum_info[t]["bins0"]):
                        if self.spectrum_info[t]["use_cross"]:
                            if field_types[t][0] == field_types[t][1]:
                                bins1 = self.spectrum_info[t]["bins1"][ii:]
                            else:
                                bins1 = self.spectrum_info[t]["bins1"][:]
                            for jj, j in enumerate(bins1):
                                try:
                                    sep_min, sep_max = scale_cut_dict[
                                        "{}_{}".format(i, j)
                                    ]
                                    mask = (sep_min <= sep_unmasked) & (
                                        sep_unmasked <= sep_max
                                    )
                                    scale_cut_mask["{}_{}".format(i, j)] = mask
                                except:
                                    raise ValueError(
                                        "Scale cuts not provided for {} bin pair {},{}".format(
                                            t, i, j
                                        )
                                    )
                        else:
                            try:
                                sep_min, sep_max = scale_cut_dict["{}_{}".format(i, i)]
                                mask = (sep_min <= sep_unmasked) & (
                                    sep_unmasked <= sep_max
                                )
                                scale_cut_mask["{}_{}".format(i, i)] = mask
                            except:
                                raise ValueError(
                                    "Scale cuts not provided for {} bin pair {},{}".format(
                                        t, i, i
                                    )
                                )

                    self.spectrum_info[t]["scale_cut_masks"] = scale_cut_mask

                else:
                    raise ValueError("No scale cuts specified for {}".format(t))
        else:
            warnings.warn("No scale cuts specified for any spectra!", UserWarning)

            for t in self.spectrum_info:
                self.spectrum_info[t]["scale_cut_masks"] = None

        zbin_counter = {}
        scale_mask = []
        for t0 in self.spectrum_types:
            zbin_counter[t0] = []
            for zb0 in self.spectrum_info[t0]["bins0"]:
                for zb1 in self.spectrum_info[t0]["bins1"]:
                    if (zb0, zb1) in zbin_counter[t0]:
                        continue

                    idxi = np.where(
                        (self.spectra["spectrum_type"] == t0)
                        & (self.spectra["zbin0"] == zb0)
                        & (self.spectra["zbin1"] == zb1)
                    )[0]

                    try:
                        start_idx = np.min(idxi)
                    except ValueError as e:
                        continue

                    # mask scales
                    if self.spectrum_info[t0]["scale_cut_masks"] is not None:
                        mask_i = np.where(
                            self.spectrum_info[t0]["scale_cut_masks"][
                                "{}_{}".format(zb0, zb1)
                            ]
                        )[0]
                    else:
                        mask_i = np.arange(self.spectrum_info[t0]["n_dv_per_bin"])

                    scale_mask.extend((mask_i + start_idx).tolist())

        self.scale_mask = jnp.unique(jnp.array(scale_mask))
        self.scale_mask.sort()
        self.n_dv_masked = len(self.scale_mask)
        self.measured_spectra = jnp.array(self.spectra["value"])

    def load_covariance_matrix(self):
        # Always need a covariance matrix. This should be a text file
        # with columns specifying the two data vector types, and four redshift
        # bin indices for each element, as well as a column for the elements
        # themselves
        dt = np.dtype(
            [
                ("spectrum_type0", "U10"),
                ("spectrum_type1", "U10"),
                ("zbin00", int),
                ("zbin01", int),
                ("zbin10", int),
                ("zbin11", int),
                ("separation0", float),
                ("separation1", float),
                ("value", float),
            ]
        )

        cov_raw = np.loadtxt(self.data_vector_info["covariance_filename"], dtype=dt)
        cov_raw = cov_raw.reshape(
            int(cov_raw.shape[0] ** 0.5), int(cov_raw.shape[0] ** 0.5)
        )

        cov_slice = cov_raw[0, :]
        idxi = np.zeros(len(self.spectra), dtype=int)

        for i in range(self.n_dv):
            idx = np.where(
                (
                    (cov_slice["spectrum_type1"] == self.spectra[i]["spectrum_type"])
                    & (cov_slice["zbin10"] == self.spectra[i]["zbin0"])
                    & (cov_slice["zbin11"] == self.spectra[i]["zbin1"])
                    & (cov_slice["separation1"] == self.spectra[i]["separation"])
                )
            )[0]
            if len(idx) > 1:
                raise (ValueError)
            elif len(idx) < 1:
                raise (
                    ValueError(
                        "No matching cov entry for {}, {}, {}, {}".format(
                            self.spectra[i]["spectrum_type"],
                            self.spectra[i]["zbin0"],
                            self.spectra[i]["zbin1"],
                            self.spectra[i]["separation"],
                        )
                    )
                )

            idxi[i] = idx[0]

        covidx, covidy = np.meshgrid(idxi, idxi, indexing="ij")
        self.cov = cov_raw[covidx, covidy]
        assert np.allclose(self.cov["value"], self.cov["value"].T, 1e-16)

        cov_scale_mask_i, cov_scale_mask_j = np.meshgrid(
            self.scale_mask, self.scale_mask, indexing="ij"
        )
        self.cinv = jnp.linalg.inv(
            self.cov["value"][cov_scale_mask_i, cov_scale_mask_j].reshape(
                self.n_dv_masked, self.n_dv_masked
            )
        )

    #        if jnp.any(jnp.linalg.eigvals(self.cinv) < 0):
    #            raise(ValueError('APS covariance matrix not PSD.'))

    def gaussian_variance(self, si, sj, z00, z01, z10, z11):
        c0 = f"c_{covariance_field_types[si][0]}{covariance_field_types[sj][0]}"
        if c0 not in field_types:
            c0 = f"c_{covariance_field_types[sj][0]}{covariance_field_types[si][0]}"

        c1 = f"c_{covariance_field_types[si][1]}{covariance_field_types[sj][1]}"
        if c1 not in field_types:
            c1 = f"c_{covariance_field_types[sj][1]}{covariance_field_types[si][1]}"

        c2 = f"c_{covariance_field_types[si][0]}{covariance_field_types[sj][1]}"
        if c2 not in field_types:
            c2 = f"c_{covariance_field_types[sj][1]}{covariance_field_types[si][0]}"

        c3 = f"c_{covariance_field_types[si][1]}{covariance_field_types[sj][0]}"
        if c3 not in field_types:
            c3 = f"c_{covariance_field_types[sj][0]}{covariance_field_types[si][1]}"

        spec_w_n = []
        for spec, za, zb, f0, f1 in zip(
            [c0, c1, c2, c3],
            [z00, z01, z00, z01],
            [z10, z11, z11, z10],
            [
                covariance_field_types[si][0],
                covariance_field_types[si][1],
                covariance_field_types[si][0],
                covariance_field_types[si][1],
            ],
            [
                covariance_field_types[sj][0],
                covariance_field_types[sj][1],
                covariance_field_types[sj][1],
                covariance_field_types[sj][0],
            ],
        ):
            if (za, zb) in self.spectrum_info[spec]["bin_pairs"]:
                c_w_n = self.spectra["value"][
                    (self.spectra["spectrum_type"] == spec)
                    & (self.spectra["zbin0"] == za)
                    & (self.spectra["zbin1"] == zb)
                ] + float(self.covariance_info[spec][f"{za}_{zb}"]["noise"])
            elif covariance_field_types[spec][0] == covariance_field_types[spec][1]:
                c_w_n = self.spectra["value"][
                    (self.spectra["spectrum_type"] == spec)
                    & (self.spectra["zbin0"] == zb)
                    & (self.spectra["zbin1"] == za)
                ] + float(self.covariance_info[spec][f"{zb}_{za}"]["noise"])
            elif f1 == "d":
                c_w_n = self.spectra["value"][
                    (self.spectra["spectrum_type"] == spec)
                    & (self.spectra["zbin0"] == zb)
                    & (self.spectra["zbin1"] == za)
                ] + float(self.covariance_info[spec][f"{zb}_{za}"]["noise"])
            else:
                raise (
                    ValueError(f"No spectrum {spec} with zbin comination {za}, {zb}")
                )

            spec_w_n.append(c_w_n)

        var = (spec_w_n[0] * spec_w_n[1] + spec_w_n[2] * spec_w_n[3]) / (
            float(self.covariance_info["f_sky"]) * self.delta_ell * (2 * self.ell_eff)
        )

        return var

    def gaussian_covariance(self):
        dt = np.dtype(
            [
                ("spectrum_type0", "U10"),
                ("spectrum_type1", "U10"),
                ("zbin00", int),
                ("zbin01", int),
                ("zbin10", int),
                ("zbin11", int),
                ("separation0", float),
                ("separation1", float),
                ("value", float),
            ]
        )

        cov = np.zeros((self.n_dv, self.n_dv), dtype=dt)
        counter_i = 0
        for si in self.spectrum_info:
            n_ell_i = self.spectrum_info[si]["n_dv_per_bin"]
            for z00, z01 in self.spectrum_info[si]["bin_pairs"]:
                counter_j = 0
                for sj in self.spectrum_info:
                    n_ell_j = self.spectrum_info[sj]["n_dv_per_bin"]
                    for z10, z11 in self.spectrum_info[sj]["bin_pairs"]:
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["spectrum_type0"] = si
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["spectrum_type1"] = sj
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["zbin00"] = z00
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["zbin01"] = z01
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["zbin10"] = z10
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["zbin11"] = z11
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["separation0"] = self.spectrum_info[si]["separation"][:, None]
                        cov[
                            counter_i : counter_i + n_ell_i,
                            counter_j : counter_j + n_ell_j,
                        ]["separation1"] = self.spectrum_info[sj]["separation"][None, :]

                        var = self.gaussian_variance(si, sj, z00, z01, z10, z11)
                        np.fill_diagonal(
                            cov[
                                counter_i : counter_i + n_ell_i,
                                counter_j : counter_j + n_ell_j,
                            ]["value"],
                            var,
                        )
                        counter_j += n_ell_j

                counter_i += n_ell_i

        return cov
