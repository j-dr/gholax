import warnings

import h5py as h5
import jax.numpy as jnp
import numpy as np
from jax.scipy.integrate import trapezoid
from jax.scipy.interpolate import RegularGridInterpolator

from .data_vector import DataVector

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
    """Data vector for two-point angular power spectra (C_ell).

    Handles loading observed spectra and covariance from HDF5, applying
    scale cuts, computing Gaussian covariance matrices, interpolating
    redshift distributions, and managing bandpower window functions.
    """

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
        """Initialize the two-point spectrum data vector.

        Args:
            data_vector_info_filename: Path to the HDF5 file with spectra and covariance.
            spectrum_info: Dict of spectrum type configs (bins, cross-correlations, etc.).
            covariance_info: Optional dict with f_sky and noise for Gaussian covariance.
            scale_cuts: Optional dict of (ell_min, ell_max) per bin pair per spectrum type.
            zeff_weighting: If True, load extra n(z) for effective-z weighting.
            dummy_cov: If True, skip loading the covariance matrix.
            generate_data_vector: If True, generate a synthetic data vector.
            zmin: Minimum redshift for n(z) interpolation grid.
            zmax: Maximum redshift for n(z) interpolation grid.
            nz: Number of redshift grid points.
        """
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
        """Load observed data, requirements, scale cuts, and covariance."""
        self.load_data_vector()
        self.load_requirements()
        self.setup_scale_cuts()

        if not self.dummy_cov:
            self.load_covariance_matrix()
        else:
            self.cinv = None

    #        print(self.cW, flush=True)

    def _compute_ell_binning(self):
        """Compute the bandpower ell_eff and delta_ell arrays and store them as
        instance attributes.  Returns the bpws index array so callers that also
        need to build a window matrix can reuse it."""
        ells = np.arange(3 * 2048, dtype="int32")
        bpws = np.zeros_like(ells) - 1

        i = 0
        counter = 25  # ell_start
        delta_ell = int(3 * np.sqrt(counter))
        bpw_widths = [delta_ell]

        while counter + delta_ell < ells.shape[0]:
            bpws[counter : counter + delta_ell] = i
            counter = counter + delta_ell
            delta_ell = int(3 * np.sqrt(ells[counter]))
            bpw_widths.append(delta_ell)
            i += 1

        self.delta_ell = np.array(bpw_widths)[:-1]
        ell_eff = np.bincount(bpws + 1, weights=ells * (2 * ells + 1)) / np.bincount(
            bpws + 1, weights=(2 * ells + 1)
        )
        self.ell_eff = ell_eff[1:]
        return bpws

    def generate_data(self):
        """Generate a synthetic data vector with bandpower binning and window matrices.

        Returns:
            Structured numpy array of spectra with fields (spectrum_type,
            zbin0, zbin1, separation, value).
        """
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

        bpws = self._compute_ell_binning()
        ell_eff = self.ell_eff
        window = np.zeros((len(ell_eff), 3 * 2048))
        for i, ell in enumerate(ell_eff):
            window[i, bpws == i] = 1 / float(self.delta_ell[i])

        dt = np.dtype(
            [
                ("spectrum_type", "S10"),
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
        """Parse spectrum metadata, apply bin selection, and populate spectrum_info.

        Args:
            spectra: Structured numpy array of spectra from the data file.
        """
        self.spectra = []
        for t in self.spectrum_types:
            n_bins0_tot = len(
                np.unique(spectra[spectra["spectrum_type"] == t.encode('utf-8')]["zbin0"])
            )
            n_bins1_tot = len(
                np.unique(spectra[spectra["spectrum_type"] == t.encode('utf-8')]["zbin1"])
            )
            if "use_cross" not in self.spectrum_info[t]:
                self.spectrum_info[t]["use_cross"] = True

            if "bins0" not in self.spectrum_info[t]:
                warnings.warn(
                    f"bins0 not specified for spectrum type {t}, using all bin0 in file",
                    UserWarning,
                )
                self.spectrum_info[t]["bins0"] = np.unique(
                    spectra[spectra["spectrum_type"] == t.encode('utf-8')]["zbin0"]
                )

            if "bins1" not in self.spectrum_info[t]:
                warnings.warn(
                    f"bins1 not specified for spectrum type {t}, using all bin1 in file",
                    UserWarning,
                )
                self.spectrum_info[t]["bins1"] = np.unique(
                    spectra[spectra["spectrum_type"] == t.encode('utf-8')]["zbin1"]
                )

            if not self.spectrum_info[t]["use_cross"]:
                assert np.all(
                    self.spectrum_info[t]["bins0"] == self.spectrum_info[t]["bins1"]
                )

            # get rid of bins we don't want
            if self.spectrum_info[t]["use_cross"]:
                idx = (
                    (spectra["spectrum_type"] == t.encode('utf-8'))
                    & (np.in1d(spectra["zbin0"], self.spectrum_info[t]["bins0"]))
                    & (np.in1d(spectra["zbin1"], self.spectrum_info[t]["bins1"]))
                )
                self.spectra.append(spectra[idx])
            else:
                for ii, i in enumerate(self.spectrum_info[t]["bins0"]):
                    if ii == 0:
                        idx = (
                            (spectra["spectrum_type"] == t.encode('utf-8'))
                            & (spectra["zbin0"] == i)
                            & (spectra["zbin1"] == i)
                        )
                    else:
                        idx |= (
                            (spectra["spectrum_type"] == t.encode('utf-8'))
                            & (spectra["zbin0"] == i)
                            & (spectra["zbin1"] == i)
                        )

                self.spectra.append(spectra[idx])

            self.spectrum_info[t]["bin_pairs"] = []
            for i in self.spectrum_info[t]["bins0"]:
                if self.spectrum_info[t]["use_cross"]:
                    for j in self.spectrum_info[t]["bins1"]:
                        idx = (
                            (spectra["spectrum_type"] == t.encode('utf-8'))
                            & (spectra["zbin0"] == i)
                            & (spectra["zbin1"] == j)
                        )
                        if np.sum(idx) > 0:
                            self.spectrum_info[t]["bin_pairs"].append((i, j))
                else:
                    j = i
                    idx = (
                        (spectra["spectrum_type"] == t.encode('utf-8'))
                        & (spectra["zbin0"] == i)
                        & (spectra["zbin1"] == j)
                    )
                    if np.sum(idx) > 0:
                        self.spectrum_info[t]["bin_pairs"].append((i, j))

            idx = spectra["spectrum_type"] == t.encode('utf-8')
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

        self.data_vector_info = h5.File(self.data_vector_info_filename, "r")

        if not self.generate_data_vector:
            spectra = self.data_vector_info["spectra"][:]
        else:
            spectra = self.generate_data()

        self.process_spectrum_info(spectra)
        
    def save_data_vector(self, filename, model):
        """Save a model prediction as a new data vector HDF5 file.

        Args:
            filename: Output HDF5 file path.
            model: Array of model values to store as the 'value' field.
        """
        with h5.File(filename, "w") as f:
            dt = np.dtype(
                [
                    ("spectrum_type", "S10"),
                    ("zbin0", int),
                    ("zbin1", int),
                    ("separation", float),
                    ("value", float),
                ]
            )
            data = np.zeros(len(model), dtype=dt)
            data["spectrum_type"] = self.spectra["spectrum_type"]
            data["zbin0"] = self.spectra["zbin0"]
            data["zbin1"] = self.spectra["zbin1"]
            data["separation"] = self.spectra["separation"]
            data["value"] = model

            f.create_dataset("spectra", data=data)
            try:
                f.create_dataset("covariance", data=self.cov.flatten())
            except AttributeError:
                print("No covariance matrix to save.")

            for k_i in self.data_vector_info.keys():
                if k_i in ["spectra", "covariance"]:
                    continue
                elif 'window' in k_i:
                    grp = f.create_group(k_i)
                    for k_j in self.data_vector_info[k_i].keys():
                        grpp = grp.create_group(k_j)
                        for k_k in self.data_vector_info[k_i][k_j].keys():
                            grpp.create_dataset(k_k, data=self.data_vector_info[k_i][k_j][k_k][:])
                else:
                    f.create_dataset(k_i, data=self.data_vector_info[k_i][:])

    def load_requirements(self):
        """Load redshift distributions and bandpower window matrices from the data file."""
        requirements = []
        for t in self.spectrum_info:
            requirements = datavector_requires[t]
            if self.zeff_weighting:
                if t == "c_dk":
                    requirements.append("nz_d_dk")
                elif t == "c_dcmbk":
                    requirements.append("nz_d_dcmbk")
            for r in requirements:
                nz_ = self.data_vector_info[r][:]
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
                for ij in list(window_matrix_files[k].keys()):
                    self.cW[k][ij] = window_matrix_files[k][ij][:]

    def setup_scale_cuts(self):
        """Build per-bin-pair scale cut masks and the combined scale_mask index array."""
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
                        (self.spectra["spectrum_type"] == t0.encode('utf-8'))
                        & (self.spectra["zbin0"] == zb0)
                        & (self.spectra["zbin1"] == zb1)
                    )[0]

                    try:
                        start_idx = np.min(idxi)
                    except ValueError:
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
        """Load the covariance matrix from the data file and compute its inverse.

        Matches covariance entries to the current spectrum ordering and applies
        scale-cut masking before inverting.
        """
        cov_raw = self.data_vector_info["covariance"][:]
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

    def _ensure_covariance_info(self):
        """Prompt interactively for any f_sky or noise terms missing from covariance_info.
        Also ensures ell_eff and delta_ell are computed if not already set."""
        if not hasattr(self, 'ell_eff') or self.ell_eff is None:
            self._compute_ell_binning()

        if self.covariance_info is None:
            self.covariance_info = {}

        if "f_sky" not in self.covariance_info:
            val = input("f_sky not found in config. Enter f_sky: ")
            self.covariance_info["f_sky"] = float(val)

        for t in self.spectrum_info:
            if t not in self.covariance_info:
                self.covariance_info[t] = {}
            for (b0, b1) in self.spectrum_info[t]["bin_pairs"]:
                key = f"{b0}_{b1}"
                entry = self.covariance_info[t].get(key, {})
                if "noise" not in entry:
                    val = input(
                        f"noise for {t} bin pair ({b0}, {b1}) not found in config. Enter noise: "
                    )
                    entry["noise"] = float(val)
                    self.covariance_info[t][key] = entry

    def _lookup_spectrum(self, spec, za, zb, model_spectra):
        """Look up a spectrum value from model_spectra dict or observed data."""
        if model_spectra is not None and (spec, za, zb) in model_spectra:
            return model_spectra[(spec, za, zb)]
        return self.spectra["value"][
            (self.spectra["spectrum_type"] == spec.encode('utf-8'))
            & (self.spectra["zbin0"] == za)
            & (self.spectra["zbin1"] == zb)
        ]

    def gaussian_variance(self, si, sj, z00, z01, z10, z11, model_spectra=None):
        """Compute the diagonal Gaussian variance for a pair of spectrum blocks.

        Args:
            si: First spectrum type string.
            sj: Second spectrum type string.
            z00: First redshift bin of spectrum si.
            z01: Second redshift bin of spectrum si.
            z10: First redshift bin of spectrum sj.
            z11: Second redshift bin of spectrum sj.
            model_spectra: Optional dict of model spectra to use instead of observed.

        Returns:
            Array of variance values per ell bin.
        """
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
                c_w_n = self._lookup_spectrum(spec, za, zb, model_spectra) \
                        + float(self.covariance_info[spec][f"{za}_{zb}"]["noise"])
            elif covariance_field_types[spec][0] == covariance_field_types[spec][1]:
                c_w_n = self._lookup_spectrum(spec, zb, za, model_spectra) \
                        + float(self.covariance_info[spec][f"{zb}_{za}"]["noise"])
            elif f1 == "d":
                c_w_n = self._lookup_spectrum(spec, zb, za, model_spectra) \
                        + float(self.covariance_info[spec][f"{zb}_{za}"]["noise"])
            else:
                raise (
                    ValueError(f"No spectrum {spec} with zbin comination {za}, {zb}")
                )

            spec_w_n.append(c_w_n)

        var = (spec_w_n[0] * spec_w_n[1] + spec_w_n[2] * spec_w_n[3]) / (
            float(self.covariance_info["f_sky"]) * self.delta_ell * (2 * self.ell_eff)
        )

        return var

    def gaussian_covariance(self, model_spectra=None):
        """Compute the full Gaussian covariance matrix for all spectrum pairs.

        Args:
            model_spectra: Optional dict of model spectra for signal-dependent covariance.

        Returns:
            Structured numpy array of shape (n_dv, n_dv) with covariance values.
        """
        self._ensure_covariance_info()
        dt = np.dtype(
            [
                ("spectrum_type0", "S10"),
                ("spectrum_type1", "S10"),
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

                        var = self.gaussian_variance(si, sj, z00, z01, z10, z11,
                                                    model_spectra=model_spectra)
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

    def spectra_dict_from_vector(self, model_vector):
        """Convert a flat model vector into a dict keyed by (type, bin0, bin1).

        Args:
            model_vector: Flat array of model predictions in data vector ordering.

        Returns:
            Dict mapping (spectrum_type, bin0, bin1) tuples to per-bin arrays.
        """
        d = {}
        offset = 0
        for t in self.spectrum_types:
            n = self.spectrum_info[t]["n_dv_per_bin"]
            for (b0, b1) in self.spectrum_info[t]["bin_pairs"]:
                d[(t, b0, b1)] = np.asarray(model_vector[offset:offset + n])
                offset += n
        return d

    def plot_spectra_vs_model(self, model_pred=None):
        """Plot measured spectra, optionally compared to model predictions.

        One figure is created per spectrum type.  When *model_pred* is given,
        each figure has a grid of (data panel, residual panel) column pairs.
        When *model_pred* is ``None``, only the data panels are shown.
        Regions excluded by scale cuts are shaded.

        Parameters
        ----------
        model_pred : array_like, optional
            Full (unmasked) model prediction in the same ordering as
            ``self.spectra`` (i.e. ``apply_scale_mask=False``).  If ``None``,
            only the measurements are plotted.

        Returns
        -------
        dict[str, matplotlib.figure.Figure]
            Keys are spectrum type strings (e.g. ``'c_kk'``).
        """
        import matplotlib.pyplot as plt

        _YLABEL = {
            'c_kk':       r'$\ell\,C_\ell^{\gamma_E \gamma_E}$',
            'c_bb':       r'$\ell\,C_\ell^{\gamma_B \gamma_B}$',
            'c_dk':       r'$\ell\,C_\ell^{\delta_g \gamma_E}$',
            'c_dd':       r'$\ell\,C_\ell^{\delta_g \delta_g}$',
            'c_dcmbk':    r'$\ell\,C_\ell^{\delta_g \kappa_{\rm CMB}}$',
            'c_cmbkcmbk': r'$\ell\,C_\ell^{\kappa_{\rm CMB} \kappa_{\rm CMB}}$',
        }

        has_model = model_pred is not None
        if has_model:
            model_pred = np.asarray(model_pred)
        figs = {}

        for t in self.spectrum_types:
            bin_pairs = self.spectrum_info[t]["bin_pairs"]
            sep = self.spectrum_info[t]["separation"]

            unique_b0 = sorted(set(b0 for b0, b1 in bin_pairs))
            unique_b1 = sorted(set(b1 for b0, b1 in bin_pairs))
            n_cols = len(unique_b0)
            n_rows = len(unique_b1)
            occupied = {(unique_b1.index(b1), unique_b0.index(b0))
                        for b0, b1 in bin_pairs}

            if has_model:
                fig, axes = plt.subplots(
                    2 * n_rows, n_cols,
                    sharex=True, sharey='row',
                    gridspec_kw={'height_ratios': [3, 1] * n_rows},
                    squeeze=False,
                )
            else:
                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    sharex=True, sharey='row',
                    squeeze=False,
                )

            for b0, b1 in bin_pairs:
                col = unique_b0.index(b0)
                row = unique_b1.index(b1)
                if has_model:
                    ax_main = axes[2 * row, col]
                    ax_res  = axes[2 * row + 1, col]
                else:
                    ax_main = axes[row, col]

                idx = np.where(
                    (self.spectra["spectrum_type"] == t.encode('utf-8'))
                    & (self.spectra["zbin0"] == b0)
                    & (self.spectra["zbin1"] == b1)
                )[0]

                data  = self.spectra["value"][idx]

                if hasattr(self, 'cov') and self.cov is not None:
                    idxx, idxy = np.meshgrid(idx, idx, indexing='ij')
                    err = np.sqrt(np.diag(self.cov["value"][idxx, idxy]))
                else:
                    err = np.ones_like(data)

                ax_main.errorbar(sep, sep * data, sep * err,
                                 color='k', ls='', marker='o', ms=3, capsize=3)

                if has_model:
                    model = model_pred[idx]
                    ax_main.plot(sep, sep * model, color='k')
                    ax_res.plot(sep, (data - model) / err,
                                color='k', ls='', marker='o', ms=3)
                    ax_res.axhline(0, color='k', lw=0.8)

                # shade excluded scale ranges
                has_cuts = (self.scale_cuts is not None
                            and t in self.scale_cuts
                            and f'{b0}_{b1}' in self.scale_cuts[t])
                if has_cuts:
                    ell_min, ell_max = self.scale_cuts[t][f'{b0}_{b1}']
                    x_max_plot = ell_max * 1.5
                    shade_axes = [ax_main, ax_res] if has_model else [ax_main]
                    for ax in shade_axes:
                        ax.axvspan(sep[0] * 0.5, ell_min,
                                   color='k', alpha=0.15, linewidth=0)
                        ax.axvspan(ell_max, x_max_plot * 2,
                                   color='k', alpha=0.15, linewidth=0)
                    ax_main.set_xlim(sep[0] * 0.8, x_max_plot)

                ax_main.set_xscale('log')
                ax_main.set_yscale('log')
                ax_main.set_title(f'({b0}, {b1})', fontsize=9)

                if has_model:
                    ax_res.set_ylim(-4, 4)

                if row == n_rows - 1:
                    (ax_res if has_model else ax_main).set_xlabel(r'$\ell$')
                if col == 0:
                    ax_main.set_ylabel(_YLABEL.get(t, rf'$\ell\,C_\ell$ [{t}]'))
                    if has_model:
                        ax_res.set_ylabel(r'$(d-m)/\sigma$')

            # hide unused subplots
            for row in range(n_rows):
                for col in range(n_cols):
                    if (row, col) not in occupied:
                        if has_model:
                            axes[2 * row, col].axis('off')
                            axes[2 * row + 1, col].axis('off')
                        else:
                            axes[row, col].axis('off')

            fig.suptitle(_YLABEL.get(t, t).replace(r'\ell\,', ''), y=1.01)
            fig.set_size_inches(4 * n_cols, 5 * n_rows)
            fig.subplots_adjust(wspace=0.02, hspace=0.14)
            figs[t] = fig

        return figs

