import abc
import warnings

import h5py as h5
import jax.numpy as jnp
import numpy as np


class DataVector(metaclass=abc.ABCMeta):
    """Abstract base class for observed data vectors.

    Provides the shared implementation for loading spectra, applying bin
    selection and scale cuts, and matching/inverting the covariance matrix.
    Subclasses customize behavior through small hooks (`_spectrum_key`,
    `_ells_for_scale_cuts`, `_covariance_match_fields`, ...) and must
    implement loading of auxiliary requirements and saving model predictions.
    """

    # Subclasses bind their module-level field_types dict here so shared
    # methods can consult it without importing a specific module's copy.
    _field_types = None

    def __init__(self, config):
        pass

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _ells_for_scale_cuts(self):
        """Multipole orders to iterate over when building scale-cut masks.

        The base (angular C_ell) data vector has no multipole dimension, so a
        single no-op pass with ``None`` reproduces its behavior.
        """
        return [None]

    def _spectrum_key(self, i, j, ell=None):
        """String key identifying a bin pair (and optionally a multipole)."""
        return "{}_{}".format(i, j)

    def _row_selector(self, spectra, t, zb0, zb1, ell=None):
        """Boolean mask selecting rows of `spectra` for one spectrum block."""
        idx = (
            (spectra["spectrum_type"] == t.encode("utf-8"))
            & (spectra["zbin0"] == zb0)
            & (spectra["zbin1"] == zb1)
        )
        if ell is not None:
            idx &= spectra["ell"] == ell
        return idx

    def _first_block_meta(self, spectra, idx):
        """Optionally restrict the first (z0, z1) block used to derive
        per-bin metadata, returning (idx, extra_info_dict)."""
        return idx, {}

    def _covariance_match_fields(self):
        """(covariance_field, spectra_field) pairs used to match each data
        vector element to a row/column of the raw covariance matrix."""
        return [
            ("spectrum_type1", "spectrum_type"),
            ("zbin10", "zbin0"),
            ("zbin11", "zbin1"),
            ("separation1", "separation"),
        ]

    def _post_load_spectra(self, spectra):
        """Hook run on the raw spectra array before process_spectrum_info."""
        pass

    # ------------------------------------------------------------------
    # Shared implementation
    # ------------------------------------------------------------------
    def load_data(self):
        """Load observed data, requirements, scale cuts, and covariance."""
        self.load_data_vector()
        self.load_requirements()
        self.setup_scale_cuts()

        if not self.dummy_cov:
            self.load_covariance_matrix()
        else:
            self.cinv = None

    def load_data_vector(self):
        """Loads the required data."""

        self.data_vector_info = h5.File(self.data_vector_info_filename, "r")

        if not self.generate_data_vector:
            spectra = self.data_vector_info["spectra"][:]
        else:
            spectra = self.generate_data()

        self._post_load_spectra(spectra)
        self.process_spectrum_info(spectra)

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
            idx, extra_info = self._first_block_meta(spectra, idx)
            ndv_per_bin = np.sum(idx)
            sep_unmasked = spectra[idx]["separation"]

            self.spectrum_info[t].update(
                {
                    "n_dv_per_bin": ndv_per_bin,
                    "separation": sep_unmasked,
                    "n_bins0_tot": n_bins0_tot,
                    "n_bins1_tot": n_bins1_tot,
                    **extra_info,
                }
            )

        self.spectra = np.hstack(self.spectra)
        self.spectrum_values = jnp.array(self.spectra["value"])
        self.n_dv = len(self.spectra)

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
                            if self._field_types[t][0] == self._field_types[t][1]:
                                bins1 = self.spectrum_info[t]["bins1"][ii:]
                            else:
                                bins1 = self.spectrum_info[t]["bins1"][:]
                            bin_pairs = [(i, j) for j in bins1]
                        else:
                            bin_pairs = [(i, i)]

                        for (bi, bj) in bin_pairs:
                            for ell in self._ells_for_scale_cuts():
                                key = self._spectrum_key(bi, bj, ell)
                                try:
                                    sep_min, sep_max = scale_cut_dict[key]
                                except (KeyError, ValueError, TypeError):
                                    ell_msg = "" if ell is None else f", ell {ell}"
                                    raise ValueError(
                                        "Scale cuts not provided for {} bin pair {},{}{}".format(
                                            t, bi, bj, ell_msg
                                        )
                                    )
                                mask = (sep_min <= sep_unmasked) & (
                                    sep_unmasked <= sep_max
                                )
                                scale_cut_mask[key] = mask

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
                    for ell in self._ells_for_scale_cuts():
                        pair_id = (zb0, zb1) if ell is None else (zb0, zb1, ell)
                        if pair_id in zbin_counter[t0]:
                            continue

                        idxi = np.where(
                            self._row_selector(self.spectra, t0, zb0, zb1, ell)
                        )[0]

                        try:
                            start_idx = np.min(idxi)
                        except ValueError:
                            continue

                        # mask scales
                        if self.spectrum_info[t0]["scale_cut_masks"] is not None:
                            mask_i = np.where(
                                self.spectrum_info[t0]["scale_cut_masks"][
                                    self._spectrum_key(zb0, zb1, ell)
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
        if len(cov_raw.shape) < 2:
            cov_raw = cov_raw.reshape(
                int(cov_raw.shape[0] ** 0.5), int(cov_raw.shape[0] ** 0.5)
            )

        cov_slice = cov_raw[0, :]
        match_fields = self._covariance_match_fields()

        # index the covariance rows once instead of an O(n_dv^2) scan
        cov_key_arrays = [cov_slice[cf] for cf, _ in match_fields]
        row_index = {}
        for j in range(len(cov_slice)):
            key = tuple(arr[j].item() for arr in cov_key_arrays)
            row_index.setdefault(key, []).append(j)

        idxi = np.zeros(len(self.spectra), dtype=int)
        for i in range(self.n_dv):
            key = tuple(self.spectra[i][sf].item() for _, sf in match_fields)
            idx = row_index.get(key, [])
            if len(idx) > 1:
                raise (ValueError)
            elif len(idx) < 1:
                raise (
                    ValueError(
                        "No matching cov entry for {}".format(
                            ", ".join(str(k) for k in key)
                        )
                    )
                )

            idxi[i] = idx[0]

        covidx, covidy = np.meshgrid(idxi, idxi, indexing="ij")
        self.cov = cov_raw[covidx, covidy]
        assert np.allclose(self.cov["value"], self.cov["value"].T, 1e-12)

        cov_scale_mask_i, cov_scale_mask_j = np.meshgrid(
            self.scale_mask, self.scale_mask, indexing="ij"
        )
        self.cinv = jnp.linalg.inv(
            self.cov["value"][cov_scale_mask_i, cov_scale_mask_j].reshape(
                self.n_dv_masked, self.n_dv_masked
            )
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load_requirements(self):
        """Load auxiliary data required by the data vector (e.g. n(z), windows)."""
        pass

    @abc.abstractmethod
    def save_data_vector(self, filename, model):
        """Saves the model to a file.

        Args:
            filename (str): The name of the file where the model will be saved.
            model (array-like): The model data to be saved.
        """
        pass
