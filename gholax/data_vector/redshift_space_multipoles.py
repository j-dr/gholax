from .data_vector import DataVector
from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.integrate import trapezoid
import jax.numpy as jnp
import numpy as np
import h5py as h5

covariance_field_types = {
    "p_gg_ell": ["drsd", "drsd"],
}

# BAO dilation parameters appended to the data vector as scalar "spectra"
# (one row per bin, ell=0). Their joint covariance with p_gg_ell must come
# from the data file; they are exempt from scale cuts.
ALPHA_TYPES = ("alpha_iso", "alpha_par", "alpha_perp")


datavector_requires = {
    "p_gg_ell": ["z_fid", "chiz_fid", "hz_fid"],
    "alpha_iso": ["rd_fid", "zeff_bao", "DV_fid_bao"],
    "alpha_par": ["rd_fid", "zeff_bao", "Hz_fid_bao"],
    "alpha_perp": ["rd_fid", "zeff_bao", "DM_fid_bao"],
}

field_types = {
    "p_gg_ell": ["d_zs", "d_zs"],
    "alpha_iso": ["bao", "bao"],
    "alpha_par": ["bao", "bao"],
    "alpha_perp": ["bao", "bao"],
}


class RedshiftSpaceMultipoles(DataVector):
    """Data vector for redshift-space power spectrum multipoles P_ell(k).

    Handles loading observed multipole spectra and covariance from HDF5,
    applying scale cuts per (bin, ell), computing Gaussian covariance
    matrices, and managing P_ell(k) window function matrices.
    """

    _field_types = field_types
    scale_cut_exempt_types = ALPHA_TYPES

    def __init__(
        self,
        data_vector_info_filename,
        spectrum_info,
        ells=(0, 2, 4),
        scale_cuts=None,
        covariance_info=None,
        dummy_cov=False,
        generate_data_vector=False,
        zmin=0,
        zmax=2.0,
        nz=125,
    ):
        """Initialize the redshift-space multipole data vector.

        Args:
            data_vector_info_filename: Path to the HDF5 data file.
            spectrum_info: Dict of spectrum type configs (bins, cross-correlations).
            ells: Tuple of multipole orders to include (default: (0, 2, 4)).
            scale_cuts: Optional dict of (k_min, k_max) per bin pair and ell.
            covariance_info: Optional dict with v_survey (survey volume) and
                noise for the Gaussian P_ell(k) covariance.
            dummy_cov: If True, skip loading the covariance matrix.
            generate_data_vector: If True, generate a synthetic data vector.
            zmin: Minimum redshift for n(z) interpolation grid.
            zmax: Maximum redshift for n(z) interpolation grid.
            nz: Number of redshift grid points.
        """
        self.data_vector_info_filename = data_vector_info_filename
        self.spectrum_info = spectrum_info
        # alphas are per-bin scalars; only auto "pairs" (bin, bin) make sense
        for t in ALPHA_TYPES:
            if t in self.spectrum_info:
                self.spectrum_info[t].setdefault("use_cross", False)
        self.scale_cuts = scale_cuts
        self.spectrum_types = list(self.spectrum_info.keys())
        self.covariance_info = covariance_info
        self.dummy_cov = dummy_cov
        self.generate_data_vector = generate_data_vector
        self.ells = ells
        self.z = jnp.linspace(zmin, zmax, nz)

    def _ells_for_scale_cuts(self):
        """Scale cuts and masks are per multipole order for P_ell(k)."""
        return self.ells

    def _spectrum_key(self, i, j, ell=None):
        """String key identifying a bin pair and multipole order."""
        return "{}_{}_{}".format(i, j, ell)

    def _first_block_meta(self, spectra, idx):
        """Restrict per-bin metadata to the highest multipole block and
        record ell_max."""
        stype = spectra["spectrum_type"][idx][0].decode()
        if stype in ALPHA_TYPES:
            return idx, {}
        ell_max = np.max(spectra["ell"][idx])
        idx = idx & (spectra["ell"] == ell_max)
        return idx, {"ell_max": ell_max}

    def _covariance_match_fields(self):
        """Covariance matching includes the multipole order."""
        return super()._covariance_match_fields() + [("ell1", "ell")]

    def _post_load_spectra(self, spectra):
        # assumes same bins for zbin0 and zbin1
        self.n_dbins = len(np.unique(spectra["zbin0"]))

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
                    ("ell", int),
                    ("separation", float),
                    ("value", float),
                ]
            )
            data = np.zeros(len(model), dtype=dt)
            data["spectrum_type"] = self.spectra["spectrum_type"]
            data["zbin0"] = self.spectra["zbin0"]
            data["zbin1"] = self.spectra["zbin1"]
            data["ell"] = self.spectra["ell"]
            data["separation"] = self.spectra["separation"]
            data["value"] = model

            f.create_dataset("spectra", data=data)
            f.create_dataset("covariance", data=self.cov)
            
            for k_i in self.data_vector_info.keys():
                if k_i in ["spectra", "covariance"]:
                    continue
                elif 'window' in k_i:
                    grp = f.create_group(k_i)
                    for k_j in self.data_vector_info[k_i].keys():
                        grp.create_dataset(k_j, data=self.data_vector_info[k_i][k_j][:])
                else:
                    ds = self.data_vector_info[k_i]
                    f.create_dataset(k_i, data=ds[()] if ds.shape == () else ds[:])

    def generate_data(self):
        """Generate a synthetic data vector with k-binning and window matrices.

        Returns:
            Structured numpy array of spectra with fields (spectrum_type,
            zbin0, zbin1, ell, separation, value).
        """
        required_spectra = []
        required_spectra = np.array(list(self.spectrum_info.keys()))
        n_ell = len(self.ells)

        bin_pairs = {}
        n_bins = 0
        for t in self.spectrum_info.keys():
            bin_pairs[t] = []
            for i in self.spectrum_info[t]["bins0"]:
                use_cross = self.spectrum_info[t].get("use_cross", True)
                if use_cross:
                    for j in self.spectrum_info[t]["bins1"]:
                        if j >= i:
                            bin_pairs[t].append((i, j))
                            n_bins += n_ell
                else:
                    j = i
                    bin_pairs[t].append((i, j))
                    n_bins += n_ell

        self.kth = np.linspace(1e-3, 0.6 + 1e-3, 600)
        self.delta_k = 0.01
        self.ko_edges = np.linspace(1e-3, 0.6 + 1e-3, 61)
        self.delta_k = self.ko_edges[1] - self.ko_edges[0]
        bpws = np.digitize(self.kth, self.ko_edges)
        bin_norm = np.bincount(bpws)[1:-1]
        self.ko_eff = np.bincount(bpws, weights=self.kth**3) / np.bincount(
            bpws, weights=self.kth**2
        )
        self.ko_eff = self.ko_eff[1:-1]

        window = np.zeros((n_ell, self.ko_eff.shape[0], n_ell, self.kth.shape[0]))

        for i, ki in enumerate(self.ko_eff):
            for li, ell in enumerate(self.ells):
                window[li, i, li, (bpws - 1) == i] = 1 / float(bin_norm[i])
        window = window.reshape(self.ko_eff.shape[0] * n_ell, self.kth.shape[0] * n_ell)

        dt = np.dtype(
            [
                ("spectrum_type", "S10"),
                ("zbin0", int),
                ("zbin1", int),
                ("ell", int),
                ("separation", float),
                ("value", float),
            ]
        )
        spectra = np.zeros(n_bins * n_ell * len(self.ko_eff), dtype=dt)
        self.W = {}
        counter = 0

        for bin_pair in bin_pairs[t]:
            for ell in self.ells:
                spectra[counter : counter + self.ko_eff.shape[0]]["spectrum_type"] = t
                spectra[counter : counter + self.ko_eff.shape[0]]["zbin0"] = bin_pair[0]
                spectra[counter : counter + self.ko_eff.shape[0]]["zbin1"] = bin_pair[1]
                spectra[counter : counter + self.ko_eff.shape[0]]["ell"] = ell
                spectra[counter : counter + self.ko_eff.shape[0]]["separation"] = (
                    self.ko_eff
                )
                counter += self.ko_eff.shape[0]

                self.W[f"{bin_pair[0]}_{bin_pair[1]}"] = window

        return spectra

    def load_requirements(self):
        """Load fiducial cosmology, redshift distributions, and window matrices from the data file."""
        requirements = []
        for t in self.spectrum_info:
            requirements = datavector_requires[t]
            for r in requirements:
                if r in ["z_fid", "chiz_fid", "hz_fid"]:
                    self.spectrum_info[t][r] = jnp.array(self.data_vector_info[r][:])
                elif t in ALPHA_TYPES:
                    if r not in self.data_vector_info:
                        raise ValueError(
                            f"Data file is missing dataset '{r}' required by "
                            f"BAO alpha spectrum type '{t}'"
                        )
                    ds = self.data_vector_info[r]
                    if ds.shape == ():
                        self.spectrum_info[t][r] = float(ds[()])
                    else:
                        self.spectrum_info[t][r] = jnp.array(ds[:])
#                elif "nz" in r:
#                    nz_ = self.data_vector_info[r][:]
#
#                    nbins = nz_.shape[1] - 1
#                    nz = jnp.zeros((nbins, len(self.z)))
#
#                    for i in range(nbins):
#                        idx = nz_[:, i + 1] > 0
#                        nz = nz.at[i, :].set(
#                            RegularGridInterpolator(
#                                [(nz_[idx, 0])],
#                                np.atleast_2d(nz_[idx, i + 1]).T,
#                                fill_value=0,
#                            )(self.z)[:, 0]
#                        )
#
#                    nz = nz / trapezoid(nz, x=self.z, axis=-1)[:, None]
#                    self.spectrum_info[t][r] = nz
#                    setattr(self, r, nz)

        if ("pkell_windows" in self.data_vector_info.keys()) & (
            not self.generate_data_vector
        ):
            window_matrix_files = self.data_vector_info["pkell_windows"]
            assert "kth_rsd" in self.data_vector_info.keys()
            assert "ko_rsd" in self.data_vector_info.keys()

            self.kth = self.data_vector_info["kth_rsd"][:]
            self.ko_eff = self.data_vector_info["ko_rsd"][:]

            self.W = {}
            # want per bin windows here
            for ij in list(window_matrix_files.keys()):
                self.W[ij] = window_matrix_files[ij]


    def _ensure_covariance_info(self):
        """Prompt interactively for any v_survey or noise terms missing from covariance_info."""
        alpha_present = [t for t in self.spectrum_info if t in ALPHA_TYPES]
        if alpha_present:
            raise ValueError(
                "The internal Gaussian covariance cannot be generated for BAO "
                f"alpha types {alpha_present}; provide a joint covariance in "
                "the data file instead."
            )

        if self.covariance_info is None:
            self.covariance_info = {}

        # P_ell(k) mode counting uses the survey volume, not a sky fraction.
        if "v_survey" not in self.covariance_info:
            val = input(
                "v_survey (survey volume, [Mpc/h]^3) not found in config. Enter v_survey: "
            )
            self.covariance_info["v_survey"] = float(val)

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

    def gaussian_variance(self, si, sj, z00, z01, z10, z11):
        """Compute the diagonal Gaussian variance for a pair of spectrum blocks.

        Args:
            si: First spectrum type string.
            sj: Second spectrum type string.
            z00: First redshift bin of spectrum si.
            z01: Second redshift bin of spectrum si.
            z10: First redshift bin of spectrum sj.
            z11: Second redshift bin of spectrum sj.

        Returns:
            Array of variance values per k bin.
        """
        # Resolve the four constituent spectra of the disconnected (Gaussian)
        # covariance term (P_ac P_bd + P_ad P_bc). Each entry pairs one field
        # of spectrum ``si`` with one field of spectrum ``sj``; the spectrum
        # that connects those two fields is found by inverting
        # ``covariance_field_types`` (which maps a spectrum type to its two
        # field labels). This replaces the "c_" + field-concatenation string
        # trick from the 2pt (C_ell) class, whose key format assumes the
        # spectrum type IS "c_<f0><f1>" and does not hold for P_ell(k) keys
        # such as "p_gg_ell".
        field_to_spectrum = {}
        for stype, fields in covariance_field_types.items():
            field_to_spectrum[(fields[0], fields[1])] = stype
            field_to_spectrum[(fields[1], fields[0])] = stype

        fi0, fi1 = covariance_field_types[si]
        fj0, fj1 = covariance_field_types[sj]
        c0 = field_to_spectrum[(fi0, fj0)]
        c1 = field_to_spectrum[(fi1, fj1)]
        c2 = field_to_spectrum[(fi0, fj1)]
        c3 = field_to_spectrum[(fi1, fj0)]

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
            # ``spectrum_type`` is stored as bytes (S10); encode the str key.
            spec_key = spec.encode("utf-8") if isinstance(spec, str) else spec
            if (za, zb) in self.spectrum_info[spec]["bin_pairs"]:
                c_w_n = self.spectra["value"][
                    (self.spectra["spectrum_type"] == spec_key)
                    & (self.spectra["zbin0"] == za)
                    & (self.spectra["zbin1"] == zb)
                ] + float(self.covariance_info[spec][f"{za}_{zb}"]["noise"])
            elif covariance_field_types[spec][0] == covariance_field_types[spec][1]:
                c_w_n = self.spectra["value"][
                    (self.spectra["spectrum_type"] == spec_key)
                    & (self.spectra["zbin0"] == zb)
                    & (self.spectra["zbin1"] == za)
                ] + float(self.covariance_info[spec][f"{zb}_{za}"]["noise"])
            elif f1 == "d":
                c_w_n = self.spectra["value"][
                    (self.spectra["spectrum_type"] == spec_key)
                    & (self.spectra["zbin0"] == zb)
                    & (self.spectra["zbin1"] == za)
                ] + float(self.covariance_info[spec][f"{zb}_{za}"]["noise"])
            else:
                raise (
                    ValueError(f"No spectrum {spec} with zbin comination {za}, {zb}")
                )

            spec_w_n.append(c_w_n)

        # k value for every entry of the block being filled. Taken from
        # spectrum ``si``'s own (z00, z01) rows so it aligns with the numerator
        # arrays (same k-grid across all constituent spectra).
        si_key = si.encode("utf-8") if isinstance(si, str) else si
        k_block = self.spectra["separation"][
            (self.spectra["spectrum_type"] == si_key)
            & (self.spectra["zbin0"] == z00)
            & (self.spectra["zbin1"] == z01)
        ]

        # Width of the k shells. Defined once in ``generate_data``; fall back to
        # the ``ko_eff`` spacing if a data vector was loaded without it.
        delta_k = getattr(self, "delta_k", None)
        if delta_k is None:
            delta_k = float(np.diff(np.asarray(self.ko_eff)).mean())

        # Gaussian (disconnected) covariance of P_ell(k) bandpowers from mode
        # counting in spherical k-shells:
        #   N_modes(k) = V_survey * k^2 * delta_k / (2 * pi^2)
        #   Var[P(k)]  = (P_ac P_bd + P_ad P_bc) / N_modes(k)
        # For a single-tracer auto-spectrum the numerator reduces to
        # 2 (P + 1/nbar)^2, i.e. the textbook 2 P^2 / N_modes result. This
        # mirrors the 2pt (C_ell) class, where the denominator is likewise the
        # number of independent modes in the bandpower
        # (N_modes = f_sky * (2 ell + 1) * delta_ell).
        n_modes = (
            float(self.covariance_info["v_survey"])
            * k_block**2
            * delta_k
            / (2.0 * np.pi**2)
        )

        var = (spec_w_n[0] * spec_w_n[1] + spec_w_n[2] * spec_w_n[3]) / n_modes

        return var

    def gaussian_covariance(self):
        """Compute the full Gaussian covariance matrix for all spectrum pairs.

        Returns:
            Structured numpy array of shape (n_dv, n_dv) with covariance values.
        """
        self._ensure_covariance_info()
        dt = np.dtype(
            [
                ("spectrum_type0", "S10"),
                ("spectrum_type1", "S10"),
                ("zbin00", np.int32),
                ("zbin01", np.int32),
                ("zbin10", np.int32),
                ("zbin11", np.int32),
                ("separation0", np.float64),
                ("separation1", np.float64),
                ("value", np.float64),
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
                        ]["separation1"] = self.spectrum_info[si]["separation"][None, :]

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

    def plot_spectra_vs_model(self, model_pred=None):
        """Plot measured spectra, optionally compared to model predictions.

        One figure is created per spectrum type.  When *model_pred* is given,
        each figure has a grid of (data panel, residual panel) pairs.
        When *model_pred* is ``None``, only the data panels are shown.
        Different multipole orders (ell = 0, 2, 4, …) are overlaid in each
        panel with distinct colours.  Regions excluded by scale cuts are shaded
        using the cuts for the first multipole order.

        Parameters
        ----------
        model_pred : array_like, optional
            Full (unmasked) model prediction in the same ordering as
            ``self.spectra`` (i.e. ``apply_scale_mask=False``).  If ``None``,
            only the measurements are plotted.

        Returns
        -------
        dict[str, matplotlib.figure.Figure]
            Keys are spectrum type strings (e.g. ``'p_gg_ell'``).
        """
        import matplotlib.pyplot as plt

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

            for pair_idx, (b0, b1) in enumerate(bin_pairs):
                col = unique_b0.index(b0)
                row = unique_b1.index(b1)
                if has_model:
                    ax_main = axes[2 * row, col]
                    ax_res  = axes[2 * row + 1, col]
                else:
                    ax_main = axes[row, col]

                for ell_idx, ell in enumerate(self.ells):
                    idx = np.where(
                        (self.spectra["spectrum_type"] == t.encode('utf-8'))
                        & (self.spectra["zbin0"] == b0)
                        & (self.spectra["zbin1"] == b1)
                        & (self.spectra["ell"] == ell)
                    )[0]

                    data  = self.spectra["value"][idx]

                    if hasattr(self, 'cov') and self.cov is not None:
                        idxx, idxy = np.meshgrid(idx, idx, indexing='ij')
                        err = np.sqrt(np.diag(self.cov["value"][idxx, idxy]))
                    else:
                        err = np.ones_like(data)

                    color = f'C{ell_idx}'
                    ax_main.errorbar(sep, sep * data, sep * err,
                                     color=color, ls='', marker='o', ms=3,
                                     capsize=3, label=rf'$\ell={ell}$')

                    if has_model:
                        model = model_pred[idx]
                        ax_main.plot(sep, sep * model, color=color)
                        ax_res.plot(sep, (data - model) / err,
                                    color=color, ls='', marker='o', ms=3)

                if has_model:
                    ax_res.axhline(0, color='k', lw=0.8)

                # shade excluded scale ranges using the first multipole's cuts
                first_ell = self.ells[0]
                cut_key = f'{b0}_{b1}_{first_ell}'
                has_cuts = (self.scale_cuts is not None
                            and t in self.scale_cuts
                            and cut_key in self.scale_cuts[t])
                if has_cuts:
                    k_min, k_max = self.scale_cuts[t][cut_key]
                    x_max_plot = k_max * 1.5
                    shade_axes = [ax_main, ax_res] if has_model else [ax_main]
                    for ax in shade_axes:
                        ax.axvspan(0, k_min,
                                   color='k', alpha=0.15, linewidth=0)
                        ax.axvspan(k_max, x_max_plot * 2,
                                   color='k', alpha=0.15, linewidth=0)
                    ax_main.set_xlim(0, x_max_plot)

                ax_main.set_title(f'({b0}, {b1})', fontsize=9)

                if has_model:
                    ax_res.set_ylim(-4, 4)

                if row == n_rows - 1:
                    (ax_res if has_model else ax_main).set_xlabel(
                        r'$k\,[h\,\mathrm{Mpc}^{-1}]$')
                if col == 0:
                    ax_main.set_ylabel(r'$k\,P_\ell(k)$')
                    if has_model:
                        ax_res.set_ylabel(r'$(d-m)/\sigma$')
                if pair_idx == 0:
                    ax_main.legend(fontsize=7)

            # hide unused subplots
            for row in range(n_rows):
                for col in range(n_cols):
                    if (row, col) not in occupied:
                        if has_model:
                            axes[2 * row, col].axis('off')
                            axes[2 * row + 1, col].axis('off')
                        else:
                            axes[row, col].axis('off')

            fig.set_size_inches(4 * n_cols, 5 * n_rows)
            fig.subplots_adjust(wspace=0.02, hspace=0.14)
            figs[t] = fig

        return figs
