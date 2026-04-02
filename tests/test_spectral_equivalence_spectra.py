"""
Tests for the spectral equivalence method applied to P_ij basis spectra.

For each w0wa cosmology, compares:
  - "exact": CLEFT / spinosaurus run on CLASS linear P(k,z) with the true w0wa cosmology
  - "spectral equiv": CLEFT / spinosaurus run on CLASS linear P(k,z) with the
    equivalent wCDM cosmology (w_equiv(z), As_equiv(z)) found by SpectralEquivalence

These tests validate that the spectral equivalence mapping produces P_ij basis
spectra that match the true w0wa spectra at each redshift slice.

Pass --plot-diagnostics to save residual plots to tests/plots/.

Requires: classy, velocileptors (spinosaurus.cleft_fftw.CLEFT),
          spinosaurus.density_shape_correlators_fftw.DensityShapeCorrelators,
          spinosaurus.shape_shape_correlators_fftw.ShapeShapeCorrelators
"""
import gholax.util.likelihood_module  # noqa: F401

import os
import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import W0WA_COSMO_PARAMS, requires_classy

# ---------------------------------------------------------------------------
# Skip markers for optional dependencies
# ---------------------------------------------------------------------------

try:
    from spinosaurus.cleft_fftw import CLEFT as _CLEFT  # noqa: F401
    cleft_available = True
except ImportError:
    cleft_available = False

try:
    from spinosaurus.density_shape_correlators_fftw import (
        DensityShapeCorrelators as _DSC,  # noqa: F401
    )
    density_shape_available = True
except ImportError:
    density_shape_available = False

try:
    from spinosaurus.shape_shape_correlators_fftw import (
        ShapeShapeCorrelators as _SSC,  # noqa: F401
    )
    shape_shape_available = True
except ImportError:
    shape_shape_available = False


try:
    from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD as _LPT_RSD  # noqa: F401
    lpt_rsd_available = True
except ImportError:
    lpt_rsd_available = False

try:
    from CEmulator.Emulator import Pkmm_CEmulator as _Pkmm  # noqa: F401
    csstemu_available = True
except ImportError:
    csstemu_available = False

requires_cleft = pytest.mark.skipif(
    not cleft_available, reason="spinosaurus CLEFT not installed"
)
requires_density_shape = pytest.mark.skipif(
    not density_shape_available, reason="spinosaurus DensityShapeCorrelators not installed"
)
requires_shape_shape = pytest.mark.skipif(
    not shape_shape_available, reason="spinosaurus ShapeShapeCorrelators not installed"
)
requires_lpt_rsd = pytest.mark.skipif(
    not lpt_rsd_available, reason="velocileptors LPT_RSD not installed"
)
requires_csstemu = pytest.mark.skipif(
    not csstemu_available, reason="csstemu not installed"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIGMA8_EMU_FILE = "aemulus_nu_tier1_cosmo_only_heft_sigma8z_emu"

_KMIN = 1e-3
_KMAX = 1.0
_NK = 200
_K = np.logspace(np.log10(_KMIN), np.log10(_KMAX), _NK)
_LOGK = np.linspace(np.log10(_KMIN), np.log10(_KMAX), _NK)
_KIR = 0.2

# Redshifts at which we compare spectra
_Z_TEST = np.array([0.0, 0.5, 1.0, 2.0])

# k range for the linear P(k) from CLASS
_K_LIN_MIN = 1e-4
_K_LIN_MAX = 20.0
_NK_LIN = 500
_K_LIN = np.logspace(np.log10(_K_LIN_MIN), np.log10(_K_LIN_MAX), _NK_LIN)

# Relative tolerance for spectra comparison (basis spectra, not final observables)
_RTOL_CLEFT = 0.01       # 1% for CLEFT P_ij
_RTOL_DENSITY_SHAPE = 0.01  # 1% for density-shape IA
_RTOL_SHAPE_SHAPE = 0.01    # 1% for shape-shape IA
_RTOL_RSD = 0.05            # 5% for LPT_RSD P_ij (growth rate f(z) adds mismatch)

# HEFT P_11 emulator config
_HEFT_PIJ_EMU_CONFIG = "emu_config_dst_irresum.yaml"

# Tolerance for HEFT P_11 vs csstemu (different emulators, so larger tolerance)
_RTOL_HEFT_VS_CSSTEMU = 0.05  # 5%


_PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ensure_plot_dir():
    os.makedirs(_PLOT_DIR, exist_ok=True)


def _cosmo_label(params):
    return f"w0={params['w']:.1f}_wa={params.get('wa', 0.0):.1f}"


def _plot_cleft_residuals(k, spec_exact, spec_equiv, params, z_values,
                          spec_indices, spec_names=None):
    """Plot relative residuals (equiv - exact) / exact vs k for CLEFT spectra."""
    import matplotlib.pyplot as plt

    if spec_names is None:
        spec_names = {i: f"spec {i}" for i in spec_indices}

    label = _cosmo_label(params)
    nz = len(z_values)
    ns = len(spec_indices)

    fig, axes = plt.subplots(ns, nz, figsize=(5 * nz, 3.5 * ns),
                             squeeze=False, sharex=True)
    fig.suptitle(f"CLEFT P_ij residuals: spectral equiv vs exact w0wa\n{label}",
                 fontsize=13)

    for col, z_i in enumerate(z_values):
        for row, s_idx in enumerate(spec_indices):
            ax = axes[row, col]
            exact = spec_exact[z_i][s_idx]
            equiv = spec_equiv[z_i][s_idx]

            mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
            rel_err = np.full_like(exact, np.nan)
            rel_err[mask] = (equiv[mask] - exact[mask]) / exact[mask]

            ax.semilogx(k, rel_err, "C0-", lw=0.8)
            ax.axhline(0, color="k", ls="--", lw=0.5)
            ax.set_ylabel(f"{spec_names.get(s_idx, f's{s_idx}')}")
            if row == 0:
                ax.set_title(f"z = {z_i}")
            if row == ns - 1:
                ax.set_xlabel(r"$k$ [$h$/Mpc]")
            ax.set_ylim(-0.05, 0.05)

    fig.tight_layout()
    fname = os.path.join(_PLOT_DIR, f"cleft_residuals_{label}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def _plot_ia_residuals(k, spec_exact, spec_equiv, params, z_values,
                       spec_indices, title_prefix, filename_prefix,
                       spec_names=None, m_values=None):
    """Plot relative residuals vs k for IA spectra (density-shape or shape-shape).

    For shape-shape, m_values=[0,1,2] adds an extra panel dimension.
    """
    import matplotlib.pyplot as plt

    if spec_names is None:
        spec_names = {i: f"spec {i}" for i in spec_indices}

    label = _cosmo_label(params)
    nz = len(z_values)
    ns = len(spec_indices)

    if m_values is not None:
        nm = len(m_values)
        fig, axes = plt.subplots(ns * nm, nz, figsize=(5 * nz, 3.0 * ns * nm),
                                 squeeze=False, sharex=True)
        fig.suptitle(f"{title_prefix} residuals: spectral equiv vs exact w0wa\n{label}",
                     fontsize=13)
        for col, z_i in enumerate(z_values):
            for mi, m in enumerate(m_values):
                for si, s_idx in enumerate(spec_indices):
                    row = mi * ns + si
                    ax = axes[row, col]
                    exact = spec_exact[z_i][m, s_idx]
                    equiv = spec_equiv[z_i][m, s_idx]

                    mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
                    rel_err = np.full_like(exact, np.nan)
                    rel_err[mask] = (equiv[mask] - exact[mask]) / exact[mask]

                    ax.semilogx(k, rel_err, "C0-", lw=0.8)
                    ax.axhline(0, color="k", ls="--", lw=0.5)
                    ax.set_ylabel(f"m={m} {spec_names.get(s_idx, f's{s_idx}')}")
                    if row == 0:
                        ax.set_title(f"z = {z_i}")
                    if row == ns * nm - 1:
                        ax.set_xlabel(r"$k$ [$h$/Mpc]")
                    ax.set_ylim(-0.05, 0.05)
    else:
        fig, axes = plt.subplots(ns, nz, figsize=(5 * nz, 3.5 * ns),
                                 squeeze=False, sharex=True)
        fig.suptitle(f"{title_prefix} residuals: spectral equiv vs exact w0wa\n{label}",
                     fontsize=13)
        for col, z_i in enumerate(z_values):
            for row, s_idx in enumerate(spec_indices):
                ax = axes[row, col]
                exact = spec_exact[z_i][s_idx]
                equiv = spec_equiv[z_i][s_idx]

                mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
                rel_err = np.full_like(exact, np.nan)
                rel_err[mask] = (equiv[mask] - exact[mask]) / exact[mask]

                ax.semilogx(k, rel_err, "C0-", lw=0.8)
                ax.axhline(0, color="k", ls="--", lw=0.5)
                ax.set_ylabel(f"{spec_names.get(s_idx, f's{s_idx}')}")
                if row == 0:
                    ax.set_title(f"z = {z_i}")
                if row == ns - 1:
                    ax.set_xlabel(r"$k$ [$h$/Mpc]")
                ax.set_ylim(-0.05, 0.05)

    fig.tight_layout()
    fname = os.path.join(_PLOT_DIR, f"{filename_prefix}_{label}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_pk(params, z_arr):
    """Run CLASS and return (k_lin, Pm_lin[nz, nk], Pcb_lin[nz, nk]).

    All in h/Mpc units. Supports w0wa via wa_fld.
    """
    from classy import Class
    boltz = Class()
    z_str = ",".join([f"{z:.4f}" for z in z_arr])
    boltz.set({
        "output": "mPk",
        "P_k_max_h/Mpc": _K_LIN_MAX * 1.1,
        "z_pk": z_str,
        "A_s": params["As"] * 1e-9,
        "n_s": params["ns"],
        "h": params["H0"] / 100,
        "omega_b": params["ombh2"],
        "omega_cdm": params["omch2"],
        "N_ur": 0.0,
        "N_ncdm": 1,
        "deg_ncdm": 3,
        "m_ncdm": params["mnu"] / 3,
        "Omega_Lambda": 0.0,
        "w0_fld": params["w"],
        "wa_fld": params.get("wa", 0.0),
        "tau_reio": 0.0568,
    })
    boltz.compute()

    h = params["H0"] / 100
    nz = len(z_arr)
    Pm = np.zeros((nz, _NK_LIN))
    Pcb = np.zeros((nz, _NK_LIN))
    for i, z in enumerate(z_arr):
        for j, k in enumerate(_K_LIN):
            Pm[i, j] = boltz.pk(k * h, float(z)) * h**3
            Pcb[i, j] = boltz.pk_cb(k * h, float(z)) * h**3

    boltz.struct_cleanup()
    boltz.empty()
    return _K_LIN, Pm, Pcb


def _run_spectral_equivalence(params):
    """Run the SpectralEquivalence module and return (w_equiv_z, As_equiv_z, z_equiv)."""
    from gholax.theory.spectral_equivalence import SpectralEquivalence

    se = SpectralEquivalence(
        z=_Z_TEST,
        z_lss=1089.0,
        n_newton=15,
        n_int_chi=4096,
        sigma8_emulator_file_name=_SIGMA8_EMU_FILE,
    )
    state = {}
    state = se.compute(state, params)
    return (
        np.array(state["w_equiv_z"]),
        np.array(state["As_equiv_z"]),
        np.array(state["z_equiv"]),
    )


def _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv):
    """Build a wCDM params dict using the interpolated equivalent w and As at z_i."""
    w_eq = float(np.interp(z_i, z_equiv, w_equiv_z))
    As_eq = float(np.interp(z_i, z_equiv, As_equiv_z))
    p = dict(params)
    p["w"] = w_eq
    p["wa"] = 0.0
    p["As"] = As_eq
    return p


def _compute_cleft_spectra(k_lin, pk_cb, pk_m, kIR=_KIR):
    """Run CLEFT on a single redshift slice and return (nspec, nk) array on _K grid."""
    from spinosaurus.cleft_fftw import CLEFT
    from scipy.interpolate import interp1d as scipy_interp1d

    pk_cb_m = np.sqrt(pk_cb * pk_m)
    nspec = 19
    spec = np.zeros((nspec, _NK))

    s_m_map = {1: 0, 3: 1, 6: 3, 10: 6, 15: 10}
    s_cb_map = {
        2: 0, 4: 1, 5: 2, 7: 3, 8: 4, 9: 5,
        11: 6, 12: 7, 13: 8, 14: 9, 16: 10, 17: 11, 18: 12,
    }

    cleft_mm = CLEFT(k_lin, pk_m, kIR=kIR)
    cleft_mm.make_ptable(kmin=_KMIN, kmax=_KMAX, nk=300)
    pk_mm = scipy_interp1d(
        cleft_mm.pktable.T[0], cleft_mm.pktable.T[1:],
        kind="cubic", axis=1, fill_value="extrapolate",
    )(_K)

    cleft_cc = CLEFT(k_lin, pk_cb, kIR=kIR)
    cleft_cc.make_ptable(kmin=_KMIN, kmax=_KMAX, nk=300)
    pk_cc = scipy_interp1d(
        cleft_cc.pktable.T[0], cleft_cc.pktable.T[1:],
        kind="cubic", axis=1, fill_value="extrapolate",
    )(_K)

    cleft_cm = CLEFT(k_lin, pk_cb_m, kIR=kIR)
    cleft_cm.make_ptable(kmin=_KMIN, kmax=_KMAX, nk=300)
    pk_cm = scipy_interp1d(
        cleft_cm.pktable.T[0], cleft_cm.pktable.T[1:],
        kind="cubic", axis=1, fill_value="extrapolate",
    )(_K)

    for s in range(nspec):
        if s == 0:
            spec[s] = pk_mm[0]
        elif s in [1, 3, 6, 10, 15]:
            spec[s] = pk_cm[s_m_map[s]]
        else:
            spec[s] = pk_cc[s_cb_map[s]]

    return spec


def _compute_density_shape_spectra(k_lin, pk_cb):
    """Run spinosaurus DensityShapeCorrelators on one z-slice, return (nspec, nk)."""
    from spinosaurus.density_shape_correlators_fftw import DensityShapeCorrelators
    from scipy.interpolate import interp1d as scipy_interp1d

    nspec = 21
    ptmod = DensityShapeCorrelators(k_lin, pk_cb, kIR=_KIR)
    ptmod.make_gdtable(kmin=_KMIN, kmax=_KMAX, nk=_NK)
    pk_ij_raw = ptmod.pktable_gd.T[1:, :].T  # (nk_emu, nspec)
    k_emu = ptmod.pktable_gd.T[0, :]

    spec = np.zeros((nspec, _NK))
    for j in range(nspec):
        spec[j] = scipy_interp1d(
            np.log10(k_emu), pk_ij_raw[:, j],
            kind="cubic", fill_value="extrapolate",
        )(_LOGK)

    return spec


def _compute_shape_shape_spectra(k_lin, pk_cb):
    """Run spinosaurus ShapeShapeCorrelators on one z-slice, return (3, nspec, nk)."""
    from spinosaurus.shape_shape_correlators_fftw import ShapeShapeCorrelators
    from scipy.interpolate import interp1d as scipy_interp1d

    nspec = 13
    ptmod = ShapeShapeCorrelators(k_lin, pk_cb, kIR=_KIR)
    spec = np.zeros((3, nspec, _NK))

    for m in [0, 1, 2]:
        ptmod.make_ggtable(m, kmin=_KMIN, kmax=_KMAX, nk=_NK)
        pk_ij_raw = ptmod.pktables_gg[m].T[1:, :].T
        k_emu = ptmod.pktables_gg[m].T[0, :]
        for j in range(nspec):
            spec[m, j] = scipy_interp1d(
                np.log10(k_emu), pk_ij_raw[:, j],
                kind="cubic", fill_value="extrapolate",
            )(_LOGK)

    return spec


def _class_pk_rsd(params, z_arr):
    """Run CLASS and return (k_lin, Pcb_lin[nz, nk], f_z[nz]).

    Like _class_pk but also returns the scale-independent growth rate f(z)
    and only returns Pcb (not Pm), since LPT_RSD only needs cb spectra.
    """
    from classy import Class
    boltz = Class()
    z_str = ",".join([f"{z:.4f}" for z in z_arr])
    boltz.set({
        "output": "mPk",
        "P_k_max_h/Mpc": _K_LIN_MAX * 1.1,
        "z_pk": z_str,
        "A_s": params["As"] * 1e-9,
        "n_s": params["ns"],
        "h": params["H0"] / 100,
        "omega_b": params["ombh2"],
        "omega_cdm": params["omch2"],
        "N_ur": 0.0,
        "N_ncdm": 1,
        "deg_ncdm": 3,
        "m_ncdm": params["mnu"] / 3,
        "Omega_Lambda": 0.0,
        "w0_fld": params["w"],
        "wa_fld": params.get("wa", 0.0),
        "tau_reio": 0.0568,
    })
    boltz.compute()

    h = params["H0"] / 100
    nz = len(z_arr)
    Pcb = np.zeros((nz, _NK_LIN))
    f_z = np.zeros(nz)
    for i, z in enumerate(z_arr):
        f_z[i] = boltz.scale_independent_growth_factor_f(float(z))
        for j, k in enumerate(_K_LIN):
            Pcb[i, j] = boltz.pk_cb(k * h, float(z)) * h**3

    boltz.struct_cleanup()
    boltz.empty()
    return _K_LIN, Pcb, f_z


def _compute_lpt_rsd_spectra(k_lin, pk_cb, f_z, kIR=_KIR):
    """Run LPT_RSD on a single redshift slice and return (3, 19, nk) array.

    Output shape is (n_ell, n_spec, nk) for ell=0,2,4.
    No AP correction (apar=aperp=1).
    """
    from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

    model = LPT_RSD(
        k_lin, pk_cb, kIR=kIR,
        use_Pzel=False, cutoff=10,
        extrap_min=-4, extrap_max=3,
        N=2000, threads=1, jn=5,
    )
    model.make_pltable(
        np.float64(f_z), kv=_K, apar=1, aperp=1,
    )

    spec = np.zeros((3, 19, _NK))
    spec[0] = model.p0ktable.T
    spec[1] = model.p2ktable.T
    spec[2] = model.p4ktable.T
    return spec


def _plot_rsd_residuals(k, spec_exact, spec_equiv, params, z_values,
                        spec_indices, spec_names=None):
    """Plot relative residuals vs k for LPT_RSD spectra, with panels for ell=0,2,4."""
    import matplotlib.pyplot as plt

    if spec_names is None:
        spec_names = {i: f"spec {i}" for i in spec_indices}

    ell_labels = [r"$\ell=0$", r"$\ell=2$", r"$\ell=4$"]
    label = _cosmo_label(params)
    nz = len(z_values)
    ns = len(spec_indices)
    n_ell = 3

    fig, axes = plt.subplots(ns * n_ell, nz, figsize=(5 * nz, 3.0 * ns * n_ell),
                             squeeze=False, sharex=True)
    fig.suptitle(f"LPT_RSD P_ij residuals: spectral equiv vs exact w0wa\n{label}",
                 fontsize=13)

    for col, z_i in enumerate(z_values):
        for li in range(n_ell):
            for si, s_idx in enumerate(spec_indices):
                row = li * ns + si
                ax = axes[row, col]
                exact = spec_exact[z_i][li, s_idx]
                equiv = spec_equiv[z_i][li, s_idx]

                mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
                rel_err = np.full_like(exact, np.nan)
                rel_err[mask] = (equiv[mask] - exact[mask]) / exact[mask]

                ax.semilogx(k, rel_err, "C0-", lw=0.8)
                ax.axhline(0, color="k", ls="--", lw=0.5)
                ax.set_ylabel(f"{ell_labels[li]} {spec_names.get(s_idx, f's{s_idx}')}")
                if row == 0:
                    ax.set_title(f"z = {z_i}")
                if row == ns * n_ell - 1:
                    ax.set_xlabel(r"$k$ [$h$/Mpc]")
                ax.set_ylim(-0.05, 0.05)

    fig.tight_layout()
    fname = os.path.join(_PLOT_DIR, f"lpt_rsd_residuals_{label}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Select test cosmologies with nonzero wa
# ---------------------------------------------------------------------------

_W0WA_NONZERO_IDS = ["wa_neg", "wa_pos"]
_W0WA_NONZERO = [W0WA_COSMO_PARAMS[k] for k in _W0WA_NONZERO_IDS]

# DESI DR2 BAO x CMB best fit (Table V of arXiv:2503.14738)
_DESI_DR2_PARAMS = W0WA_COSMO_PARAMS["desi_dr2_bao_cmb"]


# ---------------------------------------------------------------------------
# Test: CLEFT P_ij (RealSpaceBiasedTracerSpectra)
# ---------------------------------------------------------------------------

@requires_classy
@requires_cleft
@pytest.mark.parametrize("params", _W0WA_NONZERO, ids=_W0WA_NONZERO_IDS)
def test_cleft_pij_spectral_equiv_vs_exact(params, plot_diagnostics):
    """CLEFT P_ij from equivalent wCDM matches true w0wa CLEFT to within tolerance.

    For each test redshift, runs CLEFT on CLASS P_lin from the true w0wa cosmology
    and compares against CLEFT on CLASS P_lin from the equivalent wCDM cosmology
    determined by SpectralEquivalence.
    """
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    _CLEFT_SPEC_INDICES = [0, 1, 2, 3, 5]
    _CLEFT_SPEC_NAMES = {0: r"$P_{mm}$", 1: r"$P_{cb,m}^{(0)}$",
                         2: r"$P_{cb,cb}^{(0)}$", 3: r"$P_{cb,m}^{(1)}$",
                         5: r"$P_{cb,cb}^{(2)}$"}

    all_spec_exact = {}
    all_spec_equiv = {}

    for z_i in _Z_TEST:
        # Exact: CLASS with true w0wa
        k_lin, Pm_exact, Pcb_exact = _class_pk(params, [z_i])
        spec_exact = _compute_cleft_spectra(k_lin, Pcb_exact[0], Pm_exact[0])

        # Spectral equiv: CLASS with equivalent wCDM at this z
        equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
        k_lin_eq, Pm_eq, Pcb_eq = _class_pk(equiv_params, [z_i])
        spec_equiv = _compute_cleft_spectra(k_lin_eq, Pcb_eq[0], Pm_eq[0])

        all_spec_exact[z_i] = spec_exact
        all_spec_equiv[z_i] = spec_equiv

        # Compare: focus on the dominant spectra (skip near-zero components)
        for s_idx in _CLEFT_SPEC_INDICES:
            exact = spec_exact[s_idx]
            equiv = spec_equiv[s_idx]

            # Only compare where signal is nontrivial
            mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
            if not mask.any():
                continue

            rel_err = np.abs(equiv[mask] - exact[mask]) / np.abs(exact[mask])
            median_err = np.median(rel_err)
            assert median_err < _RTOL_CLEFT, (
                f"CLEFT spec {s_idx} at z={z_i}: median rel error {median_err:.4f} "
                f"> {_RTOL_CLEFT} (max={rel_err.max():.4f})"
            )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_cleft_residuals(
            _K, all_spec_exact, all_spec_equiv, params, list(_Z_TEST),
            _CLEFT_SPEC_INDICES, _CLEFT_SPEC_NAMES,
        )
        print(f"  Saved CLEFT diagnostic plot: {fname}")


# ---------------------------------------------------------------------------
# Test: DensityShapeIA P_ij
# ---------------------------------------------------------------------------

@requires_classy
@requires_density_shape
@pytest.mark.parametrize("params", _W0WA_NONZERO, ids=_W0WA_NONZERO_IDS)
def test_density_shape_ia_spectral_equiv_vs_exact(params, plot_diagnostics):
    """DensityShape IA P_ij from equivalent wCDM matches true w0wa to within tolerance.

    Compares spinosaurus DensityShapeCorrelators output between the true w0wa
    and the spectral-equivalence wCDM at each test redshift.
    """
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    _DS_SPEC_INDICES = [0, 1, 4]
    _DS_SPEC_NAMES = {0: r"$P_{g\gamma}^{(0)}$", 1: r"$P_{g\gamma}^{(1)}$",
                      4: r"$P_{g\gamma}^{(4)}$"}

    all_spec_exact = {}
    all_spec_equiv = {}

    for z_i in _Z_TEST:
        # Exact: CLASS with true w0wa
        k_lin, _, Pcb_exact = _class_pk(params, [z_i])
        spec_exact = _compute_density_shape_spectra(k_lin, Pcb_exact[0])

        # Spectral equiv
        equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
        k_lin_eq, _, Pcb_eq = _class_pk(equiv_params, [z_i])
        spec_equiv = _compute_density_shape_spectra(k_lin_eq, Pcb_eq[0])

        all_spec_exact[z_i] = spec_exact
        all_spec_equiv[z_i] = spec_equiv

        # Compare the dominant spectra
        for s_idx in _DS_SPEC_INDICES:
            exact = spec_exact[s_idx]
            equiv = spec_equiv[s_idx]

            mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
            if not mask.any():
                continue

            rel_err = np.abs(equiv[mask] - exact[mask]) / np.abs(exact[mask])
            median_err = np.median(rel_err)
            assert median_err < _RTOL_DENSITY_SHAPE, (
                f"DensityShape spec {s_idx} at z={z_i}: median rel error "
                f"{median_err:.4f} > {_RTOL_DENSITY_SHAPE} (max={rel_err.max():.4f})"
            )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_ia_residuals(
            _K, all_spec_exact, all_spec_equiv, params, list(_Z_TEST),
            _DS_SPEC_INDICES, "DensityShape IA", "density_shape_residuals",
            _DS_SPEC_NAMES,
        )
        print(f"  Saved DensityShape diagnostic plot: {fname}")


# ---------------------------------------------------------------------------
# Test: ShapeShapeIA P_mij
# ---------------------------------------------------------------------------

@requires_classy
@requires_shape_shape
@pytest.mark.parametrize("params", _W0WA_NONZERO, ids=_W0WA_NONZERO_IDS)
def test_shape_shape_ia_spectral_equiv_vs_exact(params, plot_diagnostics):
    """ShapeShape IA P_mij from equivalent wCDM matches true w0wa to within tolerance.

    Compares spinosaurus ShapeShapeCorrelators output (all three m=0,1,2 modes)
    between the true w0wa and the spectral-equivalence wCDM at each test redshift.
    """
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    _SS_SPEC_INDICES = [0, 1]
    _SS_SPEC_NAMES = {0: r"$P_{\gamma\gamma}^{(0)}$",
                      1: r"$P_{\gamma\gamma}^{(1)}$"}
    _SS_M_VALUES = [0, 1, 2]

    all_spec_exact = {}
    all_spec_equiv = {}

    for z_i in _Z_TEST:
        # Exact: CLASS with true w0wa
        k_lin, _, Pcb_exact = _class_pk(params, [z_i])
        spec_exact = _compute_shape_shape_spectra(k_lin, Pcb_exact[0])

        # Spectral equiv
        equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
        k_lin_eq, _, Pcb_eq = _class_pk(equiv_params, [z_i])
        spec_equiv = _compute_shape_shape_spectra(k_lin_eq, Pcb_eq[0])

        all_spec_exact[z_i] = spec_exact
        all_spec_equiv[z_i] = spec_equiv

        # Compare: m=0,1,2, dominant spectra
        for m in _SS_M_VALUES:
            for s_idx in _SS_SPEC_INDICES:
                exact = spec_exact[m, s_idx]
                equiv = spec_equiv[m, s_idx]

                mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
                if not mask.any():
                    continue

                rel_err = np.abs(equiv[mask] - exact[mask]) / np.abs(exact[mask])
                median_err = np.median(rel_err)
                assert median_err < _RTOL_SHAPE_SHAPE, (
                    f"ShapeShape m={m} spec {s_idx} at z={z_i}: median rel error "
                    f"{median_err:.4f} > {_RTOL_SHAPE_SHAPE} (max={rel_err.max():.4f})"
                )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_ia_residuals(
            _K, all_spec_exact, all_spec_equiv, params, list(_Z_TEST),
            _SS_SPEC_INDICES, "ShapeShape IA", "shape_shape_residuals",
            _SS_SPEC_NAMES, m_values=_SS_M_VALUES,
        )
        print(f"  Saved ShapeShape diagnostic plot: {fname}")


# ---------------------------------------------------------------------------
# Regression test: wa=0 gives identical spectra (no mapping should occur)
# ---------------------------------------------------------------------------

@requires_classy
@requires_cleft
def test_cleft_wa_zero_identity():
    """With wa=0, spectral equivalence should produce w_equiv ~ w0 and As_equiv ~ As,
    so CLEFT spectra from the equivalent wCDM should match the original exactly."""
    params = W0WA_COSMO_PARAMS["wa_zero"]
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    # w_equiv should be very close to w0
    np.testing.assert_allclose(w_equiv_z, params["w"], atol=1e-4,
                               err_msg="w_equiv should equal w0 when wa=0")

    # As_equiv should be very close to As
    np.testing.assert_allclose(As_equiv_z, params["As"], rtol=1e-3,
                               err_msg="As_equiv should equal As when wa=0")

    # Verify spectra match
    z_i = 0.5
    k_lin, Pm, Pcb = _class_pk(params, [z_i])
    spec_orig = _compute_cleft_spectra(k_lin, Pcb[0], Pm[0])

    equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
    k_lin_eq, Pm_eq, Pcb_eq = _class_pk(equiv_params, [z_i])
    spec_equiv = _compute_cleft_spectra(k_lin_eq, Pcb_eq[0], Pm_eq[0])

    for s_idx in [0, 1, 5]:
        np.testing.assert_allclose(
            spec_equiv[s_idx], spec_orig[s_idx], rtol=0.01,
            err_msg=f"wa=0 regression: CLEFT spec {s_idx} at z={z_i} differs",
        )


# ---------------------------------------------------------------------------
# Test: LPT_RSD P_ij (RedshiftSpaceBiasedTracerSpectra)
# ---------------------------------------------------------------------------

@requires_classy
@requires_lpt_rsd
@pytest.mark.parametrize("params", _W0WA_NONZERO, ids=_W0WA_NONZERO_IDS)
def test_lpt_rsd_spectral_equiv_vs_exact(params, plot_diagnostics):
    """LPT_RSD P_ij from equivalent wCDM matches true w0wa to within tolerance.

    For each test redshift, runs LPT_RSD on CLASS P_cb(k) and f(z) from
    the true w0wa cosmology and compares against LPT_RSD run on the equivalent
    wCDM cosmology determined by SpectralEquivalence.
    """
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    _RSD_SPEC_INDICES = [0, 1, 2, 3, 5]
    _RSD_SPEC_NAMES = {0: r"$P^{(0)}$", 1: r"$P^{(1)}$",
                       2: r"$P^{(2)}$", 3: r"$P^{(3)}$",
                       5: r"$P^{(5)}$"}

    all_spec_exact = {}
    all_spec_equiv = {}

    for z_i in _Z_TEST:
        # Exact: CLASS with true w0wa
        k_lin, Pcb_exact, f_z_exact = _class_pk_rsd(params, [z_i])
        spec_exact = _compute_lpt_rsd_spectra(k_lin, Pcb_exact[0], f_z_exact[0])

        # Spectral equiv: CLASS with equivalent wCDM at this z
        equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
        k_lin_eq, Pcb_eq, f_z_eq = _class_pk_rsd(equiv_params, [z_i])
        spec_equiv = _compute_lpt_rsd_spectra(k_lin_eq, Pcb_eq[0], f_z_eq[0])

        all_spec_exact[z_i] = spec_exact
        all_spec_equiv[z_i] = spec_equiv

        # Compare dominant spectra per multipole
        for ell_idx in range(3):
            for s_idx in _RSD_SPEC_INDICES:
                exact = spec_exact[ell_idx, s_idx]
                equiv = spec_equiv[ell_idx, s_idx]

                # Use a tighter mask to exclude zero crossings in RSD spectra
                mask = np.abs(exact) > 1e-4 * np.max(np.abs(exact))
                if not mask.any():
                    continue

                rel_err = np.abs(equiv[mask] - exact[mask]) / np.abs(exact[mask])
                median_err = np.median(rel_err)
                ell = ell_idx * 2
                assert median_err < _RTOL_RSD, (
                    f"LPT_RSD ell={ell} spec {s_idx} at z={z_i}: median rel error "
                    f"{median_err:.4f} > {_RTOL_RSD} (max={rel_err.max():.4f})"
                )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_rsd_residuals(
            _K, all_spec_exact, all_spec_equiv, params, list(_Z_TEST),
            _RSD_SPEC_INDICES, _RSD_SPEC_NAMES,
        )
        print(f"  Saved LPT_RSD diagnostic plot: {fname}")


# ---------------------------------------------------------------------------
# Regression test: wa=0 gives identical RSD spectra (no mapping)
# ---------------------------------------------------------------------------

@requires_classy
@requires_lpt_rsd
def test_lpt_rsd_wa_zero_identity():
    """With wa=0, spectral equivalence should produce identical LPT_RSD spectra.

    Since no w0wa mapping is needed, the equivalent wCDM should match the
    original cosmology exactly.
    """
    params = W0WA_COSMO_PARAMS["wa_zero"]
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    # w_equiv should be very close to w0
    np.testing.assert_allclose(w_equiv_z, params["w"], atol=1e-4,
                               err_msg="w_equiv should equal w0 when wa=0")

    # As_equiv should be very close to As
    np.testing.assert_allclose(As_equiv_z, params["As"], rtol=1e-3,
                               err_msg="As_equiv should equal As when wa=0")

    # Verify RSD spectra match at z=0.5
    z_i = 0.5
    k_lin, Pcb, f_z = _class_pk_rsd(params, [z_i])
    spec_orig = _compute_lpt_rsd_spectra(k_lin, Pcb[0], f_z[0])

    equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
    k_lin_eq, Pcb_eq, f_z_eq = _class_pk_rsd(equiv_params, [z_i])
    spec_equiv = _compute_lpt_rsd_spectra(k_lin_eq, Pcb_eq[0], f_z_eq[0])

    for ell_idx in range(3):
        for s_idx in [0, 1, 5]:
            orig = spec_orig[ell_idx, s_idx]
            equiv = spec_equiv[ell_idx, s_idx]

            # Only compare where signal is nontrivial (skip near-zero components)
            mask = np.abs(orig) > 1e-4 * np.max(np.abs(orig))
            if not mask.any():
                continue

            rel_err = np.abs(equiv[mask] - orig[mask]) / np.abs(orig[mask])
            median_err = np.median(rel_err)
            ell = ell_idx * 2
            assert median_err < 0.01, (
                f"wa=0 regression: LPT_RSD ell={ell} spec {s_idx} at z={z_i}: "
                f"median rel error {median_err:.4f} > 0.01"
            )


# ---------------------------------------------------------------------------
# Helpers for HEFT P_11 vs csstemu tests
# ---------------------------------------------------------------------------

def _get_heft_p11_with_spectral_equiv(params, z_arr):
    """Run SpectralEquivalence + HEFT P_ij emulator and return P_11(k, z).

    Returns:
        k_heft: 1d array of k values in h/Mpc
        p11: dict mapping z_i -> 1d array of P_11(k) in (Mpc/h)^3
    """
    from gholax.theory.spectral_equivalence import SpectralEquivalence, build_equiv_cparam_grid
    from gholax.theory.emulator import PijEmulator

    se = SpectralEquivalence(
        z=z_arr,
        z_lss=1089.0,
        n_newton=15,
        n_int_chi=4096,
        sigma8_emulator_file_name=_SIGMA8_EMU_FILE,
    )
    state = {}
    state = se.compute(state, params)

    pij_emu = PijEmulator(_HEFT_PIJ_EMU_CONFIG)
    k_heft = np.array(pij_emu.pij_emus[0].k)

    cparam_grid = build_equiv_cparam_grid(params, jnp.array(z_arr), state)
    pk_ij = pij_emu.predict(cparam_grid)  # (nz, nspec, nk)

    p11 = {}
    for i, z_i in enumerate(z_arr):
        p11[z_i] = np.array(pk_ij[i, 0, :])

    return k_heft, p11


def _get_csstemu_pnonlin(params, z_arr, use_spectral_equiv=False):
    """Run csstemu and return the non-linear matter P(k, z).

    For cosmologies within csstemu's native range, evaluates directly.
    For cosmologies outside the range (e.g. DESI DR2 best fit), uses
    csstemu's SpectralEquivalence to map into the valid region.

    Returns:
        k_csst: 1d array of k in h/Mpc
        pk_nl: dict mapping z_i -> 1d array of P_nl(k) in (Mpc/h)^3
    """
    from CEmulator.Emulator import Pkmm_CEmulator
    from CEmulator.SpectralEquivalence import SpectralEquivalence_DE, SpectralEquivalence_As

    h = params["H0"] / 100.0
    Ob = params["ombh2"] / h**2
    Oc = params["omch2"] / h**2
    mnu = params["mnu"]
    As = params["As"] * 1e-9
    ns = params["ns"]
    w0 = params["w"]
    wa = params.get("wa", 0.0)

    # csstemu k range: [0.00628, 10] h/Mpc
    _KMIN_CSST = 0.007
    k_csst = np.logspace(np.log10(_KMIN_CSST), np.log10(_KMAX), _NK)

    emu = Pkmm_CEmulator()

    if not use_spectral_equiv:
        # Direct evaluation (cosmology must be within emulator range)
        emu.set_cosmos(Omegab=Ob, Omegac=Oc, H0=params["H0"], As=As,
                       ns=ns, w=w0, wa=wa, mnu=mnu)
        pk_all = emu.get_pknl(z=list(z_arr), k=k_csst)
        pk_nl = {}
        for i, z_i in enumerate(z_arr):
            pk_nl[z_i] = pk_all[i]
        return k_csst, pk_nl

    # Use csstemu's spectral equivalence for out-of-range cosmologies
    emu.set_cosmos(Omegab=Ob, Omegac=Oc, H0=params["H0"], As=As,
                   ns=ns, w=w0, wa=wa, mnu=mnu, checkbound=False)
    z_np = np.array(z_arr, dtype=np.float64)
    w0_arr, wa_arr = SpectralEquivalence_DE(emu, z_np, wCDM=False)
    As_arr = SpectralEquivalence_As(emu, z_np, w0_arr, wa_arr, sigma8_type='CLASS')

    pk_nl = {}
    for i, z_i in enumerate(z_arr):
        emu.set_cosmos(Omegab=Ob, Omegac=Oc, H0=params["H0"], As=As_arr[i],
                       ns=ns, w=w0_arr[i], wa=wa_arr[i], mnu=mnu)
        pk = emu.get_pknl(z=[float(z_i)], k=k_csst)
        pk_nl[z_i] = pk[0]

    return k_csst, pk_nl


def _plot_heft_vs_csstemu_residuals(k_heft, p11_heft, k_csst, pk_csst,
                                     params, z_values):
    """Plot HEFT P_11 (with spectral equiv) vs csstemu residuals."""
    import matplotlib.pyplot as plt

    label = _cosmo_label(params)
    nz = len(z_values)

    fig, axes = plt.subplots(2, nz, figsize=(5 * nz, 7), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"HEFT P_11 (spectral equiv) vs csstemu\n{label}", fontsize=13)

    # Overlapping k range
    k_lo = max(k_heft.min(), k_csst.min())
    k_hi = min(k_heft.max(), k_csst.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for col, z_i in enumerate(z_values):
        p11_interp = np.interp(k_common, k_heft, p11_heft[z_i])
        csst_interp = np.interp(k_common, k_csst, pk_csst[z_i])

        # Top panel: P(k) comparison
        ax_top = axes[0, col]
        ax_top.loglog(k_common, p11_interp, "C0-", lw=1.2, label="HEFT+SE")
        ax_top.loglog(k_common, csst_interp, "C1--", lw=1.2, label="csstemu")
        ax_top.set_title(f"z = {z_i}")
        ax_top.set_ylabel(r"$P_{mm}(k)$ [(Mpc/$h$)$^3$]")
        ax_top.legend(fontsize=9)

        # Bottom panel: relative residual
        ax_bot = axes[1, col]
        mask = csst_interp > 0
        rel_err = np.full_like(k_common, np.nan)
        rel_err[mask] = (p11_interp[mask] - csst_interp[mask]) / csst_interp[mask]

        ax_bot.semilogx(k_common, rel_err, "C0-", lw=0.8)
        ax_bot.axhline(0, color="k", ls="--", lw=0.5)
        ax_bot.axhline(_RTOL_HEFT_VS_CSSTEMU, color="r", ls=":", lw=0.5, alpha=0.5)
        ax_bot.axhline(-_RTOL_HEFT_VS_CSSTEMU, color="r", ls=":", lw=0.5, alpha=0.5)
        ax_bot.set_ylabel("(HEFT - csstemu) / csstemu")
        ax_bot.set_xlabel(r"$k$ [$h$/Mpc]")
        ax_bot.set_ylim(-0.15, 0.15)

    fig.tight_layout()
    fname = os.path.join(_PLOT_DIR, f"heft_p11_vs_csstemu_{label}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Test: HEFT P_11 (with spectral equiv) vs csstemu
# ---------------------------------------------------------------------------

@requires_csstemu
@pytest.mark.parametrize("params", _W0WA_NONZERO, ids=_W0WA_NONZERO_IDS)
def test_heft_p11_spectral_equiv_vs_csstemu(params, plot_diagnostics):
    """Non-linear HEFT P_11 with spectral equivalence matches csstemu.

    For w0wa cosmologies, computes the non-linear matter power spectrum P_11 via:
      - gholax HEFT emulator fed equivalent wCDM params from SpectralEquivalence
      - csstemu (natively supports w0wa within its range)
    and compares on the overlapping k range at each test redshift.
    """
    z_test = [0.0, 0.5, 1.0]
    k_heft, p11_heft = _get_heft_p11_with_spectral_equiv(params, z_test)
    k_csst, pk_csst = _get_csstemu_pnonlin(params, z_test)

    # Compare on overlapping k range
    k_lo = max(k_heft.min(), k_csst.min())
    k_hi = min(k_heft.max(), k_csst.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for z_i in z_test:
        p11_interp = np.interp(k_common, k_heft, p11_heft[z_i])
        csst_interp = np.interp(k_common, k_csst, pk_csst[z_i])

        mask = csst_interp > 0
        rel_err = np.abs(p11_interp[mask] - csst_interp[mask]) / csst_interp[mask]
        median_err = np.median(rel_err)
        assert median_err < _RTOL_HEFT_VS_CSSTEMU, (
            f"HEFT P_11 vs csstemu at z={z_i}: median rel error {median_err:.4f} "
            f"> {_RTOL_HEFT_VS_CSSTEMU} (max={rel_err.max():.4f})"
        )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_heft_vs_csstemu_residuals(
            k_heft, p11_heft, k_csst, pk_csst, params, z_test,
        )
        print(f"  Saved HEFT vs csstemu diagnostic plot: {fname}")


@requires_csstemu
def test_heft_p11_vs_csstemu_wa_zero(plot_diagnostics):
    """With wa=0, HEFT P_11 (no spectral equiv needed) should match csstemu.

    Baseline test establishing that HEFT and csstemu emulators agree
    for standard wCDM before spectral equivalence is involved.
    """
    params = W0WA_COSMO_PARAMS["wa_zero"]
    z_test = [0.0, 0.5, 1.0]

    k_heft, p11_heft = _get_heft_p11_with_spectral_equiv(params, z_test)
    k_csst, pk_csst = _get_csstemu_pnonlin(params, z_test)

    k_lo = max(k_heft.min(), k_csst.min())
    k_hi = min(k_heft.max(), k_csst.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for z_i in z_test:
        p11_interp = np.interp(k_common, k_heft, p11_heft[z_i])
        csst_interp = np.interp(k_common, k_csst, pk_csst[z_i])

        mask = csst_interp > 0
        rel_err = np.abs(p11_interp[mask] - csst_interp[mask]) / csst_interp[mask]
        median_err = np.median(rel_err)
        assert median_err < _RTOL_HEFT_VS_CSSTEMU, (
            f"HEFT P_11 vs csstemu (wa=0 baseline) at z={z_i}: median rel error "
            f"{median_err:.4f} > {_RTOL_HEFT_VS_CSSTEMU} (max={rel_err.max():.4f})"
        )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_heft_vs_csstemu_residuals(
            k_heft, p11_heft, k_csst, pk_csst, params, z_test,
        )
        print(f"  Saved HEFT vs csstemu wa=0 baseline plot: {fname}")


@requires_csstemu
def test_heft_p11_spectral_equiv_improves_over_no_correction(plot_diagnostics):
    """Spectral equivalence should reduce the HEFT vs csstemu discrepancy for w0wa.

    Compares the HEFT P_11 residual vs csstemu with and without spectral equivalence
    for a w0wa cosmology. The spectral equivalence version should have smaller
    residuals (or at least not be worse).
    """
    from gholax.theory.emulator import PijEmulator
    from gholax.theory.spectral_equivalence import build_equiv_cparam_grid

    params = W0WA_COSMO_PARAMS["wa_neg"]

    # With spectral equivalence
    k_heft, p11_with_se = _get_heft_p11_with_spectral_equiv(params, _Z_TEST)

    # Without spectral equivalence: just use w=w0, ignore wa
    pij_emu = PijEmulator(_HEFT_PIJ_EMU_CONFIG)
    params_no_wa = dict(params)
    params_no_wa.pop("wa", None)
    cparam_grid_naive = build_equiv_cparam_grid(params_no_wa, jnp.array(_Z_TEST), {})
    pk_naive = pij_emu.predict(cparam_grid_naive)
    p11_no_se = {z_i: np.array(pk_naive[i, 0, :]) for i, z_i in enumerate(_Z_TEST)}

    # csstemu reference
    k_csst, pk_csst = _get_csstemu_pnonlin(params, _Z_TEST)

    k_lo = max(k_heft.min(), k_csst.min())
    k_hi = min(k_heft.max(), k_csst.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for z_i in _Z_TEST:
        csst_interp = np.interp(k_common, k_csst, pk_csst[z_i])
        p11_se_interp = np.interp(k_common, k_heft, p11_with_se[z_i])
        p11_no_se_interp = np.interp(k_common, k_heft, p11_no_se[z_i])

        mask = csst_interp > 0
        err_with_se = np.median(np.abs(p11_se_interp[mask] - csst_interp[mask]) / csst_interp[mask])
        err_no_se = np.median(np.abs(p11_no_se_interp[mask] - csst_interp[mask]) / csst_interp[mask])

        assert err_with_se <= err_no_se * 1.1, (
            f"Spectral equiv should not degrade HEFT vs csstemu at z={z_i}: "
            f"with_SE={err_with_se:.4f}, without_SE={err_no_se:.4f}"
        )

    if plot_diagnostics:
        import matplotlib.pyplot as plt

        _ensure_plot_dir()
        label = _cosmo_label(params)
        nz = len(_Z_TEST)
        fig, axes = plt.subplots(1, nz, figsize=(5 * nz, 4), squeeze=False)
        fig.suptitle(f"HEFT P_11 vs csstemu: with vs without spectral equiv\n{label}",
                     fontsize=13)

        for col, z_i in enumerate(_Z_TEST):
            ax = axes[0, col]
            csst_interp = np.interp(k_common, k_csst, pk_csst[z_i])
            p11_se_interp = np.interp(k_common, k_heft, p11_with_se[z_i])
            p11_no_se_interp = np.interp(k_common, k_heft, p11_no_se[z_i])

            mask = csst_interp > 0
            err_se = np.full_like(k_common, np.nan)
            err_no = np.full_like(k_common, np.nan)
            err_se[mask] = (p11_se_interp[mask] - csst_interp[mask]) / csst_interp[mask]
            err_no[mask] = (p11_no_se_interp[mask] - csst_interp[mask]) / csst_interp[mask]

            ax.semilogx(k_common, err_se, "C0-", lw=1, label="with SE")
            ax.semilogx(k_common, err_no, "C1--", lw=1, label="without SE")
            ax.axhline(0, color="k", ls="--", lw=0.5)
            ax.set_title(f"z = {z_i}")
            ax.set_xlabel(r"$k$ [$h$/Mpc]")
            ax.set_ylabel("(HEFT - csstemu) / csstemu")
            ax.set_ylim(-0.15, 0.15)
            ax.legend(fontsize=9)

        fig.tight_layout()
        fname = os.path.join(_PLOT_DIR, f"heft_p11_se_improvement_{label}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved SE improvement plot: {fname}")


# ---------------------------------------------------------------------------
# Tests at DESI DR2 BAO x CMB best-fit w0wa
# (Table V of arXiv:2503.14738: w0=-0.42, wa=-1.75)
#
# The large |wa| pushes the spectral equivalence harder than the generic
# test cosmologies, so tolerances are relaxed accordingly.
# ---------------------------------------------------------------------------

_RTOL_CLEFT_DESI = 0.03         # 3% for CLEFT P_ij (2.1% observed at z=2)
_RTOL_DENSITY_SHAPE_DESI = 0.05 # 5% for density-shape IA (3.2% observed)
_RTOL_SHAPE_SHAPE_DESI = 0.07   # 7% for shape-shape IA (5.5% observed at m=1, z=1)
_RTOL_RSD_DESI = 0.25           # 25% for LPT_RSD (ell=4 hexadecapole sensitive to f(z) mismatch)

@requires_classy
@requires_cleft
def test_cleft_pij_desi_dr2_best_fit(plot_diagnostics):
    """CLEFT P_ij spectral equivalence accuracy at DESI DR2 BAO+CMB best-fit w0wa.

    The DESI DR2 best fit (w0=-0.42, wa=-1.75) has large |wa|, making it a
    stringent test of the spectral equivalence approximation.
    """
    params = _DESI_DR2_PARAMS
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    _CLEFT_SPEC_INDICES = [0, 1, 2, 3, 5]
    _CLEFT_SPEC_NAMES = {0: r"$P_{mm}$", 1: r"$P_{cb,m}^{(0)}$",
                         2: r"$P_{cb,cb}^{(0)}$", 3: r"$P_{cb,m}^{(1)}$",
                         5: r"$P_{cb,cb}^{(2)}$"}

    all_spec_exact = {}
    all_spec_equiv = {}

    for z_i in _Z_TEST:
        k_lin, Pm_exact, Pcb_exact = _class_pk(params, [z_i])
        spec_exact = _compute_cleft_spectra(k_lin, Pcb_exact[0], Pm_exact[0])

        equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
        k_lin_eq, Pm_eq, Pcb_eq = _class_pk(equiv_params, [z_i])
        spec_equiv = _compute_cleft_spectra(k_lin_eq, Pcb_eq[0], Pm_eq[0])

        all_spec_exact[z_i] = spec_exact
        all_spec_equiv[z_i] = spec_equiv

        for s_idx in _CLEFT_SPEC_INDICES:
            exact = spec_exact[s_idx]
            equiv = spec_equiv[s_idx]

            mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
            if not mask.any():
                continue

            rel_err = np.abs(equiv[mask] - exact[mask]) / np.abs(exact[mask])
            median_err = np.median(rel_err)
            assert median_err < _RTOL_CLEFT_DESI, (
                f"DESI DR2 best fit: CLEFT spec {s_idx} at z={z_i}: "
                f"median rel error {median_err:.4f} > {_RTOL_CLEFT_DESI} "
                f"(max={rel_err.max():.4f})"
            )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_cleft_residuals(
            _K, all_spec_exact, all_spec_equiv, params, list(_Z_TEST),
            _CLEFT_SPEC_INDICES, _CLEFT_SPEC_NAMES,
        )
        print(f"  Saved DESI DR2 CLEFT diagnostic plot: {fname}")


@requires_classy
@requires_density_shape
def test_density_shape_ia_desi_dr2_best_fit(plot_diagnostics):
    """DensityShape IA spectral equivalence accuracy at DESI DR2 BAO+CMB best-fit w0wa."""
    params = _DESI_DR2_PARAMS
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    _DS_SPEC_INDICES = [0, 1, 4]
    _DS_SPEC_NAMES = {0: r"$P_{g\gamma}^{(0)}$", 1: r"$P_{g\gamma}^{(1)}$",
                      4: r"$P_{g\gamma}^{(4)}$"}

    all_spec_exact = {}
    all_spec_equiv = {}

    for z_i in _Z_TEST:
        k_lin, _, Pcb_exact = _class_pk(params, [z_i])
        spec_exact = _compute_density_shape_spectra(k_lin, Pcb_exact[0])

        equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
        k_lin_eq, _, Pcb_eq = _class_pk(equiv_params, [z_i])
        spec_equiv = _compute_density_shape_spectra(k_lin_eq, Pcb_eq[0])

        all_spec_exact[z_i] = spec_exact
        all_spec_equiv[z_i] = spec_equiv

        for s_idx in _DS_SPEC_INDICES:
            exact = spec_exact[s_idx]
            equiv = spec_equiv[s_idx]

            mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
            if not mask.any():
                continue

            rel_err = np.abs(equiv[mask] - exact[mask]) / np.abs(exact[mask])
            median_err = np.median(rel_err)
            assert median_err < _RTOL_DENSITY_SHAPE_DESI, (
                f"DESI DR2 best fit: DensityShape spec {s_idx} at z={z_i}: "
                f"median rel error {median_err:.4f} > {_RTOL_DENSITY_SHAPE_DESI} "
                f"(max={rel_err.max():.4f})"
            )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_ia_residuals(
            _K, all_spec_exact, all_spec_equiv, params, list(_Z_TEST),
            _DS_SPEC_INDICES, "DensityShape IA (DESI DR2)",
            "density_shape_residuals_desi_dr2", _DS_SPEC_NAMES,
        )
        print(f"  Saved DESI DR2 DensityShape diagnostic plot: {fname}")


@requires_classy
@requires_shape_shape
def test_shape_shape_ia_desi_dr2_best_fit(plot_diagnostics):
    """ShapeShape IA spectral equivalence accuracy at DESI DR2 BAO+CMB best-fit w0wa."""
    params = _DESI_DR2_PARAMS
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    _SS_SPEC_INDICES = [0, 1]
    _SS_SPEC_NAMES = {0: r"$P_{\gamma\gamma}^{(0)}$",
                      1: r"$P_{\gamma\gamma}^{(1)}$"}
    _SS_M_VALUES = [0, 1, 2]

    all_spec_exact = {}
    all_spec_equiv = {}

    for z_i in _Z_TEST:
        k_lin, _, Pcb_exact = _class_pk(params, [z_i])
        spec_exact = _compute_shape_shape_spectra(k_lin, Pcb_exact[0])

        equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
        k_lin_eq, _, Pcb_eq = _class_pk(equiv_params, [z_i])
        spec_equiv = _compute_shape_shape_spectra(k_lin_eq, Pcb_eq[0])

        all_spec_exact[z_i] = spec_exact
        all_spec_equiv[z_i] = spec_equiv

        for m in _SS_M_VALUES:
            for s_idx in _SS_SPEC_INDICES:
                exact = spec_exact[m, s_idx]
                equiv = spec_equiv[m, s_idx]

                mask = np.abs(exact) > 1e-6 * np.max(np.abs(exact))
                if not mask.any():
                    continue

                rel_err = np.abs(equiv[mask] - exact[mask]) / np.abs(exact[mask])
                median_err = np.median(rel_err)
                assert median_err < _RTOL_SHAPE_SHAPE_DESI, (
                    f"DESI DR2 best fit: ShapeShape m={m} spec {s_idx} at z={z_i}: "
                    f"median rel error {median_err:.4f} > {_RTOL_SHAPE_SHAPE_DESI} "
                    f"(max={rel_err.max():.4f})"
                )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_ia_residuals(
            _K, all_spec_exact, all_spec_equiv, params, list(_Z_TEST),
            _SS_SPEC_INDICES, "ShapeShape IA (DESI DR2)",
            "shape_shape_residuals_desi_dr2", _SS_SPEC_NAMES,
            m_values=_SS_M_VALUES,
        )
        print(f"  Saved DESI DR2 ShapeShape diagnostic plot: {fname}")


@requires_classy
@requires_lpt_rsd
def test_lpt_rsd_desi_dr2_best_fit(plot_diagnostics):
    """LPT_RSD spectral equivalence accuracy at DESI DR2 BAO+CMB best-fit w0wa.

    The large |wa|=-1.75 makes this a particularly stringent test of the
    growth rate f(z) approximation under spectral equivalence.
    """
    params = _DESI_DR2_PARAMS
    w_equiv_z, As_equiv_z, z_equiv = _run_spectral_equivalence(params)

    _RSD_SPEC_INDICES = [0, 1, 2, 3, 5]
    _RSD_SPEC_NAMES = {0: r"$P^{(0)}$", 1: r"$P^{(1)}$",
                       2: r"$P^{(2)}$", 3: r"$P^{(3)}$",
                       5: r"$P^{(5)}$"}

    all_spec_exact = {}
    all_spec_equiv = {}

    for z_i in _Z_TEST:
        k_lin, Pcb_exact, f_z_exact = _class_pk_rsd(params, [z_i])
        spec_exact = _compute_lpt_rsd_spectra(k_lin, Pcb_exact[0], f_z_exact[0])

        equiv_params = _make_equiv_params(params, z_i, w_equiv_z, As_equiv_z, z_equiv)
        k_lin_eq, Pcb_eq, f_z_eq = _class_pk_rsd(equiv_params, [z_i])
        spec_equiv = _compute_lpt_rsd_spectra(k_lin_eq, Pcb_eq[0], f_z_eq[0])

        all_spec_exact[z_i] = spec_exact
        all_spec_equiv[z_i] = spec_equiv

        for ell_idx in range(3):
            for s_idx in _RSD_SPEC_INDICES:
                exact = spec_exact[ell_idx, s_idx]
                equiv = spec_equiv[ell_idx, s_idx]

                mask = np.abs(exact) > 1e-4 * np.max(np.abs(exact))
                if not mask.any():
                    continue

                rel_err = np.abs(equiv[mask] - exact[mask]) / np.abs(exact[mask])
                median_err = np.median(rel_err)
                ell = ell_idx * 2
                assert median_err < _RTOL_RSD_DESI, (
                    f"DESI DR2 best fit: LPT_RSD ell={ell} spec {s_idx} at z={z_i}: "
                    f"median rel error {median_err:.4f} > {_RTOL_RSD_DESI} "
                    f"(max={rel_err.max():.4f})"
                )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_rsd_residuals(
            _K, all_spec_exact, all_spec_equiv, params, list(_Z_TEST),
            _RSD_SPEC_INDICES, _RSD_SPEC_NAMES,
        )
        print(f"  Saved DESI DR2 LPT_RSD diagnostic plot: {fname}")


@requires_csstemu
def test_heft_p11_desi_dr2_best_fit(plot_diagnostics):
    """HEFT P_11 with gholax spectral equiv vs csstemu spectral equiv at DESI DR2 best fit.

    Both gholax and csstemu use independent spectral equivalence implementations
    to map the DESI DR2 best-fit w0wa cosmology (w0=-0.42, wa=-1.75) into their
    respective emulator-valid ranges. This tests that the two independent SE+emulator
    pipelines agree on P_mm(k,z).
    """
    params = _DESI_DR2_PARAMS
    z_test = [0.0, 0.5, 1.0]

    k_heft, p11_heft = _get_heft_p11_with_spectral_equiv(params, z_test)
    k_csst, pk_csst = _get_csstemu_pnonlin(params, z_test, use_spectral_equiv=True)

    k_lo = max(k_heft.min(), k_csst.min())
    k_hi = min(k_heft.max(), k_csst.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for z_i in z_test:
        p11_interp = np.interp(k_common, k_heft, p11_heft[z_i])
        csst_interp = np.interp(k_common, k_csst, pk_csst[z_i])

        mask = csst_interp > 0
        rel_err = np.abs(p11_interp[mask] - csst_interp[mask]) / csst_interp[mask]
        median_err = np.median(rel_err)
        assert median_err < _RTOL_HEFT_VS_CSSTEMU, (
            f"DESI DR2 best fit: HEFT P_11 vs csstemu at z={z_i}: "
            f"median rel error {median_err:.4f} > {_RTOL_HEFT_VS_CSSTEMU} "
            f"(max={rel_err.max():.4f})"
        )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_heft_vs_csstemu_residuals(
            k_heft, p11_heft, k_csst, pk_csst, params, z_test,
        )
        print(f"  Saved DESI DR2 HEFT vs csstemu diagnostic plot: {fname}")


# ---------------------------------------------------------------------------
# Tests comparing gholax vs csstemu spectral equivalence intermediate outputs
# ---------------------------------------------------------------------------

# Use z_lss=1100 for both implementations so distance targets are identical
_Z_LSS_COMMON = 1100.0

_SE_TEST_IDS = ["wa_neg", "wa_pos", "desi_dr2_bao_cmb"]
_SE_TEST_PARAMS = [W0WA_COSMO_PARAMS[k] for k in _SE_TEST_IDS]

# wa_pos produces w_equiv > -0.7 at low z, which is outside csstemu's
# emulator range, so SpectralEquivalence_As fails with a bound check.
# Use a subset that stays in range for wCDM-mode As tests.
_SE_WCDM_IDS = ["wa_neg", "desi_dr2_bao_cmb"]
_SE_WCDM_PARAMS = [W0WA_COSMO_PARAMS[k] for k in _SE_WCDM_IDS]


def _run_gholax_se(params, z_arr, z_lss=_Z_LSS_COMMON):
    """Run gholax SpectralEquivalence and return (w_equiv, As_equiv)."""
    from gholax.theory.spectral_equivalence import SpectralEquivalence

    se = SpectralEquivalence(
        z=z_arr,
        z_lss=z_lss,
        n_newton=15,
        n_int_chi=4096,
        sigma8_emulator_file_name=_SIGMA8_EMU_FILE,
    )
    state = {}
    state = se.compute(state, params)
    return np.array(state["w_equiv_z"]), np.array(state["As_equiv_z"])


def _run_csstemu_se(params, z_arr, wCDM=True, z_lss=_Z_LSS_COMMON):
    """Run csstemu SpectralEquivalence and return equivalent DE params + As.

    Returns:
        If wCDM=True:  (w_arr, As_arr)        — constant w per redshift
        If wCDM=False: (w0_arr, wa_arr, As_arr) — (w0, wa) pair per redshift
    """
    from CEmulator.Emulator import Pkmm_CEmulator
    from CEmulator.SpectralEquivalence import SpectralEquivalence_DE, SpectralEquivalence_As

    h = params["H0"] / 100.0
    Ob = params["ombh2"] / h**2
    Oc = params["omch2"] / h**2

    emu = Pkmm_CEmulator()
    emu.set_cosmos(Omegab=Ob, Omegac=Oc, H0=params["H0"],
                   As=params["As"] * 1e-9, ns=params["ns"],
                   w=params["w"], wa=params.get("wa", 0.0),
                   mnu=params["mnu"], checkbound=False)

    z_np = np.array(z_arr, dtype=np.float64)

    if wCDM:
        w_arr = SpectralEquivalence_DE(emu, z_np, wCDM=True)
        As_arr = SpectralEquivalence_As(emu, z_np, w_arr, sigma8_type='CLASS')
        return w_arr, As_arr
    else:
        w0_arr, wa_arr = SpectralEquivalence_DE(emu, z_np, wCDM=False)
        As_arr = SpectralEquivalence_As(emu, z_np, w0_arr, wa_arr,
                                        sigma8_type='CLASS')
        return w0_arr, wa_arr, As_arr


def _compute_chi_to_zlss(params, z_arr, w_de, wa_de=None, z_lss=_Z_LSS_COMMON):
    """Compute chi(z_i -> z_LSS) for a set of DE params using csstemu Cosmology.

    Args:
        params: base cosmological parameters
        z_arr: redshifts
        w_de: array of w values (one per z)
        wa_de: array of wa values (one per z), or None for wCDM
        z_lss: redshift of last scattering surface

    Returns:
        chi_arr: array of comoving distances chi(z_i -> z_LSS) in Mpc
    """
    from CEmulator.cosmology import Cosmology

    h = params["H0"] / 100.0
    Ob = params["ombh2"] / h**2
    Om = (params["ombh2"] + params["omch2"] + params["mnu"] / 93.14) / h**2

    chi_arr = np.zeros(len(z_arr))
    for i, z_i in enumerate(z_arr):
        c = Cosmology()
        wa_i = wa_de[i] if wa_de is not None else 0.0
        c.set_cosmos({
            'Omegab': Ob, 'Omegam': Om, 'H0': params["H0"],
            'A': params["As"], 'ns': params["ns"],
            'mnu': params["mnu"], 'w': float(w_de[i]), 'wa': float(wa_i),
        })
        chi_arr[i] = float(c.comoving_distance(z_lss) - c.comoving_distance(float(z_i)))
    return chi_arr


def _plot_se_comparison(z_arr, gholax_w, csstemu_w_wcdm, csstemu_w0_w0wa,
                        csstemu_wa_w0wa, gholax_As, csstemu_As_wcdm,
                        csstemu_As_w0wa, chi_target, chi_wcdm, chi_w0wa,
                        params):
    """Plot comparison of gholax vs csstemu spectral equivalence outputs."""
    import matplotlib.pyplot as plt

    label = _cosmo_label(params)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Spectral Equivalence: gholax vs csstemu\n{label}", fontsize=13)

    # Panel (0,0): w_equiv(z) comparison — wCDM mode
    ax = axes[0, 0]
    ax.plot(z_arr, gholax_w, "C0o-", label="gholax", ms=5)
    ax.plot(z_arr, csstemu_w_wcdm, "C1s--", label="csstemu wCDM", ms=5)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$w_{\rm equiv}(z)$")
    ax.set_title("wCDM: equivalent w(z)")
    ax.legend(fontsize=9)

    # Panel (0,1): csstemu w0wa mode — w0(z) and wa(z)
    ax = axes[0, 1]
    ax.plot(z_arr, csstemu_w0_w0wa, "C2o-", label=r"$w_0^{\rm equiv}$", ms=5)
    ax.plot(z_arr, csstemu_wa_w0wa, "C3s-", label=r"$w_a^{\rm equiv}$", ms=5)
    ax.axhline(params["w"], color="C2", ls=":", lw=0.7, alpha=0.5)
    ax.axhline(params.get("wa", 0.0), color="C3", ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel("DE parameter")
    ax.set_title(r"$w_0 w_a$CDM: equivalent $(w_0, w_a)(z)$")
    ax.legend(fontsize=9)

    # Panel (0,2): As_equiv(z)
    ax = axes[0, 2]
    ax.plot(z_arr, gholax_As * 1e9, "C0o-", label="gholax", ms=5)
    ax.plot(z_arr, csstemu_As_wcdm * 1e9, "C1s--", label="csstemu wCDM", ms=5)
    ax.plot(z_arr, csstemu_As_w0wa * 1e9, "C4d:", label=r"csstemu $w_0 w_a$", ms=5)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$A_{s,\rm equiv} \times 10^9$")
    ax.set_title("Equivalent $A_s(z)$")
    ax.legend(fontsize=9)

    # Panel (1,0): w_equiv relative difference (gholax vs csstemu wCDM)
    ax = axes[1, 0]
    rel_dw = (gholax_w - csstemu_w_wcdm) / np.abs(csstemu_w_wcdm)
    ax.plot(z_arr, rel_dw, "C0o-", ms=5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$(w_{\rm gholax} - w_{\rm csstemu}) / |w_{\rm csstemu}|$")
    ax.set_title("wCDM: relative w difference")

    # Panel (1,1): chi(z→z_LSS) residuals for both modes
    ax = axes[1, 1]
    rel_chi_wcdm = np.abs(1.0 - chi_wcdm / chi_target)
    rel_chi_w0wa = np.abs(1.0 - chi_w0wa / chi_target)
    ax.semilogy(z_arr, rel_chi_wcdm, "C1s-", label="wCDM", ms=5)
    ax.semilogy(z_arr, rel_chi_w0wa, "C4d-", label=r"$w_0 w_a$CDM", ms=5)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$|1 - \chi_{\rm equiv}/\chi_{\rm target}|$")
    ax.set_title("Distance matching residuals")
    ax.legend(fontsize=9)

    # Panel (1,2): As relative difference
    ax = axes[1, 2]
    rel_dAs_wcdm = (gholax_As - csstemu_As_wcdm) / csstemu_As_wcdm
    rel_dAs_w0wa = (gholax_As - csstemu_As_w0wa) / csstemu_As_w0wa
    ax.plot(z_arr, rel_dAs_wcdm, "C1s-", label="vs csstemu wCDM", ms=5)
    ax.plot(z_arr, rel_dAs_w0wa, "C4d-", label=r"vs csstemu $w_0 w_a$", ms=5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$(A_{s,\rm gholax} - A_{s,\rm csstemu}) / A_{s,\rm csstemu}$")
    ax.set_title("Relative $A_s$ difference")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fname = os.path.join(_PLOT_DIR, f"se_comparison_{label}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


@requires_csstemu
@pytest.mark.parametrize("params", _SE_WCDM_PARAMS, ids=_SE_WCDM_IDS)
def test_spectral_equivalence_w_equiv_wcdm(params, plot_diagnostics):
    """gholax and csstemu wCDM spectral equivalence produce the same w_equiv(z).

    Both implementations solve for a constant w per redshift such that
    chi_wCDM(z -> z_LSS) = chi_w0wa(z -> z_LSS). This test verifies they
    agree on the resulting w_equiv values.

    Note: wa_pos is excluded because its w_equiv > -0.7 at low z falls outside
    csstemu's emulator range, causing SpectralEquivalence_As to fail.
    """
    z_arr = _Z_TEST

    gholax_w, gholax_As = _run_gholax_se(params, z_arr)
    csstemu_w, csstemu_As = _run_csstemu_se(params, z_arr, wCDM=True)

    # w_equiv comparison
    rel_err_w = np.abs(gholax_w - csstemu_w) / np.abs(csstemu_w)
    for i, z_i in enumerate(z_arr):
        assert rel_err_w[i] < 0.001, (
            f"w_equiv mismatch at z={z_i}: gholax={gholax_w[i]:.6f}, "
            f"csstemu={csstemu_w[i]:.6f}, rel_err={rel_err_w[i]:.6f}"
        )

    # As_equiv comparison (gholax uses NN sigma8 emulator, csstemu uses CLASS)
    rel_err_As = np.abs(gholax_As * 1e-9 - csstemu_As) / csstemu_As
    for i, z_i in enumerate(z_arr):
        assert rel_err_As[i] < 0.01, (
            f"As_equiv mismatch at z={z_i}: gholax={gholax_As[i]*1e-9:.6e}, "
            f"csstemu={csstemu_As[i]:.6e}, rel_err={rel_err_As[i]:.6f}"
        )


@requires_csstemu
@pytest.mark.parametrize("params", _SE_TEST_PARAMS, ids=_SE_TEST_IDS)
def test_spectral_equivalence_w0wa_distance_matching(params, plot_diagnostics):
    """csstemu w0waCDM spectral equivalence accurately matches distances.

    In the wCDM=False mode, csstemu finds (w0, wa) pairs per redshift.
    This test verifies that the resulting chi(z -> z_LSS) matches the
    target w0wa cosmology's chi to high accuracy.
    """
    z_arr = _Z_TEST

    # Target distances from the true w0wa cosmology
    chi_target = _compute_chi_to_zlss(params, z_arr,
                                       np.full(len(z_arr), params["w"]),
                                       np.full(len(z_arr), params.get("wa", 0.0)))

    # Equivalent w0wa params from csstemu
    w0_arr, wa_arr, As_arr = _run_csstemu_se(params, z_arr, wCDM=False)

    # Distances from the equivalent w0wa cosmologies
    chi_equiv = _compute_chi_to_zlss(params, z_arr, w0_arr, wa_arr)

    rel_err = np.abs(1.0 - chi_equiv / chi_target)
    for i, z_i in enumerate(z_arr):
        assert rel_err[i] < 1e-4, (
            f"w0wa SE distance mismatch at z={z_i}: "
            f"|1 - chi_equiv/chi_target| = {rel_err[i]:.2e}, "
            f"w0_equiv={w0_arr[i]:.4f}, wa_equiv={wa_arr[i]:.4f}"
        )


@requires_csstemu
@pytest.mark.parametrize("params", _SE_WCDM_PARAMS, ids=_SE_WCDM_IDS)
def test_spectral_equivalence_wcdm_distance_matching(params, plot_diagnostics):
    """Both gholax and csstemu wCDM spectral equivalence accurately match distances.

    Verifies that the constant-w values found by both implementations produce
    chi(z -> z_LSS) that matches the target w0wa cosmology.
    """
    z_arr = _Z_TEST

    # Target distances
    chi_target = _compute_chi_to_zlss(params, z_arr,
                                       np.full(len(z_arr), params["w"]),
                                       np.full(len(z_arr), params.get("wa", 0.0)))

    # gholax wCDM equiv distances
    gholax_w, _ = _run_gholax_se(params, z_arr)
    chi_gholax = _compute_chi_to_zlss(params, z_arr, gholax_w)

    # csstemu wCDM equiv distances
    csstemu_w, _ = _run_csstemu_se(params, z_arr, wCDM=True)
    chi_csstemu = _compute_chi_to_zlss(params, z_arr, csstemu_w)

    for i, z_i in enumerate(z_arr):
        err_gholax = abs(1.0 - chi_gholax[i] / chi_target[i])
        err_csstemu = abs(1.0 - chi_csstemu[i] / chi_target[i])
        assert err_gholax < 1e-4, (
            f"gholax wCDM distance mismatch at z={z_i}: "
            f"|1 - chi/chi_target| = {err_gholax:.2e}"
        )
        assert err_csstemu < 1e-4, (
            f"csstemu wCDM distance mismatch at z={z_i}: "
            f"|1 - chi/chi_target| = {err_csstemu:.2e}"
        )


@requires_csstemu
@pytest.mark.parametrize("params", _SE_WCDM_PARAMS, ids=_SE_WCDM_IDS)
def test_spectral_equivalence_full_comparison(params, plot_diagnostics):
    """Full comparison of gholax vs csstemu SE with diagnostic plots.

    Compares both wCDM and w0waCDM modes side-by-side:
    w_equiv, As_equiv, and distance residuals.
    """
    z_arr = _Z_TEST

    # gholax
    gholax_w, gholax_As = _run_gholax_se(params, z_arr)

    # csstemu wCDM
    csstemu_w_wcdm, csstemu_As_wcdm = _run_csstemu_se(params, z_arr, wCDM=True)

    # csstemu w0wa
    csstemu_w0_w0wa, csstemu_wa_w0wa, csstemu_As_w0wa = _run_csstemu_se(
        params, z_arr, wCDM=False
    )

    # Distance residuals
    chi_target = _compute_chi_to_zlss(params, z_arr,
                                       np.full(len(z_arr), params["w"]),
                                       np.full(len(z_arr), params.get("wa", 0.0)))
    chi_wcdm = _compute_chi_to_zlss(params, z_arr, csstemu_w_wcdm)
    chi_w0wa = _compute_chi_to_zlss(params, z_arr, csstemu_w0_w0wa, csstemu_wa_w0wa)

    # w0wa mode should match distances at least as well as wCDM
    err_wcdm = np.abs(1.0 - chi_wcdm / chi_target)
    err_w0wa = np.abs(1.0 - chi_w0wa / chi_target)
    for i, z_i in enumerate(z_arr):
        assert err_w0wa[i] <= err_wcdm[i] + 1e-6, (
            f"w0wa mode should match distances at least as well as wCDM at z={z_i}: "
            f"err_w0wa={err_w0wa[i]:.2e}, err_wCDM={err_wcdm[i]:.2e}"
        )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_se_comparison(
            z_arr, gholax_w, csstemu_w_wcdm, csstemu_w0_w0wa, csstemu_wa_w0wa,
            gholax_As, csstemu_As_wcdm, csstemu_As_w0wa,
            chi_target, chi_wcdm, chi_w0wa, params,
        )
        print(f"  Saved SE comparison plot: {fname}")
