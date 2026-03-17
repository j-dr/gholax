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
    import euclidemu2 as _ee2  # noqa: F401
    euclidemu2_available = True
except ImportError:
    euclidemu2_available = False

requires_cleft = pytest.mark.skipif(
    not cleft_available, reason="spinosaurus CLEFT not installed"
)
requires_density_shape = pytest.mark.skipif(
    not density_shape_available, reason="spinosaurus DensityShapeCorrelators not installed"
)
requires_shape_shape = pytest.mark.skipif(
    not shape_shape_available, reason="spinosaurus ShapeShapeCorrelators not installed"
)
requires_euclidemu2 = pytest.mark.skipif(
    not euclidemu2_available, reason="euclidemu2 not installed"
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

# HEFT P_11 emulator config
_HEFT_PIJ_EMU_CONFIG = "emu_config_dst_irresum.yaml"

# Tolerance for HEFT P_11 vs EuclidEmulator2 (different emulators, so larger tolerance)
_RTOL_HEFT_VS_EE2 = 0.05  # 5%


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


# ---------------------------------------------------------------------------
# Select test cosmologies with nonzero wa
# ---------------------------------------------------------------------------

_W0WA_NONZERO_IDS = ["wa_neg", "wa_pos"]
_W0WA_NONZERO = [W0WA_COSMO_PARAMS[k] for k in _W0WA_NONZERO_IDS]


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
# Helpers for HEFT P_11 vs EuclidEmulator2 tests
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


def _get_ee2_pnonlin(params, z_arr):
    """Run EuclidEmulator2 and return the non-linear matter P(k, z).

    Returns:
        k_ee2: 1d array of k in h/Mpc
        pk_nl: dict mapping z_i -> 1d array of P_nl(k) in (Mpc/h)^3
    """
    import euclidemu2 as ee2

    h = params["H0"] / 100.0
    omega_nu = params["mnu"] / 93.14
    Omm = (params["omch2"] + params["ombh2"] + omega_nu) / h**2
    Omb = params["ombh2"] / h**2

    cosmo_par = {
        "As": params["As"] * 1e-9,
        "ns": params["ns"],
        "Omb": Omb,
        "Omm": Omm,
        "h": h,
        "mnu": params["mnu"],
        "w": params["w"],
        "wa": params.get("wa", 0.0),
    }

    emu = ee2.PyEuclidEmulator()
    z_list = [float(z) for z in z_arr]
    k_ee2, pk_nl_dict, _, _ = emu.get_pnonlin(cosmo_par, z_list)

    pk_nl = {}
    for i, z_i in enumerate(z_arr):
        pk_nl[z_i] = np.array(pk_nl_dict[i])

    return np.array(k_ee2), pk_nl


def _plot_heft_vs_ee2_residuals(k_heft, p11_heft, k_ee2, pk_ee2, params, z_values):
    """Plot HEFT P_11 (with spectral equiv) vs EuclidEmulator2 residuals."""
    import matplotlib.pyplot as plt

    label = _cosmo_label(params)
    nz = len(z_values)

    fig, axes = plt.subplots(2, nz, figsize=(5 * nz, 7), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"HEFT P_11 (spectral equiv) vs EuclidEmulator2\n{label}", fontsize=13)

    # Overlapping k range
    k_lo = max(k_heft.min(), k_ee2.min())
    k_hi = min(k_heft.max(), k_ee2.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for col, z_i in enumerate(z_values):
        p11_interp = np.interp(k_common, k_heft, p11_heft[z_i])
        ee2_interp = np.interp(k_common, k_ee2, pk_ee2[z_i])

        # Top panel: P(k) comparison
        ax_top = axes[0, col]
        ax_top.loglog(k_common, p11_interp, "C0-", lw=1.2, label="HEFT+SE")
        ax_top.loglog(k_common, ee2_interp, "C1--", lw=1.2, label="EE2")
        ax_top.set_title(f"z = {z_i}")
        ax_top.set_ylabel(r"$P_{mm}(k)$ [(Mpc/$h$)$^3$]")
        ax_top.legend(fontsize=9)

        # Bottom panel: relative residual
        ax_bot = axes[1, col]
        mask = ee2_interp > 0
        rel_err = np.full_like(k_common, np.nan)
        rel_err[mask] = (p11_interp[mask] - ee2_interp[mask]) / ee2_interp[mask]

        ax_bot.semilogx(k_common, rel_err, "C0-", lw=0.8)
        ax_bot.axhline(0, color="k", ls="--", lw=0.5)
        ax_bot.axhline(_RTOL_HEFT_VS_EE2, color="r", ls=":", lw=0.5, alpha=0.5)
        ax_bot.axhline(-_RTOL_HEFT_VS_EE2, color="r", ls=":", lw=0.5, alpha=0.5)
        ax_bot.set_ylabel("(HEFT - EE2) / EE2")
        ax_bot.set_xlabel(r"$k$ [$h$/Mpc]")
        ax_bot.set_ylim(-0.15, 0.15)

    fig.tight_layout()
    fname = os.path.join(_PLOT_DIR, f"heft_p11_vs_ee2_{label}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Test: HEFT P_11 (with spectral equiv) vs EuclidEmulator2
# ---------------------------------------------------------------------------

@requires_euclidemu2
@pytest.mark.parametrize("params", _W0WA_NONZERO, ids=_W0WA_NONZERO_IDS)
def test_heft_p11_spectral_equiv_vs_euclidemu2(params, plot_diagnostics):
    """Non-linear HEFT P_11 with spectral equivalence matches EuclidEmulator2.

    For w0wa cosmologies, computes the non-linear matter power spectrum P_11 via:
      - gholax HEFT emulator fed equivalent wCDM params from SpectralEquivalence
      - EuclidEmulator2 (natively supports w0wa)
    and compares on the overlapping k range at each test redshift.
    """
    z_test = [0.0, 0.5, 1.0]
    k_heft, p11_heft = _get_heft_p11_with_spectral_equiv(params, z_test)
    k_ee2, pk_ee2 = _get_ee2_pnonlin(params, z_test)

    # Compare on overlapping k range
    k_lo = max(k_heft.min(), k_ee2.min())
    k_hi = min(k_heft.max(), k_ee2.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for z_i in z_test:
        p11_interp = np.interp(k_common, k_heft, p11_heft[z_i])
        ee2_interp = np.interp(k_common, k_ee2, pk_ee2[z_i])

        mask = ee2_interp > 0
        rel_err = np.abs(p11_interp[mask] - ee2_interp[mask]) / ee2_interp[mask]
        median_err = np.median(rel_err)
        assert median_err < _RTOL_HEFT_VS_EE2, (
            f"HEFT P_11 vs EE2 at z={z_i}: median rel error {median_err:.4f} "
            f"> {_RTOL_HEFT_VS_EE2} (max={rel_err.max():.4f})"
        )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_heft_vs_ee2_residuals(
            k_heft, p11_heft, k_ee2, pk_ee2, params, z_test,
        )
        print(f"  Saved HEFT vs EE2 diagnostic plot: {fname}")


@requires_euclidemu2
def test_heft_p11_vs_ee2_wa_zero(plot_diagnostics):
    """With wa=0, HEFT P_11 (no spectral equiv needed) should match EE2 at the same level.

    This is a baseline test establishing that the HEFT and EE2 emulators agree
    for standard wCDM before spectral equivalence is involved.
    """
    params = W0WA_COSMO_PARAMS["wa_zero"]
    z_test = [0.0, 0.5, 1.0]

    k_heft, p11_heft = _get_heft_p11_with_spectral_equiv(params, z_test)
    k_ee2, pk_ee2 = _get_ee2_pnonlin(params, z_test)

    k_lo = max(k_heft.min(), k_ee2.min())
    k_hi = min(k_heft.max(), k_ee2.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for z_i in z_test:
        p11_interp = np.interp(k_common, k_heft, p11_heft[z_i])
        ee2_interp = np.interp(k_common, k_ee2, pk_ee2[z_i])

        mask = ee2_interp > 0
        rel_err = np.abs(p11_interp[mask] - ee2_interp[mask]) / ee2_interp[mask]
        median_err = np.median(rel_err)
        assert median_err < _RTOL_HEFT_VS_EE2, (
            f"HEFT P_11 vs EE2 (wa=0 baseline) at z={z_i}: median rel error "
            f"{median_err:.4f} > {_RTOL_HEFT_VS_EE2} (max={rel_err.max():.4f})"
        )

    if plot_diagnostics:
        _ensure_plot_dir()
        fname = _plot_heft_vs_ee2_residuals(
            k_heft, p11_heft, k_ee2, pk_ee2, params, z_test,
        )
        print(f"  Saved HEFT vs EE2 wa=0 baseline plot: {fname}")


@requires_euclidemu2
def test_heft_p11_spectral_equiv_improves_over_no_correction(plot_diagnostics):
    """Spectral equivalence should reduce the HEFT vs EE2 discrepancy for w0wa.

    Compares the HEFT P_11 residual vs EE2 with and without spectral equivalence
    for a w0wa cosmology. The spectral equivalence version should have smaller
    residuals (or at least not be worse).
    """
    from gholax.theory.emulator import PijEmulator
    from gholax.theory.spectral_equivalence import build_equiv_cparam_grid
    import euclidemu2 as ee2_mod

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

    # EE2 reference
    k_ee2, pk_ee2 = _get_ee2_pnonlin(params, _Z_TEST)

    k_lo = max(k_heft.min(), k_ee2.min())
    k_hi = min(k_heft.max(), k_ee2.max())
    k_common = np.logspace(np.log10(k_lo), np.log10(k_hi), 200)

    for z_i in _Z_TEST:
        ee2_interp = np.interp(k_common, k_ee2, pk_ee2[z_i])
        p11_se_interp = np.interp(k_common, k_heft, p11_with_se[z_i])
        p11_no_se_interp = np.interp(k_common, k_heft, p11_no_se[z_i])

        mask = ee2_interp > 0
        err_with_se = np.median(np.abs(p11_se_interp[mask] - ee2_interp[mask]) / ee2_interp[mask])
        err_no_se = np.median(np.abs(p11_no_se_interp[mask] - ee2_interp[mask]) / ee2_interp[mask])

        assert err_with_se <= err_no_se * 1.1, (
            f"Spectral equiv should not degrade HEFT vs EE2 at z={z_i}: "
            f"with_SE={err_with_se:.4f}, without_SE={err_no_se:.4f}"
        )

    if plot_diagnostics:
        import matplotlib.pyplot as plt

        _ensure_plot_dir()
        label = _cosmo_label(params)
        nz = len(_Z_TEST)
        fig, axes = plt.subplots(1, nz, figsize=(5 * nz, 4), squeeze=False)
        fig.suptitle(f"HEFT P_11 vs EE2: with vs without spectral equiv\n{label}",
                     fontsize=13)

        for col, z_i in enumerate(_Z_TEST):
            ax = axes[0, col]
            ee2_interp = np.interp(k_common, k_ee2, pk_ee2[z_i])
            p11_se_interp = np.interp(k_common, k_heft, p11_with_se[z_i])
            p11_no_se_interp = np.interp(k_common, k_heft, p11_no_se[z_i])

            mask = ee2_interp > 0
            err_se = np.full_like(k_common, np.nan)
            err_no = np.full_like(k_common, np.nan)
            err_se[mask] = (p11_se_interp[mask] - ee2_interp[mask]) / ee2_interp[mask]
            err_no[mask] = (p11_no_se_interp[mask] - ee2_interp[mask]) / ee2_interp[mask]

            ax.semilogx(k_common, err_se, "C0-", lw=1, label="with SE")
            ax.semilogx(k_common, err_no, "C1--", lw=1, label="without SE")
            ax.axhline(0, color="k", ls="--", lw=0.5)
            ax.set_title(f"z = {z_i}")
            ax.set_xlabel(r"$k$ [$h$/Mpc]")
            ax.set_ylabel("(HEFT - EE2) / EE2")
            ax.set_ylim(-0.15, 0.15)
            ax.legend(fontsize=9)

        fig.tight_layout()
        fname = os.path.join(_PLOT_DIR, f"heft_p11_se_improvement_{label}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved SE improvement plot: {fname}")
