"""Bias-convention consistency between RSD and real-space combine functions.

The redshift-space density operator vanishes at mu=0, so the LPT redshift-space
power spectrum at mu=0 must equal the real-space LPT prediction for the same
(b1, b2, bs). We exploit this to detect inconsistent conventions between
``combine_real_space_spectra`` and ``combine_lpt_redshift_space_spectra``.

Reconstruction:
    P(k, mu=0) = P_0(k) - 0.5 P_2(k) + 0.375 P_4(k)
(using P(k,mu) = sum_l P_l(k) L_l(mu); L_2(0) = -1/2, L_4(0) = 3/8.)

Out of scope: b3 / bk2 / alpha_* counterterms, cross-spectra, AP corrections,
fracb1_counterterm path, and the linear-bias-only variant.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from .conftest import COSMO_PARAMS, requires_classy

from gholax.theory.real_space_biased_tracer_spectra import (
    RealSpaceBiasedTracerSpectra,
    combine_real_space_spectra,
)
from gholax.theory.redshift_space_biased_tracer_spectra import (
    RedshiftSpaceBiasedTracerSpectra,
    combine_lpt_redshift_space_spectra,
)


_ZEFF = 0.5
_Z_GRID = np.array([0.4, 0.5, 0.6])  # 3 z-points so jnp.interp is well-behaved
_ZEFF_IDX = 1                         # index of _ZEFF in _Z_GRID

_KIR = 0.2

# Wavenumber grids. Cap at k=0.3 h/Mpc: at higher k, the truncation of the
# Legendre series at l<=4 leaks into P(k, mu=0).
_KMIN, _KMAX, _NK = 1e-3, 0.3, 60
_K_LIN = np.geomspace(1e-4, 5.0, 256)

_BIAS_CASES = [
    pytest.param(1.0, 0.0, 0.0, id="b1_only"),
    pytest.param(1.5, 0.5, 0.0, id="b1_b2"),
    pytest.param(1.5, 0.5, -0.3, id="b1_b2_bs"),
    pytest.param(2.0, -0.4, 0.7, id="b1_b2_bs_alt"),
]

_RTOL = 0.05      # full k range — limited by Legendre truncation at l<=4
_RTOL_LOWK = 0.01 # k < _K_LOWK; truncation error is negligible there
_K_LOWK = 0.15
_PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


def _build_state(cosmo):
    """Run CLASS and assemble the minimal state-dict for both modules."""
    from .conftest import class_instance_sigma8

    params = dict(cosmo)
    params.setdefault("As", 2.1)
    params.setdefault("ns", 0.965)

    boltz = class_instance_sigma8(params)
    h = params["H0"] / 100.0

    nz = len(_Z_GRID)
    Pcb = np.zeros((nz, _K_LIN.size))
    Pm = np.zeros((nz, _K_LIN.size))
    f_z = np.zeros(nz)
    e_z = np.zeros(nz)      # H(z) [Mpc^-1]
    chi_z = np.zeros(nz)    # comoving distance [Mpc]

    for i, z in enumerate(_Z_GRID):
        f_z[i] = boltz.scale_independent_growth_factor_f(float(z))
        e_z[i] = boltz.Hubble(float(z))
        chi_z[i] = boltz.comoving_distance(float(z))
        for j, k in enumerate(_K_LIN):
            Pcb[i, j] = boltz.pk_cb(k * h, float(z)) * h**3
            Pm[i, j] = boltz.pk(k * h, float(z)) * h**3

    boltz.struct_cleanup()
    boltz.empty()

    return {
        "k_lin": jnp.asarray(_K_LIN),
        "Pcb_lin_z": jnp.asarray(Pcb),
        "Pm_lin_z": jnp.asarray(Pm),
        "z_pk": jnp.asarray(_Z_GRID),
        "z_limber": jnp.asarray(_Z_GRID),
        "f_z": jnp.asarray(f_z),
        "e_z_limber": jnp.asarray(e_z),
        "chi_z_limber": jnp.asarray(chi_z),
        # Stored for the test body:
        "_f_at_zeff": float(f_z[_ZEFF_IDX]),
        "_e_at_zeff": float(e_z[_ZEFF_IDX]),
        "_chi_at_zeff": float(chi_z[_ZEFF_IDX]),
    }


def _save_residual_plot(k, p_real, p_rsd_mu0, label):
    import matplotlib.pyplot as plt

    os.makedirs(_PLOT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.loglog(k, np.abs(p_real), label="real-space P(k)")
    ax1.loglog(k, np.abs(p_rsd_mu0), "--", label=r"RSD P(k, $\mu$=0)")
    ax1.legend()
    ax1.set_ylabel(r"$P(k)$  [$h^{-3}\mathrm{Mpc}^3$]")

    rel = (p_rsd_mu0 - p_real) / p_real
    ax2.semilogx(k, rel)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(_RTOL, color="r", ls="--", lw=0.5)
    ax2.axhline(-_RTOL, color="r", ls="--", lw=0.5)
    ax2.set_ylim(-3 * _RTOL, 3 * _RTOL)
    ax2.set_ylabel("rel. resid.")
    ax2.set_xlabel(r"$k$  [$h$/Mpc]")
    fig.suptitle(label)
    fig.tight_layout()
    fname = os.path.join(_PLOT_DIR, f"bias_conv_consistency_{label}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


@requires_classy
@pytest.mark.parametrize("b1, b2, bs", _BIAS_CASES)
def test_mu0_matches_real_space(b1, b2, bs, plot_diagnostics):
    """Reconstructed P_RSD(k, mu=0) must match P_real(k) for the same (b1, b2, bs)."""
    state = _build_state(COSMO_PARAMS["standard"])

    # 1. Populate state['p_ij_real_space_bias_grid'] via the analytic CLEFT path.
    real_module = RealSpaceBiasedTracerSpectra(
        zmin=_Z_GRID[0], zmax=_Z_GRID[-1], nz=len(_Z_GRID),
        kmin=_KMIN, kmax=_KMAX, nk=_NK,
        use_emulator=False, bias_model="cleft", kIR=_KIR,
    )
    real_module.compute_cleft_analytic(state, params_values={})

    # compute_cleft_analytic stores shape (nz, nspec=19, nk); transpose to
    # (nspec, nk, nz) and keep the first 15 cols (the function fills cols 15-18
    # internally with k^2 counterterms).
    p_ij_real = jnp.transpose(state["p_ij_real_space_bias_grid"], (1, 2, 0))
    p_ij_real_zslice = p_ij_real[:15, :, _ZEFF_IDX:_ZEFF_IDX + 1]  # (15, nk, 1)

    # 2. Populate state['p_ij_ell_redshift_space_bias_grid'] via the LPT_RSD path.
    # apar=aperp=1 by setting hz_fid=H(zeff), chiz_fid=chi(zeff).
    rsd_module = RedshiftSpaceBiasedTracerSpectra(
        zeff=np.array([_ZEFF]),
        hz_fid=np.array([state["_e_at_zeff"]]),
        chiz_fid=np.array([state["_chi_at_zeff"]]),
        zmin=_Z_GRID[0], zmax=_Z_GRID[-1], nz=len(_Z_GRID),
        kmin=_KMIN, kmax=_KMAX, nk=_NK,
        use_emulator=False, save_noap_spectra=False, kIR=_KIR,
    )
    rsd_module.compute_analytic(state, params_values={})
    # Stored shape: (nspec, n_ell=3, nk, nzeff=1).
    p_ij_ell = jnp.asarray(state["p_ij_ell_redshift_space_bias_grid"])

    # 3. Apply both combine functions with matched (b1, b2, bs); zero everything else.
    k = jnp.asarray(np.geomspace(_KMIN, _KMAX, _NK))
    p_real = combine_real_space_spectra(
        k,
        p_ij_real_zslice,
        bias_params=(b1, b2, bs, 0.0, 0.0, 0.0),
        cross=False,
        fracb1_counterterm=False,
        s8z=jnp.array([1.0]),
        b1e=True,  # production callers use b1e=True
    )  # -> (nk, 1)

    # combine_lpt_redshift_space_spectra expects spectra shaped (nspec, n_ell, nk).
    # State has (nspec, n_ell, nk, nzeff): slice nzeff.
    spec_rsd = p_ij_ell[..., 0]  # (nspec, n_ell, nk)
    p_ell = combine_lpt_redshift_space_spectra(
        spec_rsd,
        bias_params=(b1, b2, bs, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        fracb1_counterterm=False,
        s8=1.0,
        f=state["_f_at_zeff"],
        b1e=True,
        aap=None,
    )  # -> (n_ell, nk)

    p0 = np.asarray(p_ell[0, :])
    p2 = np.asarray(p_ell[1, :])
    p4 = np.asarray(p_ell[2, :])
    p_rsd_mu0 = p0 - 0.5 * p2 + 0.375 * p4

    p_real_np = np.asarray(p_real[:, 0])

    if plot_diagnostics:
        _save_residual_plot(
            np.asarray(k), p_real_np, p_rsd_mu0,
            label=f"b1_{b1}_b2_{b2}_bs_{bs}",
        )

    err_msg = (
        f"RSD mu=0 reconstruction disagrees with real-space P(k) for "
        f"(b1={b1}, b2={b2}, bs={bs}). The bias monomial wiring or "
        f"basis-spectrum indexing differs between the two combine functions."
    )
    np.testing.assert_allclose(p_rsd_mu0, p_real_np, rtol=_RTOL, atol=0.0,
                               err_msg=err_msg)
    # Tighter check at low k where Legendre truncation is negligible — catches
    # subtle bugs that 5% tolerance would miss.
    lowk = np.asarray(k) < _K_LOWK
    np.testing.assert_allclose(p_rsd_mu0[lowk], p_real_np[lowk],
                               rtol=_RTOL_LOWK, atol=0.0,
                               err_msg=err_msg + f" (low-k k<{_K_LOWK} h/Mpc)")
