"""Lock the RK4/Newton-hoist SpectralEquivalence refactor to verified outputs.

Golden values recorded 2026-08-14 (CPU fp32) from the refactored module after
verifying agreement with the previous odeint/AD-quadrature implementation to
<=1.4e-5 relative (w_equiv) and <=1.7e-6 (D) at five (w0, wa) corners.
"""

import jax.numpy as jnp
import jax
import numpy as np

from gholax.theory.spectral_equivalence import SpectralEquivalence

REF = dict(As=2.1, ns=0.9649, H0=67.36, ombh2=0.02237, omch2=0.12, mnu=0.06)
IDX = [0, 25, 49]

GOLDEN = {
    (-1.0, 0.0): {
        "w_equiv": [-1.0000322, -1.0001239, -1.0003841],
        "D": [0.7780801, 0.38198754, 0.24615122],
    },
    (-0.7, -1.0): {
        "w_equiv": [-0.9612722, -1.117, -1.2057815],
        "D": [0.77589536, 0.38530475, 0.24713993],
    },
}


def test_spectral_equivalence_golden():
    z_pk = jnp.linspace(0.0001, 3.0, 50)
    se = SpectralEquivalence(z=z_pk)
    for (w0, wa), gold in GOLDEN.items():
        p = dict(REF, w=w0, wa=wa)
        state = se.compute({}, p)
        np.testing.assert_allclose(
            np.array(state["w_equiv_z"])[IDX], gold["w_equiv"], rtol=1e-4,
            err_msg=f"w_equiv at (w0, wa)=({w0}, {wa})",
        )

        h = p["H0"] / 100.0
        from gholax.theory.linear_growth import (
            neutrino_density_ratio, _T_NU0_EV, _OMEGA_GAMMA_H2,
            _OMEGA_NU_REL_H2_PER_SPECIES)
        mnu_ps = p["mnu"] / 3
        Omega_cb = (p["omch2"] + p["ombh2"]) / h**2
        Omega_r = _OMEGA_GAMMA_H2 / h**2
        F_y0 = neutrino_density_ratio(
            mnu_ps / _T_NU0_EV, se.gl_nu_nodes, se.gl_nu_weights)
        Omega_nu0 = 3 * (_OMEGA_NU_REL_H2_PER_SPECIES / h**2) * F_y0
        Omega_m = Omega_cb + Omega_nu0
        Omega_L = 1.0 - Omega_cb - Omega_nu0 - Omega_r
        f_nu = Omega_nu0 / Omega_m
        D = se._compute_D_unnorm(Omega_m, Omega_L, f_nu, w0, wa, se.z)
        np.testing.assert_allclose(
            np.array(D)[IDX], gold["D"], rtol=1e-5,
            err_msg=f"D(z) at (w0, wa)=({w0}, {wa})",
        )


def test_w_equiv_finite_and_penalized_across_prior_box():
    """Unguarded Newton returned NaN for w0+wa >~ -0.3 (24% DES divergences);
    iterates are now clamped and out-of-box w_equiv is smoothly penalized."""
    z_pk = jnp.linspace(0.0001, 3.0, 20)
    se = SpectralEquivalence(z=z_pk)

    def wsum(w0, wa):
        return jnp.sum(se.compute({}, dict(REF, w=w0, wa=wa))["w_equiv_z"])

    for w0 in (-1.9, -1.1, -0.7, -0.3):
        for wa in (-2.9, -1.0, 0.0, 0.8, 1.1, 1.9):
            st = se.compute({}, dict(REF, w=w0, wa=wa))
            g = jax.grad(wsum, argnums=(0, 1))(jnp.asarray(w0), jnp.asarray(wa))
            assert np.isfinite(np.asarray(st["w_equiv_z"])).all(), (w0, wa)
            assert np.isfinite(np.asarray(st["log_penalty"])) and st["log_penalty"] <= 0
            assert all(np.isfinite(np.asarray(v)) for v in g), (w0, wa)
    inside = se.compute({}, dict(REF, w=-1.0, wa=0.0))
    assert float(inside["log_penalty"]) == 0.0
    out = se.compute({}, dict(REF, w=-0.7, wa=1.1))
    assert float(out["log_penalty"]) < -1.0
    assert np.all(np.asarray(out["w_equiv_z"]) <= 0.0)
