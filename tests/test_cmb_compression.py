"""theta_*, z_*, r_d against CAMB 2.0 (Recfast) references, and the
CompressedCMBLikelihood at the DESI DR2 prior mean."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gholax.theory.cmb_compression import theta_star, z_star, z_drag, sound_horizon_drag
from gholax.theory.bao_alphas import sound_horizon_aizpuru

MNU = 0.06
# (H0, ombh2, omch2, w, wa, thetastar, zstar, rdrag) from CAMB 2.0,
# nnu=3.044, THREE degenerate massive neutrinos sharing 0.06 eV (the
# ExpansionHistory / gholax default), PPF dark energy.
REF = [
    (67.36, 0.02237, 0.12, -1.0, 0.0, 1.04099911e-02, 1089.903, 147.1074),
    (62.0, 0.021, 0.1, -0.7, -1.0, 1.00798898e-02, 1089.822, 154.4091),
    (72.0, 0.0238, 0.135, -1.2, 0.4, 1.05723623e-02, 1089.414, 141.8378),
    (69.3, 0.02237, 0.118, -0.964, -0.293, 1.04090245e-02, 1089.729, 147.6385),
    (71.3, 0.02237, 0.118, -1.18, 0.3, 1.04127804e-02, 1089.729, 147.6385),
]
# same fiducial point with ONE massive species (CAMB / DESI-chain convention)
REF_1NU = (67.36, 0.02237, 0.12, -1.0, 0.0, 1.04119803e-02, 147.1027)


@pytest.mark.parametrize("H0,ombh2,omch2,w,wa,th_ref,zs_ref,rd_ref", REF)
def test_theta_star_matches_camb(H0, ombh2, omch2, w, wa, th_ref, zs_ref, rd_ref):
    """theta_* within 1e-4 of CAMB (Planck sigma is 2.5e-4 relative)."""
    th = float(theta_star(H0, ombh2, omch2, w, wa, MNU))
    assert abs(th / th_ref - 1) < 1e-4
    assert abs(float(z_star(ombh2, omch2, MNU)) - zs_ref) < 0.05
    assert abs(float(sound_horizon_aizpuru(omch2, ombh2, MNU)) / rd_ref - 1) < 3e-4
    # integral r_d (what BAOAlphas uses): 3.4e-5 offset vs CAMB, 3e-7 scatter
    assert abs(float(sound_horizon_drag(H0, ombh2, omch2, w, wa, MNU)) / rd_ref - 1) < 1e-4


def test_one_massive_species_option():
    """n_massive=1 reproduces CAMB's single-species theta_* and r_d; the two
    conventions differ by ~1.9e-4 in theta_* (0.8 Planck sigma)."""
    H0, ombh2, omch2, w, wa, th_ref, rd_ref = REF_1NU
    th1 = float(theta_star(H0, ombh2, omch2, w, wa, MNU, n_massive=1))
    th3 = float(theta_star(H0, ombh2, omch2, w, wa, MNU))
    assert abs(th1 / th_ref - 1) < 1e-4
    assert abs(float(sound_horizon_drag(H0, ombh2, omch2, w, wa, MNU, n_massive=1)) / rd_ref - 1) < 1e-4
    assert 1.5e-4 < th1 / th3 - 1 < 2.5e-4


def test_z_drag_matches_camb():
    """CAMB zdrag at the fiducial point is 1059.93 (Recfast); fit + offset within 0.05."""
    assert abs(float(z_drag(0.02237, 0.12)) - 1059.93) < 0.05


def test_theta_star_gradient_finite():
    g = jax.grad(lambda x: theta_star(x[0], x[1], x[2], x[3], x[4], MNU))(
        jnp.array([67.36, 0.02237, 0.12, -1.0, 0.0])
    )
    assert np.all(np.isfinite(np.array(g)))
    assert g[0] > 0  # theta_* grows with H0 at fixed physical densities


def _cfg(**over):
    params = {
        "H0": {"prior": {"dist": "uniform", "min": 52.0, "max": 82.0}, "ref": 67.36},
        "ombh2": {"prior": {"dist": "uniform", "min": 0.017, "max": 0.027}, "ref": 0.02223},
        "omch2": {"prior": {"dist": "uniform", "min": 0.08, "max": 0.16}, "ref": 0.11985},
        "w": {"prior": {"dist": "uniform", "min": -2.0, "max": 0.0}, "ref": -1.0},
        "wa": {"prior": {"dist": "uniform", "min": -3.0, "max": 2.0}, "ref": 0.0},
        "logmnu": {"value": -1.2218487},
        "mnu": {"derived": "lambda logmnu: 10**logmnu"},
    }
    return {"likelihood": {"CompressedCMBLikelihood": dict(over), "params": params}}


def test_compressed_cmb_likelihood_at_mean():
    from gholax.likelihood import CompressedCMBLikelihood
    from scipy.optimize import brentq

    like = CompressedCMBLikelihood(_cfg())
    assert like.n_massive == 3
    assert CompressedCMBLikelihood(_cfg(n_massive_neutrinos=1)).n_massive == 1
    assert set(like.sampled_params) == {"H0", "ombh2", "omch2", "w", "wa"}
    base = {"ombh2": 0.02223, "omch2": 0.14208 - 0.02223, "w": -1.0, "wa": 0.0}
    f = lambda H0: float(like.observables({**base, "H0": H0})[0]) - 0.01041
    H0 = brentq(f, 60.0, 75.0)
    lp = float(like.compute({**base, "H0": H0}))
    assert abs(lp) < 1e-6
    # one-sigma shift in theta_* costs ~0.5 (marginal, but correlated):
    # just check it decreases and the gradient is finite
    assert float(like.compute({**base, "H0": H0 + 1.0})) < -1.0
    g = jax.grad(lambda x: like.compute({**base, "H0": x}))(jnp.float64(H0) if jax.config.jax_enable_x64 else jnp.float32(H0))
    assert np.isfinite(float(g))
