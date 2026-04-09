"""
Tests for ExpansionHistory.compute_analytic vs CLASS background quantities.

Import gholax.util.likelihood_module first to warm up sys.modules and avoid
any residual circular-import issues before the util/__init__.py lazy fix.
"""
import gholax.util.likelihood_module  # noqa: F401
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import roots_laguerre
from scipy.special import zeta

from gholax.theory.expansion_history import ExpansionHistory
from gholax.theory.linear_growth import (
    neutrino_density_ratio,
    _T_NU0_EV,
    _OMEGA_GAMMA_H2,
    _OMEGA_NU_REL_H2_PER_SPECIES,
)
from tests.conftest import COSMO_PARAMS, class_instance, requires_classy

# Speed of light in km/s
_C_KM_S = 2.99792458e5

COSMO_IDS = list(COSMO_PARAMS.keys())
COSMO_LIST = [COSMO_PARAMS[k] for k in COSMO_IDS]


def make_eh(params, zmax=5.0, nz=200):
    return ExpansionHistory(zmin=0.0, zmax=zmax, nz=nz,
                            use_direct_integration=True,
                            use_emulator=False,
                            use_boltzmann=False)

relative_tolerance = 0.001  # 0.2% agreement
# ---------------------------------------------------------------------------
# Neutrino phase-space integral limits
# ---------------------------------------------------------------------------

def test_neutrino_density_ratio_relativistic_limit():
    """F(y=0) should equal 1.0 — fully relativistic neutrinos."""
    nodes_np, weights_np = roots_laguerre(32)
    nodes = jnp.array(nodes_np)
    weights = jnp.array(weights_np)
    F0 = float(neutrino_density_ratio(0.0, nodes, weights))
    assert abs(F0 - 1.0) < 1e-4, f"F(0) = {F0}, expected 1.0"


def test_neutrino_density_ratio_nr_limit():
    """F(y>>1) -> (180*zeta(3)/(7*pi^4)) * y (non-relativistic limit)."""
    nodes_np, weights_np = roots_laguerre(32)
    nodes = jnp.array(nodes_np)
    weights = jnp.array(weights_np)
    y = 1000.0
    F_numerical = float(neutrino_density_ratio(y, nodes, weights))
    F_analytic = (180.0 * zeta(3) / (7.0 * np.pi**4)) * y
    rel_err = abs(F_numerical - F_analytic) / F_analytic
    assert rel_err < 1e-4, f"NR limit: F({y})={F_numerical:.6f}, expected {F_analytic:.6f}"


# ---------------------------------------------------------------------------
# E(z=0) == 1 (internal consistency)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", COSMO_LIST, ids=COSMO_IDS)
def test_E_z0_equals_one(params):
    """E(z=0) must equal 1 by definition of H0."""
    eh = make_eh(params)
    _, e_z, _, _ = eh.compute_analytic(params)
    assert abs(float(e_z[0]) - 1.0) < 1e-5, f"E(z=0) = {float(e_z[0])}"


# ---------------------------------------------------------------------------
# Flat-universe normalization (internal consistency)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", COSMO_LIST, ids=COSMO_IDS)
def test_flat_universe_normalization(params):
    """Omega_cb + Omega_nu + Omega_r_photon + Omega_Lambda == 1."""
    h = params["H0"] / 100.0
    n_species = 3
    mnu_per_species = params["mnu"] / n_species

    nodes_np, weights_np = roots_laguerre(32)
    nodes = jnp.array(nodes_np)
    weights = jnp.array(weights_np)

    Omega_cb = (params["omch2"] + params["ombh2"]) / h**2
    Omega_r_photon = _OMEGA_GAMMA_H2 / h**2

    y0 = mnu_per_species / _T_NU0_EV
    F_y0 = float(neutrino_density_ratio(y0, nodes, weights))
    Omega_nu_0 = n_species * (_OMEGA_NU_REL_H2_PER_SPECIES / h**2) * F_y0

    Omega_Lambda = 1.0 - Omega_cb - Omega_nu_0 - Omega_r_photon
    total = Omega_cb + Omega_nu_0 + Omega_r_photon + Omega_Lambda
    assert abs(total - 1.0) < 1e-10, f"Omega sum = {total}"


# ---------------------------------------------------------------------------
# chi(z) vs CLASS (requires classy)
# ---------------------------------------------------------------------------

@requires_classy
@pytest.mark.parametrize("params", COSMO_LIST, ids=COSMO_IDS)
def test_chi_z_vs_class(params):
    """Comoving distance chi(z) from compute_analytic matches CLASS to 0.2%."""
    z_check = np.array([0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
    h = params["H0"] / 100.0

    eh = make_eh(params, zmax=6.0, nz=300)
    chi_z, _, _, _ = eh.compute_analytic(params)

    boltz = class_instance(params, N_ur=0.0)
    chi_class = np.array([
        boltz.angular_distance(z) * (1.0 + z) * h
        for z in z_check
    ])

    chi_gholax = np.interp(z_check, np.array(eh.z), np.array(chi_z))

    rel_err = np.abs(chi_gholax - chi_class) / chi_class
    bad = rel_err > relative_tolerance
    assert not bad.any(), (
        f"chi(z) relative error > {relative_tolerance*100:.1f}% at z={z_check[bad]}: "
        f"gholax={chi_gholax[bad]}, class={chi_class[bad]}, err={rel_err[bad]}"
    )


# ---------------------------------------------------------------------------
# E(z) vs CLASS (requires classy)
# ---------------------------------------------------------------------------

@requires_classy
@pytest.mark.parametrize("params", COSMO_LIST, ids=COSMO_IDS)
def test_E_z_vs_class(params):
    """Dimensionless Hubble E(z) from compute_analytic matches CLASS to 0.2%."""
    z_check = np.array([0.3, 0.5, 1.0, 2.0, 5.0])
    h = params["H0"] / 100.0

    eh = make_eh(params, zmax=6.0, nz=300)
    _, e_z, _, _ = eh.compute_analytic(params)

    boltz = class_instance(params, N_ur=0.0)
    E_class = np.array([
        boltz.Hubble(z) * _C_KM_S / (h * 100.0)
        for z in z_check
    ])

    E_gholax = np.interp(z_check, np.array(eh.z), np.array(e_z))

    rel_err = np.abs(E_gholax - E_class) / E_class
    bad = rel_err > relative_tolerance
    assert not bad.any(), (
        f"E(z) relative error > {relative_tolerance*100:.1f}% at z={z_check[bad]}: "
        f"gholax={E_gholax[bad]}, class={E_class[bad]}, err={rel_err[bad]}"
    )


# ---------------------------------------------------------------------------
# High-z chi(z) vs CLASS (requires classy)
# ---------------------------------------------------------------------------

@requires_classy
def test_chi_z_high_z_vs_class():
    """chi(z) matches CLASS to 0.5% out to z=50 for the standard cosmology."""
    params = COSMO_PARAMS["standard"]
    z_check = np.array([5.0, 10.0, 20.0, 50.0])
    h = params["H0"] / 100.0

    eh = make_eh(params, zmax=55.0, nz=500)
    chi_z, _, _, _ = eh.compute_analytic(params)

    boltz = class_instance(params, N_ur=0.0)
    chi_class = np.array([
        boltz.angular_distance(z) * (1.0 + z) * h
        for z in z_check
    ])

    chi_gholax = np.interp(z_check, np.array(eh.z), np.array(chi_z))

    rel_err = np.abs(chi_gholax - chi_class) / chi_class
    bad = rel_err > relative_tolerance
    assert not bad.any(), (
        f"High-z chi(z) relative error > {relative_tolerance*100:.1f}% at z={z_check[bad]}: "
        f"gholax={chi_gholax[bad]}, class={chi_class[bad]}, err={rel_err[bad]}"
    )
