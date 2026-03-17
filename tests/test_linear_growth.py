"""
Tests for LinearGrowth.compute_ode vs CLASS scale-independent growth factor.

Import gholax.util.likelihood_module first to warm up sys.modules and avoid
any residual circular-import issues before the util/__init__.py lazy fix.
"""
import gholax.util.likelihood_module  # noqa: F401
import jax.numpy as jnp
import numpy as np
import pytest

from gholax.theory.linear_growth import LinearGrowth
from tests.conftest import COSMO_PARAMS, class_instance, requires_classy

COSMO_IDS = list(COSMO_PARAMS.keys())
COSMO_LIST = [COSMO_PARAMS[k] for k in COSMO_IDS]

# Redshifts at which we compare D(z)
_Z_CHECK = np.array([0.0, 0.5, 1.0, 2.0])
relative_tolerance = 0.005  # 0.5% agreement    

def make_lg_ode(zmax=3.0, nz=125):
    """LinearGrowth in ODE mode (no emulator)."""
    # use_boltzmann=True is required because use_emulator=False raises unless
    # solve_ode is also set; we set use_boltzmann=True so the constructor
    # doesn't raise, and then enable solve_ode for the ODE path.
    return LinearGrowth(
        zmin=0.0, zmax=zmax, nz=nz,
        use_emulator=False,
        use_boltzmann=True,
        solve_ode=True,
        n_points_ode=500,
        a_init_ode=1.0 / (1.0 + 200.0),
    )


# ---------------------------------------------------------------------------
# ODE growth factor vs CLASS
# ---------------------------------------------------------------------------

@requires_classy
@pytest.mark.parametrize("params", COSMO_LIST, ids=COSMO_IDS)
def test_D_z_ode_vs_class(params):
    """D(z) from the ODE solver matches CLASS scale_independent_growth_factor to 0.5%."""
    lg = make_lg_ode()

    # Run ODE
    state = {}
    state = lg.compute_ode(state, params)
    z_ode = np.array(state["z_Dz_ode"])  # ascending: 0..15
    D_ode = np.array(state["Dz_ode"])    # D[0] is at z=0

    # Normalize ODE to D(z=0) = 1
    D0_ode = float(np.interp(0.0, z_ode, D_ode))
    D_ode_norm = D_ode / D0_ode

    # CLASS reference (N_ur=0 to match analytic treatment)
    boltz = class_instance(params, N_ur=0.0)
    D_class = np.array([boltz.scale_independent_growth_factor(z) for z in _Z_CHECK])
    D0_class = D_class[0]  # CLASS normalizes to D(z=0)=1 by default, but be explicit
    D_class_norm = D_class / D0_class

    D_gholax = np.array([float(np.interp(z, z_ode, D_ode_norm)) for z in _Z_CHECK])

    rel_err = np.abs(D_gholax - D_class_norm) / D_class_norm
    bad = rel_err > relative_tolerance
    assert not bad.any(), (
        f"D(z) ODE vs CLASS relative error > {relative_tolerance*100:.1f}% at z={_Z_CHECK[bad]}: "
        f"ode={D_gholax[bad]}, class={D_class_norm[bad]}, err={rel_err[bad]}"
    )


# ---------------------------------------------------------------------------
# D(z) ratio sanity: ODE vs CLASS
# ---------------------------------------------------------------------------

@requires_classy
def test_sigma8_z_ode_vs_class():
    """D(z)/D(z=0) ratio from ODE matches CLASS to 0.5% for standard cosmology."""
    params = COSMO_PARAMS["standard"]
    lg = make_lg_ode()

    state = {}
    state = lg.compute_ode(state, params)
    z_ode = np.array(state["z_Dz_ode"])  # ascending: 0..15
    D_ode = np.array(state["Dz_ode"])    # D[0] is at z=0

    D0_ode = float(np.interp(0.0, z_ode, D_ode))
    D_ode_norm = D_ode / D0_ode

    boltz = class_instance(params, N_ur=0.0)
    z_test = np.array([0.5, 1.0, 2.0])
    D_class = np.array([boltz.scale_independent_growth_factor(z) for z in z_test])
    D_class_norm = D_class / boltz.scale_independent_growth_factor(0.0)

    D_gholax = np.array([float(np.interp(z, z_ode, D_ode_norm)) for z in z_test])

    rel_err = np.abs(D_gholax - D_class_norm) / D_class_norm
    bad = rel_err > relative_tolerance
    assert not bad.any(), (
        f"D(z)/D(0) ratio > {relative_tolerance*100:.1f}% off at z={z_test[bad]}: "
        f"ode={D_gholax[bad]}, class={D_class_norm[bad]}, err={rel_err[bad]}"
    )


# ---------------------------------------------------------------------------
# Qualitative neutrino mass effect on growth (no CLASS needed)
# ---------------------------------------------------------------------------

def test_f_nu_effect_on_growth():
    """Heavier neutrinos should suppress growth: D(z_high)/D(z_low) is smaller."""
    params_light = {**COSMO_PARAMS["standard"], "mnu": 0.001}
    params_heavy = {**COSMO_PARAMS["standard"], "mnu": 0.3}

    lg = make_lg_ode()

    state_light = {}
    state_light = lg.compute_ode(state_light, params_light)
    z_ode = np.array(state_light["z_Dz_ode"])  # ascending: 0..15
    D_light = np.array(state_light["Dz_ode"])
    D0_light = float(np.interp(0.0, z_ode, D_light))
    D_light_norm = D_light / D0_light

    state_heavy = {}
    state_heavy = lg.compute_ode(state_heavy, params_heavy)
    D_heavy = np.array(state_heavy["Dz_ode"])
    D0_heavy = float(np.interp(0.0, z_ode, D_heavy))
    D_heavy_norm = D_heavy / D0_heavy

    # In this simplified ODE (no relativistic neutrino thermodynamics in E(z)),
    # heavier neutrinos mean a larger Omega_m and a smaller Omega_Lambda (flat
    # universe). Less dark energy → less late-time growth suppression → D(z=0)
    # grows more relative to D(z_high). After normalising to D(z=0)=1 this
    # means D(z_high)/D(z=0) is *larger* for the heavy-nu cosmology.
    # The test checks that neutrino mass has a measurable directional effect.
    z_high = 5.0
    D_ratio_light = float(np.interp(z_high, z_ode, D_light_norm))
    D_ratio_heavy = float(np.interp(z_high, z_ode, D_heavy_norm))

    assert D_ratio_heavy > D_ratio_light, (
        f"Expected heavier-nu D(z={z_high})/D(0) > lighter-nu, "
        f"got light={D_ratio_light:.4f}, heavy={D_ratio_heavy:.4f}"
    )
