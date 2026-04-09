"""
Tests for the w0wa growth factor correction in LinearGrowth.compute_w0wa_emulator.

The approximation is:
    sigma8_w0wa(z) = sigma8_wCDM_emu(z; w=w0) * [D_w0wa(z, unnorm) / D_wCDM(z, unnorm)]

Tests:
  1. wa=0 regression: ratio=1 everywhere, so result matches standard emulator path.
  2. Accuracy vs CLASS: sigma8(z) from the approximation matches CLASS sigma8(z)
     (computed directly with wa_fld != 0) to within tolerance.
  3. Directional effect: wa<0 reduces sigma8(z=0) relative to wa=0 (less late-time growth).
"""
import gholax.util.likelihood_module  # noqa: F401

import jax.numpy as jnp
import numpy as np
import pytest

from gholax.theory.linear_growth import LinearGrowth
from tests.conftest import W0WA_COSMO_PARAMS, class_instance_sigma8, requires_classy

# Base filename for the sigma8(z) emulator (no path, no .h5 — emulator prepends emu_weights/)
_EMU_FILE = "aemulus_nu_tier1_cosmo_only_heft_sigma8z_emu"

_Z_CHECK = np.array([0.0, 0.3, 0.5, 1.0, 2.0])

# 0.5% tolerance for the w0wa approximation vs CLASS
_SIGMA8_TOLERANCE = 0.005


def make_lg_w0wa(zmax=2.5, nz=125):
    return LinearGrowth(
        zmin=0.0, zmax=zmax, nz=nz,
        use_emulator=True,
        compute_w0wa=True,
        emulator_file_name=_EMU_FILE,
        n_points_ode=500,
        a_init_ode=1.0 / (1.0 + 200.0),
    )


def make_lg_wcdm(zmax=2.5, nz=125):
    return LinearGrowth(
        zmin=0.0, zmax=zmax, nz=nz,
        use_emulator=True,
        compute_w0wa=False,
        emulator_file_name=_EMU_FILE,
    )


# ---------------------------------------------------------------------------
# Regression: wa=0 gives identical result to the standard emulator path
# ---------------------------------------------------------------------------

def test_wa_zero_regression():
    """With wa=0, the w0wa path must reproduce the standard emulator exactly."""
    params = W0WA_COSMO_PARAMS["wa_zero"]

    lg_w0wa = make_lg_w0wa()
    lg_wcdm = make_lg_wcdm()

    state_w0wa = lg_w0wa.compute({}, params)
    state_wcdm = lg_wcdm.compute({}, params)

    sigma8_w0wa = np.array(state_w0wa["sigma8_z"])
    sigma8_wcdm = np.array(state_wcdm["sigma8_z"])

    rel_err = np.abs(sigma8_w0wa - sigma8_wcdm) / sigma8_wcdm
    assert rel_err.max() < 1e-5, (
        f"wa=0 regression failed: max relative error = {rel_err.max():.2e}"
    )


# ---------------------------------------------------------------------------
# Accuracy vs CLASS sigma8(z)
# ---------------------------------------------------------------------------

W0WA_IDS = list(W0WA_COSMO_PARAMS.keys())
W0WA_LIST = [W0WA_COSMO_PARAMS[k] for k in W0WA_IDS]


@requires_classy
@pytest.mark.parametrize("params", W0WA_LIST, ids=W0WA_IDS)
def test_w0wa_sigma8_vs_class(params):
    """sigma8(z) from the w0wa approximation matches CLASS to within 2%."""
    lg = make_lg_w0wa()
    state = lg.compute({}, params)
    sigma8_approx = np.array(state["sigma8_z"])
    z_lg = np.array(state["z_D"])

    # Direct CLASS calculation with wa_fld != 0
    boltz = class_instance_sigma8(params, N_ur=0.0)
    sigma8_class = np.array([boltz.sigma(8, float(z), h_units=True) for z in _Z_CHECK])

    sigma8_approx_at_z = np.interp(_Z_CHECK, z_lg, sigma8_approx)

    rel_err = np.abs(sigma8_approx_at_z - sigma8_class) / sigma8_class
    bad = rel_err > _SIGMA8_TOLERANCE
    assert not bad.any(), (
        f"sigma8(z) w0wa approximation vs CLASS > {_SIGMA8_TOLERANCE*100:.0f}% "
        f"at z={_Z_CHECK[bad]}: "
        f"approx={sigma8_approx_at_z[bad]}, class={sigma8_class[bad]}, "
        f"rel_err={rel_err[bad]}"
    )


# ---------------------------------------------------------------------------
# Directional effect of wa on sigma8(z=0)
# ---------------------------------------------------------------------------

@requires_classy
def test_wa_directional_effect_vs_class():
    """
    wa<0 means dark energy was weaker in the past → more growth → larger sigma8(z=0).
    wa>0 means dark energy was stronger in the past → less growth → smaller sigma8(z=0).
    Both relative to wa=0 at the same w0.
    Verify the sign of the effect matches CLASS.
    """
    lg = make_lg_w0wa()

    params_base = W0WA_COSMO_PARAMS["wa_zero"].copy()
    params_base["w"] = -0.9  # use same w0 as wa_neg / wa_pos

    params_neg = W0WA_COSMO_PARAMS["wa_neg"]   # w=-0.9, wa=-0.5
    params_pos = W0WA_COSMO_PARAMS["wa_pos"]   # w=-0.9, wa=+0.5

    def sigma8_z0_approx(p):
        state = lg.compute({}, p)
        return float(np.array(state["sigma8_z"])[0])

    def sigma8_z0_class(p):
        boltz = class_instance_sigma8(p, N_ur=0.0)
        return boltz.sigma(8, 0.0, h_units=True)

    s8_neg_approx = sigma8_z0_approx(params_neg)
    s8_pos_approx = sigma8_z0_approx(params_pos)
    s8_neg_class = sigma8_z0_class(params_neg)
    s8_pos_class = sigma8_z0_class(params_pos)

    # Both approximation and CLASS should agree on which direction wa shifts sigma8
    approx_sign = np.sign(s8_neg_approx - s8_pos_approx)
    class_sign = np.sign(s8_neg_class - s8_pos_class)
    assert approx_sign == class_sign, (
        f"Directional effect of wa disagrees with CLASS: "
        f"approx wa_neg={s8_neg_approx:.4f}, wa_pos={s8_pos_approx:.4f}; "
        f"class wa_neg={s8_neg_class:.4f}, wa_pos={s8_pos_class:.4f}"
    )
