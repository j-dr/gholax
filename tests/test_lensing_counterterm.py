"""Tests for LensingCounterterm module.

No external dependencies (CLASS, data files) required.

We insert a stub ``gholax.likelihood`` package into ``sys.modules`` before the
real ``__init__`` runs, so that ``flowjax`` (an optional dep) is never imported.
"""
import os
import sys
import types

# Stub gholax.likelihood as a package (with __path__) so sub-imports work,
# but skip its real __init__ which pulls in flowjax.
if "gholax.likelihood" not in sys.modules:
    _pkg = types.ModuleType("gholax.likelihood")
    import gholax
    _pkg.__path__ = [os.path.join(os.path.dirname(gholax.__file__), "likelihood")]
    _pkg.__package__ = "gholax.likelihood"
    sys.modules["gholax.likelihood"] = _pkg

import jax.numpy as jnp
import numpy as np
import pytest

from gholax.likelihood.projection.limber import required_components
from gholax.likelihood.projection.lensing_counterterm import LensingCounterterm


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockObservedDataVector:
    def __init__(self, n_lens_bins=2, nz=50):
        self.nz_d = np.zeros((n_lens_bins, nz))


DEFAULT_GRID = dict(
    kmin=1e-4, kmax=4.0, nk=100, k_cutoff=0.5,
    zmin_proj=0.0, zmax_proj=3.0, nz_proj=50,
    zmin_pk=0.0, zmax_pk=3.0, nz_pk=30,
    n_ell=20, l_max=3001,
)


def make_lensing_ct(order, integration_method="gl_quad", spectrum_types=None,
                    mean_model="dmo", **extra):
    if spectrum_types is None:
        spectrum_types = ["c_kk"]
    spectrum_info = {
        t: {"bins0": list(range(4)), "bins1": list(range(4))}
        for t in spectrum_types
    }
    kw = {**DEFAULT_GRID, **extra}
    odv = MockObservedDataVector()
    return LensingCounterterm(
        observed_data_vector=odv,
        spectrum_types=spectrum_types,
        spectrum_info=spectrum_info,
        lensing_counterterm_order=order,
        integration_method=integration_method,
        mean_model=mean_model,
        **kw,
    )


def make_state(nk=100, nz_pk=30, nz_proj=50, n_bins=4, n_ell=20):
    k = jnp.logspace(-4, jnp.log10(4.0), nk)
    z_pk = jnp.linspace(0.0, 3.0, nz_pk)
    z_proj = jnp.linspace(0.0, 3.0, nz_proj)
    # Power-law P(k) = k^1.5 at each z, shape (nk, nz_pk)
    p11 = jnp.outer(k**1.5, jnp.ones(nz_pk)) * 1e3
    # sigma8(z) decreasing with z — shape (nz_pk,) broadcasted to (nk, nz_pk) via D_grid[None,:]
    sigma8_z = 0.8 * jnp.exp(-0.3 * z_pk)
    omegam = 0.3
    chi_z = jnp.linspace(0.0, 5000.0, nz_proj)
    chi_inv_eff = 1.0 / (chi_z[1:].mean()) * jnp.ones(n_bins)
    # In the real pipeline, Limber outputs all cross-pairs (n_bins^2) not just unique
    n_spectra = n_bins * n_bins
    c_kk = jnp.ones((n_spectra, n_ell)) * 1e-7
    return {
        "sigma8_z": sigma8_z,
        "p_11_real_space_bias_grid": p11,
        "omegam": omegam,
        "chi_z_limber": chi_z,
        "z_limber": z_proj,
        "chi_inv_eff_w_k": chi_inv_eff,
        "c_kk": c_kk,
    }


def setup_param_indices(module, order):
    """Wire param_indices and return a params_values dict."""
    # "all" indexed_params shape is (order, order)
    param_names = []
    for N in range(2, order + 2):
        for o in range(N - 1):
            param_names.append(f"sigma_{N}_{o}")

    params_values = {name: 1.0 for name in param_names}
    params_values["NA"] = 0.0
    sorted_keys = sorted(params_values.keys())
    params_values = {k: params_values[k] for k in sorted_keys}
    param_list = list(params_values.keys())

    module.param_indices = {}
    # Build indices for "all"
    all_arr = module.indexed_params["all"]
    idx_all = np.zeros_like(all_arr, dtype=int)
    for i in range(all_arr.shape[0]):
        for j in range(all_arr.shape[1]):
            idx_all[i, j] = param_list.index(all_arr[i, j])
    module.param_indices["all"] = idx_all

    # Build indices for any magnification keys
    for k in module.indexed_params:
        if k != "all":
            arr = module.indexed_params[k]
            idx = np.zeros_like(arr, dtype=int)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    idx[i, j] = param_list.index(arr[i, j])
            module.param_indices[k] = idx

    return params_values


# ===========================================================================
# 1. Initialization tests
# ===========================================================================

class TestInit:
    def test_init_order_0(self):
        ct = make_lensing_ct(order=0)
        assert ct.lensing_counterterms == []
        assert "c_kk_w_lensing_ct" in ct.output_requirements

    def test_init_order_1(self):
        ct = make_lensing_ct(order=1)
        assert ct.lensing_counterterms == ["sigma_2_0"]
        assert "sigma_2_0_emu" in ct.output_requirements
        assert "w_k_1" in ct.output_requirements

    def test_init_order_2(self):
        ct = make_lensing_ct(order=2)
        assert ct.lensing_counterterms == ["sigma_2_0", "sigma_3_0", "sigma_3_1"]
        assert "w_k_2" in ct.output_requirements

    def test_init_order_3(self):
        ct = make_lensing_ct(order=3)
        expected = ["sigma_2_0", "sigma_3_0", "sigma_3_1",
                    "sigma_4_0", "sigma_4_1", "sigma_4_2"]
        assert ct.lensing_counterterms == expected
        assert "w_k_3" in ct.output_requirements
        assert ct.indexed_params["all"].shape == (3, 3)

    def test_init_gl_quad_nodes(self):
        ct = make_lensing_ct(order=1, nk_gl=30)
        assert len(ct.k_nodes) == 30
        assert ct.k_nodes.min() >= ct.k_cutoff
        assert ct.k_nodes.max() <= ct.kmax_emu

    def test_init_magnification_indexed_params(self):
        ct = make_lensing_ct(order=1, spectrum_types=["c_dk"])
        assert "w_mag_dk" in ct.indexed_params
        arr = ct.indexed_params["w_mag_dk"]
        # Should have entries for each lens bin
        assert arr.shape[0] > 0


# ===========================================================================
# 2. sigma_N_o tests
# ===========================================================================

class TestSigmaNO:
    def test_sigma_N_o_shape_order3(self):
        ct = make_lensing_ct(order=3)
        state = make_state()
        logk = jnp.log10(ct.k_nodes)
        result = ct.sigma_N_o(state, logk)
        assert result.shape == (3, 3)

    def test_sigma_N_o_positive_sigma_2_0(self):
        ct = make_lensing_ct(order=1)
        state = make_state()
        logk = jnp.log10(ct.k_nodes)
        result = ct.sigma_N_o(state, logk)
        # sigma_2_0 = integral of P(k)*k^{-2} dk > 0 for positive P(k)
        assert float(result[0, 0]) > 0

    def test_sigma_N_o_power_law_analytic(self):
        """For P(k) = A*k^n with D(z)=1, compare sigma_2_0 to analytic integral."""
        n_spec = 1.5
        A = 1e3
        kmin, kmax, k_cutoff = 1e-4, 4.0, 0.5
        nk = 100

        ct = make_lensing_ct(order=1, kmin=kmin, kmax=kmax, nk=nk,
                             k_cutoff=k_cutoff, integration_method="trapezoid")
        k = ct.k
        z_pk = jnp.linspace(0.0, 3.0, 30)
        p11 = jnp.outer(A * k**n_spec, jnp.ones(30))
        sigma8_z = jnp.ones(30)  # D(z) = 1

        state = {
            "sigma8_z": sigma8_z,
            "p_11_real_space_bias_grid": p11,
        }
        logk = jnp.log10(k)
        result = ct.sigma_N_o(state, logk)

        # Analytic: integral from k_cutoff to kmax of A * k^(n-2) dk
        # = A * [k^(n-1)/(n-1)] from k_cutoff to kmax
        analytic = A * (kmax**(n_spec - 1) - k_cutoff**(n_spec - 1)) / (n_spec - 1)
        numerical = float(result[0, 0])
        rel_err = abs(numerical - analytic) / abs(analytic)
        assert rel_err < 0.02, f"rel_err={rel_err:.4f}, numerical={numerical:.4f}, analytic={analytic:.4f}"

    def test_sigma_N_o_gl_quad_vs_trapezoid(self):
        state = make_state()

        ct_gl = make_lensing_ct(order=3, integration_method="gl_quad", nk_gl=80)
        logk_gl = jnp.log10(ct_gl.k_nodes)
        result_gl = ct_gl.sigma_N_o(state, logk_gl)

        ct_trap = make_lensing_ct(order=3, integration_method="trapezoid")
        logk_trap = jnp.log10(ct_trap.k)
        result_trap = ct_trap.sigma_N_o(state, logk_trap)

        # Compare non-zero entries
        mask = np.array(result_gl) != 0
        if mask.any():
            rel = np.abs(np.array(result_gl)[mask] - np.array(result_trap)[mask]) / (
                np.abs(np.array(result_gl)[mask]) + 1e-30
            )
            assert rel.max() < 0.05, f"max rel diff = {rel.max():.4f}"

    def test_sigma_N_o_validity_mask_order3(self):
        ct = make_lensing_ct(order=3)
        state = make_state()
        logk = jnp.log10(ct.k_nodes)
        result = np.array(ct.sigma_N_o(state, logk))
        # Valid entries: o < N-1, where N = row_index + 2
        # Row 0 (N=2): o=0 valid, o=1,2 zero
        # Row 1 (N=3): o=0,1 valid, o=2 zero
        # Row 2 (N=4): o=0,1,2 all valid
        assert result[0, 1] == 0.0
        assert result[0, 2] == 0.0
        assert result[1, 2] == 0.0
        assert result[0, 0] != 0.0
        assert result[1, 0] != 0.0
        assert result[1, 1] != 0.0
        assert result[2, 0] != 0.0


# ===========================================================================
# 3. w_n kernel moment tests
# ===========================================================================

class TestWN:
    def _make_ct_and_state(self, order=3):
        ct = make_lensing_ct(order=order)
        state = make_state(n_bins=4, nz_proj=50)
        return ct, state

    def test_w_n_order1_first_entry_is_ones(self):
        ct, state = self._make_ct_and_state(order=1)
        wn = ct.w_n("w_k", state)
        # First column (n=0) should be ones
        np.testing.assert_allclose(np.array(wn[:, 0]), 1.0)

    def test_w_n_shape_order3(self):
        ct, state = self._make_ct_and_state(order=3)
        wn = ct.w_n("w_k", state)
        # shape: (n_bins, order)
        assert wn.shape == (4, 3)

    def test_w_n_no_nan_order3(self):
        ct, state = self._make_ct_and_state(order=3)
        wn = ct.w_n("w_k", state)
        assert not jnp.any(jnp.isnan(wn))

    def test_w_n_second_entry_order3(self):
        ct, state = self._make_ct_and_state(order=3)
        wn = ct.w_n("w_k", state)
        expected = 1.0 / ct.hubble_radius - state["chi_inv_eff_w_k"]
        np.testing.assert_allclose(np.array(wn[:, 1]), np.array(expected), rtol=1e-5)


# ===========================================================================
# 4. compute pipeline tests
# ===========================================================================

class TestCompute:
    def test_compute_order0_passthrough(self):
        ct = make_lensing_ct(order=0)
        state = make_state(n_ell=ct.n_ell)
        params = {"NA": 0.0}
        ct.param_indices = {}
        result = ct.compute(state, params)
        np.testing.assert_array_equal(
            np.array(result["c_kk_w_lensing_ct"]),
            np.array(state["c_kk"]),
        )

    def test_compute_order3_updates_state_keys(self):
        ct = make_lensing_ct(order=3)
        state = make_state(n_ell=ct.n_ell)
        params = setup_param_indices(ct, order=3)
        result = ct.compute(state, params)
        assert "sigma_N_o_emu" in result
        assert "c_kk_w_lensing_ct" in result

    def test_compute_order3_no_nan_inf(self):
        ct = make_lensing_ct(order=3)
        state = make_state(n_ell=ct.n_ell)
        params = setup_param_indices(ct, order=3)
        result = ct.compute(state, params)
        arr = np.array(result["c_kk_w_lensing_ct"])
        assert not np.any(np.isnan(arr)), "NaN in c_kk_w_lensing_ct"
        assert not np.any(np.isinf(arr)), "Inf in c_kk_w_lensing_ct"
        sigma = np.array(result["sigma_N_o_emu"])
        assert not np.any(np.isnan(sigma)), "NaN in sigma_N_o_emu"

    def test_compute_order3_counterterm_modifies_spectra(self):
        ct = make_lensing_ct(order=3)
        state = make_state(n_ell=ct.n_ell)
        params = setup_param_indices(ct, order=3)
        # Set non-zero sigma params
        for k in params:
            if k.startswith("sigma_"):
                params[k] = 2.0
        result = ct.compute(state, params)
        assert not np.allclose(
            np.array(result["c_kk_w_lensing_ct"]),
            np.array(state["c_kk"]),
        )

    def test_compute_order3_mean_model_zero(self):
        ct_dmo = make_lensing_ct(order=3, mean_model="dmo")
        ct_zero = make_lensing_ct(order=3, mean_model="zero")

        state_dmo = make_state(n_ell=ct_dmo.n_ell)
        state_zero = make_state(n_ell=ct_zero.n_ell)
        # mean_model="zero" reads state["p_mm"][0, ...] instead of p_11_real_space_bias_grid
        state_zero["p_mm"] = state_zero["p_11_real_space_bias_grid"][jnp.newaxis, ...]

        params_dmo = setup_param_indices(ct_dmo, order=3)
        params_zero = setup_param_indices(ct_zero, order=3)

        result_dmo = ct_dmo.compute(state_dmo, params_dmo)
        result_zero = ct_zero.compute(state_zero, params_zero)

        # Both should produce valid output
        arr_dmo = np.array(result_dmo["c_kk_w_lensing_ct"])
        arr_zero = np.array(result_zero["c_kk_w_lensing_ct"])
        assert not np.any(np.isnan(arr_zero))
        # They should differ because mean_model="zero" uses (1+sigma) instead of sigma
        assert not np.allclose(arr_dmo, arr_zero)
