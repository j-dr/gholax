"""
Isolated test for AngularPowerSpectrum_to_CorrelationFunction.
Tests the C_ell -> correlation function transform without needing
a real HDF5 file or full pipeline.

Validates against pyccl using bin-averaged reference values computed
from realistic weak lensing and galaxy clustering C_ells.
All four transforms (w, gamma_t, xi+, xi-) should agree to within 10%.

Run from repo root with:
    python tests/test_configspace_transform.py

Requires pyccl for the reference comparison (optional but recommended).
"""
import sys
import os
from unittest.mock import MagicMock
from jax import config
config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---- Mock heavy dependencies before any gholax imports ----
sys.modules['h5py']                  = MagicMock()
sys.modules['jax.scipy.integrate']   = MagicMock()
sys.modules['jax.scipy.interpolate'] = MagicMock()
sys.modules['mpi4py']                = MagicMock()
sys.modules['spinosaurus']           = MagicMock()
sys.modules['spinosaurus.cleft_fftw']= MagicMock()
sys.modules['spinosaurus.density_shape_correlators_fftw'] = MagicMock()
sys.modules['spinosaurus.shape_shape_correlators_fftw']   = MagicMock()

import numpy as np
import jax.numpy as jnp


# ---- Mock objects to replace TwoPointSpectrum ----

class MockDataVector:
    """Mimics the attributes of TwoPointSpectrum that the module uses."""
    def __init__(self, spectrum_types, spectrum_info):
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info


def make_mock_spectrum_info(spectrum_types, n_bins, n_theta):
    theta_centers = np.deg2rad(np.logspace(np.log10(10/60), np.log10(300/60), n_theta))
    spectrum_info = {}
    for t in spectrum_types:
        bins = np.arange(n_bins)
        spectrum_info[t] = {
            "bins0": bins,
            "bins1": bins,
            "use_cross": True,
            "separation": theta_centers,
            "n_bins0_tot": n_bins,
            "n_bins1_tot": n_bins,
        }
    return spectrum_info


# ---- Import module under test ----
try:
    from gholax.likelihood.window.AngularPowerSpectrum_to_CorrelationFunction import (
        AngularPowerSpectrum_to_CorrelationFunction
    )
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


# ---- Test parameters ----
N_BINS  = 2
N_THETA = 10
N_ELL   = 500
L_MAX   = 3000

spectrum_types = ["w_theta", "gamma_t", "xi_plus", "xi_minus"]
spectrum_info  = make_mock_spectrum_info(spectrum_types, N_BINS, N_THETA)
dv             = MockDataVector(spectrum_types, spectrum_info)


# ---- Instantiate module ----
print("\n=== Instantiating module ===")
try:
    module = AngularPowerSpectrum_to_CorrelationFunction(
        observed_data_vector=dv,
        spectrum_types=spectrum_types,
        spectrum_info=spectrum_info,
        n_ell=N_ELL,
        l_max=L_MAX,
        cl_tag="_mbias",
    )
    print("Module instantiated successfully.")
except Exception as e:
    print(f"Instantiation failed: {e}")
    raise


# ---- Check output_requirements ----
print("\n=== Checking output_requirements ===")
print(f"Number of output requirements: {len(module.output_requirements)}")
for k, v in module.output_requirements.items():
    print(f"  {k} <- {v}")


# ---- Check all_spectra ----
print("\n=== Checking all_spectra ===")
for t in spectrum_types:
    print(f"  {t}: {module.all_spectra[t]}")


# ---- Build C_ells from pyccl ----
# Use realistic C_ells from pyccl for all transforms.
# Fallback to beam-suppressed power laws if pyccl is not available.
print("\n=== Building C_ells ===")
ell_module = np.array(module.ell)

try:
    import pyccl

    cosmo = pyccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96,
        transfer_function="bbks",
    )

    z = np.linspace(0, 3, 100)

    # Source n(z): Gaussian at z=0.7
    nz_source = np.exp(-0.5*((z - 0.7)/0.1)**2)
    source_tracer = pyccl.WeakLensingTracer(cosmo, dndz=(z, nz_source))

    # Lens n(z): Gaussian at z=0.5, linear bias=1
    nz_lens = np.exp(-0.5*((z - 0.5)/0.1)**2)
    lens_tracer = pyccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz_lens), bias=(z, np.ones_like(z))
    )

    C_ell_dd = pyccl.angular_cl(cosmo, lens_tracer,   lens_tracer,   ell_module)
    C_ell_dk = pyccl.angular_cl(cosmo, lens_tracer,   source_tracer, ell_module)
    C_ell_kk = pyccl.angular_cl(cosmo, source_tracer, source_tracer, ell_module)

    print("Using realistic C_ells from pyccl.")
    print(f"  C_dd range: {C_ell_dd.min():.3e} to {C_ell_dd.max():.3e}")
    print(f"  C_dk range: {C_ell_dk.min():.3e} to {C_ell_dk.max():.3e}")
    print(f"  C_kk range: {C_ell_kk.min():.3e} to {C_ell_kk.max():.3e}")
    pyccl_available = True

except ImportError:
    print("pyccl not available — using beam-suppressed power-law C_ells.")
    beam = 1.0 / (1.0 + (ell_module / 1500.0)**4)
    C_ell_dd = 1e-5 * (ell_module / 100) ** -2 * beam
    C_ell_dk = 5e-6 * (ell_module / 100) ** -2 * beam
    C_ell_kk = 1e-6 * (ell_module / 100) ** -2 * beam
    pyccl_available = False


# ---- Build mock state ----
print("\n=== Building mock state ===")
n_bins = N_BINS
state = {}
state["c_dd_mbias"] = jnp.stack([jnp.array(C_ell_dd)] * (n_bins * n_bins))
state["c_dk_mbias"] = jnp.stack([jnp.array(C_ell_dk)] * (n_bins * n_bins))
state["c_kk_mbias"] = jnp.stack([jnp.array(C_ell_kk)] * (n_bins * n_bins))
print(f"State keys: {list(state.keys())}")


# ---- Run compute() ----
print("\n=== Running compute() ===")
try:
    state = module.compute(state, params_values={})
    print("compute() ran successfully.")
except Exception as e:
    print(f"compute() failed: {e}")
    raise


# ---- Check outputs ----
print("\n=== Checking compute() outputs ===")
expected_outputs = list(module.output_requirements.keys())
all_ok = True
for k in expected_outputs:
    if k in state:
        arr = state[k]
        finite = jnp.all(jnp.isfinite(arr))
        print(f"  {k}: shape={arr.shape}, finite={finite}")
        if not finite:
            print(f"    WARNING: non-finite values detected!")
            all_ok = False
    else:
        print(f"  {k}: MISSING from state!")
        all_ok = False

if all_ok:
    print("\nAll expected outputs present and finite.")
else:
    print("\nSome outputs missing or non-finite — check above.")


# ---- Run get_model_from_state() ----
print("\n=== Running get_model_from_state() ===")
try:
    model = module.get_model_from_state(state)
    print(f"Model vector shape: {model.shape}")
    print(f"Model finite: {jnp.all(jnp.isfinite(model))}")
    print(f"Model values (first 5): {model[:5]}")
except Exception as e:
    print(f"get_model_from_state() failed: {e}")
    raise


# ---- pyccl comparison (authoritative reference) ----
print("\n=== pyccl comparison (optional) ===")
if not pyccl_available:
    print("pyccl not installed, skipping. Install with: pip install pyccl")
else:
    ell_dense  = np.arange(2, L_MAX + 1)
    C_dd_dense = np.interp(ell_dense, ell_module, np.array(C_ell_dd))
    C_dk_dense = np.interp(ell_dense, ell_module, np.array(C_ell_dk))
    C_kk_dense = np.interp(ell_dense, ell_module, np.array(C_ell_kk))

    theta_centers = spectrum_info["w_theta"]["separation"]
    half_gaps = np.diff(theta_centers) / 2
    theta_edges = np.concatenate([
        [theta_centers[0] - half_gaps[0]],
        theta_centers[:-1] + half_gaps,
        [theta_centers[-1] + half_gaps[-1]]
    ])

    def pyccl_bin_average(cosmo, ell, C_ell, theta_edges_rad, corr_type, n_points=200):
        """Bin-average pyccl by dense sampling within each bin.
        Matches the area-weighted bin averaging in AngularPowerSpectrum_to_CorrelationFunction."""
        n_bins = len(theta_edges_rad) - 1
        result = np.zeros(n_bins)
        for i in range(n_bins):
            theta_dense = np.logspace(
                np.log10(np.rad2deg(theta_edges_rad[i])),
                np.log10(np.rad2deg(theta_edges_rad[i+1])),
                n_points
            )
            xi_dense = pyccl.correlation(
                cosmo, ell=ell, C_ell=C_ell,
                theta=theta_dense, type=corr_type, method='fftlog'
            )
            theta_dense_rad = np.deg2rad(theta_dense)
            weights = np.sin(theta_dense_rad)
            result[i] = np.trapezoid(xi_dense * weights, theta_dense_rad) / \
                        np.trapezoid(weights, theta_dense_rad)
        return result

    print("Computing bin-averaged pyccl reference (may take ~30s)...")
    w_pyccl      = pyccl_bin_average(cosmo, ell_dense, C_dd_dense, theta_edges, "NN")
    gamma_pyccl  = pyccl_bin_average(cosmo, ell_dense, C_dk_dense, theta_edges, "NG")
    xi_pos_pyccl = pyccl_bin_average(cosmo, ell_dense, C_kk_dense, theta_edges, "GG+")
    xi_neg_pyccl = pyccl_bin_average(cosmo, ell_dense, C_kk_dense, theta_edges, "GG-")

    w_ours      = np.array(state["w_0_0_obs"])
    gamma_ours  = np.array(state["gamma_0_0_obs"])
    xi_pos_ours = np.array(state["xi_pos_0_0_obs"])
    xi_neg_ours = np.array(state["xi_neg_0_0_obs"])

    def compare(name, ours, ref, rtol=0.10, atol_fraction=0.01):
        """atol_fraction: ignore bins where |ref| < atol_fraction * max(|ref|)"""
        mask = np.abs(ref) > atol_fraction * np.max(np.abs(ref))
        if mask.sum() == 0:
            print(f"  {name}: reference is zero everywhere, skipping")
            return True
        frac_diff = np.abs(ours[mask] - ref[mask]) / np.abs(ref[mask])
        max_diff  = np.max(frac_diff)
        mean_diff = np.mean(frac_diff)
        ok = max_diff < rtol
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: max={max_diff:.3f} mean={mean_diff:.3f} ({status}, rtol={rtol})")
        print(f"    ours: {np.array2string(ours, precision=4)}")
        print(f"    ref:  {np.array2string(ref,  precision=4)}")
        return ok

    print("Comparing to bin-averaged pyccl reference (10% tolerance):")
    all_pass_ccl = True
    all_pass_ccl &= compare("w(theta)", w_ours,      w_pyccl)
    all_pass_ccl &= compare("gamma_t ", gamma_ours,  gamma_pyccl)
    all_pass_ccl &= compare("xi_+    ", xi_pos_ours, xi_pos_pyccl)
    all_pass_ccl &= compare("xi_-    ", xi_neg_ours, xi_neg_pyccl)

    if all_pass_ccl:
        print("\npyccl comparison: all transforms pass.")
    else:
        print("\npyccl comparison: some transforms differ — check above.")

print("\n=== All tests complete ===")