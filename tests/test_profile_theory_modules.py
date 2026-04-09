"""
Profiling measurements for all theory modules used in the implemented likelihoods.

Run with:
    pytest tests/test_profile_theory_modules.py -v -s

Each test JIT-compiles on a warmup call, then times N_REPEAT subsequent calls.
Timing and memory results are printed to stdout (visible with -s).
"""
import gholax.util.likelihood_module  # noqa: F401
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gholax.theory.expansion_history import ExpansionHistory
from gholax.theory.linear_growth import LinearGrowth
from gholax.theory.linear_growth_rate import LinearGrowthRate
from gholax.theory.real_space_biased_tracer_spectra import (
    RealSpaceBiasedTracerSpectra,
    RealSpaceMatterPowerSpectrum,
)
from gholax.theory.ia import DensityShapeIA, ShapeShapeIA
from gholax.theory.spectral_equivalence import SpectralEquivalence
from gholax.theory.redshift_space_biased_tracer_spectra import (
    RedshiftSpaceBiasedTracerSpectra,
)
import sys
import types
# Stub out flowjax so gholax.likelihood.__init__ doesn't fail if it's not installed
if "flowjax" not in sys.modules:
    _fj = types.ModuleType("flowjax")
    _fj_flows = types.ModuleType("flowjax.flows")
    _fj_dist = types.ModuleType("flowjax.distributions")
    _fj_bij = types.ModuleType("flowjax.bijections")
    _fj_flows.masked_autoregressive_flow = None
    _fj_dist.Normal = None
    _fj_bij.RationalQuadraticSpline = None
    sys.modules["flowjax"] = _fj
    sys.modules["flowjax.flows"] = _fj_flows
    sys.modules["flowjax.distributions"] = _fj_dist
    sys.modules["flowjax.bijections"] = _fj_bij

from gholax.likelihood.projection.limber import Limber
from gholax.likelihood.projection.lensing_counterterm import LensingCounterterm

N_REPEAT = 20

PARAMS_RAW = {
    "H0": 67.5,
    "ombh2": 0.022,
    "omch2": 0.12,
    "mnu": 0.06,
    "w": -1.0,
    "As": 2.083,
    "ns": 0.9649,
}
PARAMS = {k: jnp.array(v) for k, v in PARAMS_RAW.items()}
PARAMS_WITH_WA = {**PARAMS, "wa": jnp.array(0.0)}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _block(result):
    """Call block_until_ready on every leaf array in a pytree."""
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)


def _time_fn(fn, *args, n=N_REPEAT):
    """JIT-compile fn, warmup, then time n post-compile calls.

    Returns (jit_compile_ms, mean_ms, min_ms).
    """
    jit_fn = jax.jit(fn)

    t0 = time.perf_counter()
    _block(jit_fn(*args))
    jit_ms = (time.perf_counter() - t0) * 1e3

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _block(jit_fn(*args))
        times.append((time.perf_counter() - t0) * 1e3)

    return jit_ms, float(np.mean(times)), float(np.min(times))


def _measure_memory():
    """Return current JAX device memory usage in MB, or None if unavailable."""
    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
        if stats is not None:
            return {
                "current_MB": stats.get("bytes_in_use", 0) / 1e6,
                "peak_MB": stats.get("peak_bytes_in_use", 0) / 1e6,
            }
    except Exception:
        pass
    return None


def _print_results(label, jit_ms, mean_ms, min_ms, mem_before=None, mem_after=None):
    """Print formatted profiling results."""
    print(f"\n  {label}")
    print(f"    JIT compile: {jit_ms:.1f} ms")
    print(f"    Per-call:    mean={mean_ms:.2f} ms  min={min_ms:.2f} ms  (n={N_REPEAT})")
    if mem_before is not None and mem_after is not None:
        delta = mem_after["current_MB"] - mem_before["current_MB"]
        print(f"    Memory:      delta={delta:.1f} MB  peak={mem_after['peak_MB']:.1f} MB")
    elif mem_after is not None:
        print(f"    Memory:      current={mem_after['current_MB']:.1f} MB  peak={mem_after['peak_MB']:.1f} MB")


# ---------------------------------------------------------------------------
# ExpansionHistory (pure JAX analytic integration, no emulator)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nz,label", [
    (125, "nz=125 (default)"),
    (500, "nz=500"),
])
def test_profile_expansion_history(nz, label):
    mem_before = _measure_memory()
    eh = ExpansionHistory(
        zmin=0.0, zmax=2.0, nz=nz,
        use_direct_integration=True,
        use_emulator=False,
        use_boltzmann=False,
    )
    jit_ms, mean_ms, min_ms = _time_fn(eh.compute_analytic, PARAMS_WITH_WA)
    mem_after = _measure_memory()
    _print_results(f"ExpansionHistory.compute_analytic  {label}",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# LinearGrowth (ScalarEmulator for sigma8(z))
# ---------------------------------------------------------------------------

def test_profile_linear_growth():
    mem_before = _measure_memory()
    lg = LinearGrowth(
        zmin=0.0, zmax=2.0, nz=125,
        use_emulator=True,
        compute_w0wa=False,
        emulator_file_name="aemulus_nu_tier1_cosmo_only_heft_sigma8z_emu",
    )

    def fn(params):
        state = lg.compute_emulator({}, params)
        return state["sigma8_z"], state["D_z"]

    jit_ms, mean_ms, min_ms = _time_fn(fn, PARAMS)
    mem_after = _measure_memory()
    _print_results("LinearGrowth.compute_emulator (sigma8z)",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# LinearGrowthRate (ScalarEmulator for f(z))
# ---------------------------------------------------------------------------

def test_profile_linear_growth_rate():
    mem_before = _measure_memory()
    lgr = LinearGrowthRate(
        zmin=0.0, zmax=2.0, nz=125,
        use_emulator=True,
        emulator_file_name="aemulus_nu_tier1_cosmo_only_f_z_emu",
    )

    def fn(params):
        state = lgr.compute_emulator({}, params)
        return state["f_z"]

    jit_ms, mean_ms, min_ms = _time_fn(fn, PARAMS)
    mem_after = _measure_memory()
    _print_results("LinearGrowthRate.compute_emulator (f_z)",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# RealSpaceBiasedTracerSpectra (PijEmulator, 15 basis spectra)
# ---------------------------------------------------------------------------

def test_profile_real_space_biased_tracer_spectra():
    mem_before = _measure_memory()
    rsbts = RealSpaceBiasedTracerSpectra(
        zmin=0.0, zmax=2.0, nz=50,
        kmin=1e-3, kmax=3.95, nk=200,
        use_emulator=True,
        bias_model="heft",
        emulator_file_names="emu_config_dst_irresum.yaml",
    )

    def fn(params):
        state = rsbts.compute_emulator({}, params)
        return state["p_ij_real_space_bias_grid"]

    jit_ms, mean_ms, min_ms = _time_fn(fn, PARAMS)
    mem_after = _measure_memory()
    _print_results("RealSpaceBiasedTracerSpectra.compute_emulator (15 P_ij)",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# RealSpaceMatterPowerSpectrum (PijEmulator, P_11 only)
# ---------------------------------------------------------------------------

def test_profile_real_space_matter_power_spectrum():
    mem_before = _measure_memory()
    rsmp = RealSpaceMatterPowerSpectrum(
        zmin=0.0, zmax=2.0, nz=50,
        kmin=1e-3, kmax=3.95, nk=200,
        use_emulator=True,
        emulator_file_names="emu_config_dst_irresum.yaml",
    )

    def fn(params):
        state = rsmp.compute_emulator({}, params)
        return state["p_11_real_space_bias_grid"]

    jit_ms, mean_ms, min_ms = _time_fn(fn, PARAMS)
    mem_after = _measure_memory()
    _print_results("RealSpaceMatterPowerSpectrum.compute_emulator (P_11)",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# DensityShapeIA (MultiSpectrumEmulator)
# ---------------------------------------------------------------------------

def test_profile_density_shape_ia():
    mem_before = _measure_memory()
    dsia = DensityShapeIA(
        zmin=0.0, zmax=2.0, nz=50,
        kmin=1e-3, kmax=3.95, nk=200,
        use_emulator=True,
        emulator_file_names="emu_p_density_shape.yaml",
    )

    def fn(params):
        state = dsia.compute_emulator({}, params)
        return state["p_ij_real_space_density_shape_grid"]

    jit_ms, mean_ms, min_ms = _time_fn(fn, PARAMS)
    mem_after = _measure_memory()
    _print_results("DensityShapeIA.compute_emulator",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# ShapeShapeIA (3x MultiSpectrumEmulator, m=0,1,2)
# ---------------------------------------------------------------------------

def test_profile_shape_shape_ia():
    mem_before = _measure_memory()
    ssia = ShapeShapeIA(
        zmin=0.0, zmax=2.0, nz=50,
        kmin=1e-3, kmax=3.95, nk=200,
        use_emulator=True,
        emulator_file_names=[
            "emu_p0_shape_shape.yaml",
            "emu_p1_shape_shape.yaml",
            "emu_p2_shape_shape.yaml",
        ],
    )

    def fn(params):
        state = ssia.compute_emulator({}, params)
        return state["p_mij_real_space_shape_shape_grid"]

    jit_ms, mean_ms, min_ms = _time_fn(fn, PARAMS)
    mem_after = _measure_memory()
    _print_results("ShapeShapeIA.compute_emulator (3x13 spectra)",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# RedshiftSpaceBiasedTracerSpectra (4x MultiSpectrumEmulator + AP correction)
# ---------------------------------------------------------------------------

def test_profile_redshift_space_biased_tracer_spectra():
    # Pre-populate state from ExpansionHistory (needed for AP correction)
    eh = ExpansionHistory(
        zmin=0.0, zmax=2.3, nz=50,
        use_direct_integration=True,
        use_emulator=False,
        use_boltzmann=False,
    )
    eh_state = eh.compute({}, PARAMS_WITH_WA)

    # Representative effective redshifts and fiducial cosmology
    zeff = np.array([0.295, 0.510, 0.706, 0.930, 1.317])
    hz_fid = np.array([
        float(jnp.interp(z, eh_state["z_limber"], eh_state["e_z_limber"]))
        for z in zeff
    ])
    chiz_fid = np.array([
        float(jnp.interp(z, eh_state["z_limber"], eh_state["chi_z_limber"]))
        for z in zeff
    ])

    mem_before = _measure_memory()
    rsdpk = RedshiftSpaceBiasedTracerSpectra(
        zeff=zeff,
        hz_fid=hz_fid,
        chiz_fid=chiz_fid,
        zmin=0, zmax=2.3, nz=50,
        kmin=1e-3, kmax=0.601, nk=200,
        use_emulator=True,
        emulator_file_names=[
            "emu_pk0.yaml", "emu_pk2.yaml", "emu_pk4.yaml", "emu_pk6.yaml",
        ],
    )

    def fn(params):
        state = dict(eh_state)
        state = rsdpk.compute_emulator(state, params)
        return state["p_ij_ell_redshift_space_bias_grid"]

    jit_ms, mean_ms, min_ms = _time_fn(fn, PARAMS)
    mem_after = _measure_memory()
    _print_results("RedshiftSpaceBiasedTracerSpectra.compute_emulator (4 ell, AP)",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# SpectralEquivalence (w0wa → wCDM mapping via Newton + ODE)
# ---------------------------------------------------------------------------

def test_profile_spectral_equivalence():
    z_pk = jnp.linspace(0.0001, 3.0, 50)
    mem_before = _measure_memory()
    se = SpectralEquivalence(
        z=z_pk,
        sigma8_emulator_file_name="aemulus_nu_tier1_cosmo_only_heft_sigma8z_emu",
    )

    def fn(params):
        state = se.compute({}, params)
        return state["w_equiv_z"], state["As_equiv_z"]

    params_w0wa = {**PARAMS, "wa": jnp.array(-0.5)}
    jit_ms, mean_ms, min_ms = _time_fn(fn, params_w0wa)
    mem_after = _measure_memory()
    _print_results("SpectralEquivalence.compute (nz=50, Newton+ODE+sigma8_emu)",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# Limber integral (c_kk cosmic shear, synthetic state)
# ---------------------------------------------------------------------------

def _mock_data_vector(n_bins, nz_proj):
    """Create a minimal mock data vector object for Limber/LensingCounterterm."""
    class MockDataVector:
        pass
    dv = MockDataVector()
    dv.spectrum_info = {
        "c_kk": {
            "bins0": list(range(n_bins)),
            "bins1": list(range(n_bins)),
            "use_cross": True,
            "n_bins0_tot": n_bins,
            "n_bins1_tot": n_bins,
        },
    }
    dv.nz_d = np.zeros((n_bins, nz_proj))
    return dv


def _build_limber_state(n_bins, nz_proj, nz_pk, nk, n_ell):
    """Build synthetic state dict with projection kernels and power spectra."""
    key = jax.random.PRNGKey(0)
    n_pairs = n_bins * n_bins
    state = {
        "w_k": jnp.ones((n_bins, nz_proj)) * 1e-3,
        "w_ia": jnp.ones((n_bins, nz_proj)) * 1e-4,
        "p_mm": jnp.ones((1, nk, nz_pk)) * 1e3,
        "p_mi": jnp.ones((n_bins, nk, nz_pk)) * 1e1,
        "p_ii_ee": jnp.ones((n_pairs, nk, nz_pk)) * 1e-1,
        "chi_z_limber": jnp.linspace(1.0, 5000.0, nz_proj),
        "z_limber": jnp.linspace(0.0001, 3.0, nz_proj),
        "gl_scaled_weights": jnp.ones(nz_proj) / nz_proj * 5000.0,
        "zeff_w_d": jnp.linspace(0.2, 1.0, n_bins),
    }
    return state


def test_profile_limber():
    n_bins = 4
    nz_proj = 200
    nz_pk = 50
    nk = 200
    n_ell = 200
    kmin, kmax = 1e-3, 3.95

    dv = _mock_data_vector(n_bins, nz_proj)
    spectrum_types = ["c_kk"]
    spectrum_info = dv.spectrum_info

    mem_before = _measure_memory()
    limber = Limber(
        dv, spectrum_types, spectrum_info,
        zmin_proj=0.0001, zmax_proj=3.0, nz_proj=nz_proj,
        zmin_pk=0.0001, zmax_pk=3.0, nz_pk=nz_pk,
        kmin=kmin, kmax=kmax, nk=nk,
        n_ell=n_ell, l_max=3001,
        no_ia=False,
    )

    state = _build_limber_state(n_bins, nz_proj, nz_pk, nk, n_ell)

    def fn(state):
        s = dict(state)
        s = limber.compute_c_l(s, "c_kk")
        return s["c_kk"]

    jit_ms, mean_ms, min_ms = _time_fn(fn, state)
    mem_after = _measure_memory()
    _print_results("Limber.compute_c_l (c_kk, 4 bins, nz_proj=200)",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)


# ---------------------------------------------------------------------------
# LensingCounterterm (c_kk cosmic shear, synthetic state)
# ---------------------------------------------------------------------------

def test_profile_lensing_counterterm():
    n_bins = 4
    nz_proj = 200
    nz_pk = 50
    nk = 200
    n_ell = 200
    kmin, kmax = 1e-3, 3.95
    k_cutoff = 1.0
    lensing_ct_order = 3

    dv = _mock_data_vector(n_bins, nz_proj)
    spectrum_types = ["c_kk"]
    spectrum_info = dv.spectrum_info

    mem_before = _measure_memory()
    lct = LensingCounterterm(
        dv, spectrum_types, spectrum_info,
        zmin_proj=0.0001, zmax_proj=3.0, nz_proj=nz_proj,
        zmin_pk=0.0001, zmax_pk=3.0, nz_pk=nz_pk,
        kmin=kmin, kmax=kmax, nk=nk,
        k_cutoff=k_cutoff,
        n_ell=n_ell, l_max=3001,
        lensing_counterterm_order=lensing_ct_order,
    )

    state = _build_limber_state(n_bins, nz_proj, nz_pk, nk, n_ell)
    # LensingCounterterm additionally needs:
    n_pairs = n_bins * n_bins
    state["sigma8_z"] = jnp.linspace(0.8, 0.4, nz_pk)
    state["p_11_real_space_bias_grid"] = jnp.ones((nk, nz_pk)) * 1e3
    state["omegam"] = jnp.array(0.3)
    state["c_kk"] = jnp.ones((n_pairs, n_ell)) * 1e-6
    for kernel in ["w_k", "w_ia"]:
        state[f"chi_inv_eff_{kernel}"] = jnp.ones(n_bins) / 3000.0

    # LensingCounterterm.compute needs param_indices set up
    # Build a fake param_vec and param_indices matching lct.indexed_params
    param_names = []
    for N in range(lensing_ct_order):
        for o in range(lensing_ct_order):
            if o <= N:
                param_names.append(f"sigma_{N + 2}_{o}")
            else:
                param_names.append("NA")
    param_names = list(dict.fromkeys(param_names))
    param_vec_vals = {p: jnp.array(0.0) for p in param_names}

    # Build param_indices mapping (same as GaussianLikelihood does)
    all_param_names = list(param_vec_vals.keys())
    lct.param_indices = {}
    for key, idx_arr in lct.indexed_params.items():
        indices = np.array([[all_param_names.index(p) for p in row] for row in idx_arr])
        lct.param_indices[key] = indices

    param_vec = jnp.array([param_vec_vals[p] for p in all_param_names])

    def fn(param_vec, state):
        s = dict(state)
        # Inline the core of LensingCounterterm.compute
        log_kval = jnp.log10(lct.k_nodes)
        sigma_N_o_emu = lct.sigma_N_o(s, log_kval)
        return sigma_N_o_emu

    jit_ms, mean_ms, min_ms = _time_fn(fn, param_vec, state)
    mem_after = _measure_memory()
    _print_results(f"LensingCounterterm.sigma_N_o (order={lensing_ct_order}, k_cutoff={k_cutoff})",
                   jit_ms, mean_ms, min_ms, mem_before, mem_after)
