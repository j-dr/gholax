"""Tests for BAO alpha predictions appended to the RSD data vector.

Builds a synthetic FS+BAO data file by extending data/abacus_dr1_rsd.h5 with
alpha rows (DESI-like: iso for bins 0,3; par/perp for bins 1,2) whose
fiducials are computed at the config reference cosmology, so the predicted
alphas at the reference point must all equal 1.
"""

import copy
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from tests.conftest import requires_classy

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'example_configs', 'abcacus_dr1_rsd.yaml'
)
DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'abacus_dr1_rsd.h5'
)

requires_data = pytest.mark.skipif(
    not os.path.exists(DATA_PATH), reason="data/abacus_dr1_rsd.h5 not available"
)

# reference cosmology of example_configs/abcacus_dr1_rsd.yaml
FID_PARAMS = {
    "omch2": 0.12,
    "ombh2": 0.02237,
    "H0": 67.36,
    "mnu": 10 ** -1.22,
    "w": -1.0,
    "wa": 0.0,
}

ISO_BINS = [0, 3]
ANISO_BINS = [1, 2]
ALPHA_SIGMA = 0.02


def _fiducial_geometry():
    """Compute per-bin H(z) [km/s/Mpc], D_M(z) [Mpc], D_V(z) [Mpc], and rd
    [Mpc] at FID_PARAMS with the same ExpansionHistory settings RSDPK uses."""
    from gholax.theory.expansion_history import ExpansionHistory
    from gholax.theory.bao_alphas import sound_horizon_aubourg, C_KMS

    with h5py.File(DATA_PATH, "r") as f:
        zeff = f["z_fid"][:]

    eh = ExpansionHistory(zmin=0.0001, zmax=3.0, nz=200)
    state = eh.compute({}, dict(FID_PARAMS))

    e_z = np.array(jnp.interp(zeff, state["z_limber"], state["e_z_limber"]))
    chi = np.array(jnp.interp(zeff, state["z_limber"], state["chi_z_limber"]))
    h = FID_PARAMS["H0"] / 100.0
    Hz = FID_PARAMS["H0"] * e_z
    DM = chi / h
    DV = (DM**2 * C_KMS * zeff / Hz) ** (1.0 / 3.0)
    rd = float(
        sound_horizon_aubourg(
            FID_PARAMS["omch2"], FID_PARAMS["ombh2"], FID_PARAMS["mnu"]
        )
    )
    return zeff, Hz, DM, DV, rd


def _build_fsbao_file(out_path):
    """Extend the base RSD file with alpha rows, aux datasets, and a joint
    (block-diagonal) covariance. Measured alphas are 1 (the fiducial value)."""
    zeff, Hz, DM, DV, rd = _fiducial_geometry()

    with h5py.File(DATA_PATH, "r") as f:
        spectra = f["spectra"][:]
        cov = f["covariance"][:]
        n = len(spectra)
        cov = cov.reshape(n, n)
        aux = {
            k: f[k][:]
            for k in f.keys()
            if k not in ("spectra", "covariance", "pkell_windows")
        }
        windows = {k: f["pkell_windows"][k][:] for k in f["pkell_windows"]}

    alpha_rows = []
    for t, bins in (
        ("alpha_iso", ISO_BINS),
        ("alpha_par", ANISO_BINS),
        ("alpha_perp", ANISO_BINS),
    ):
        for b in bins:
            alpha_rows.append((t.encode("utf-8"), b, b, 0, 0.0, 1.0))
    alpha_rows = np.array(alpha_rows, dtype=spectra.dtype)
    m = len(alpha_rows)

    new_spectra = np.hstack([spectra, alpha_rows])
    new_cov = np.zeros((n + m, n + m), dtype=cov.dtype)
    new_cov[:n, :n] = cov

    # metadata: "*0" fields label the row element, "*1" fields the column
    for k, row in enumerate(alpha_rows):
        i = n + k
        new_cov[i, :]["spectrum_type0"] = row["spectrum_type"]
        new_cov[i, :]["zbin00"] = row["zbin0"]
        new_cov[i, :]["zbin01"] = row["zbin1"]
        new_cov[i, :]["ell0"] = row["ell"]
        new_cov[i, :]["separation0"] = row["separation"]
        new_cov[:, i]["spectrum_type1"] = row["spectrum_type"]
        new_cov[:, i]["zbin10"] = row["zbin0"]
        new_cov[:, i]["zbin11"] = row["zbin1"]
        new_cov[:, i]["ell1"] = row["ell"]
        new_cov[:, i]["separation1"] = row["separation"]
    # existing-element metadata on the new off-diagonal blocks
    for field in ("spectrum_type1", "zbin10", "zbin11", "ell1", "separation1"):
        new_cov[n:, :n][field] = cov[0, :][field][None, :]
    for field in ("spectrum_type0", "zbin00", "zbin01", "ell0", "separation0"):
        new_cov[:n, n:][field] = cov[:, 0][field][:, None]

    for k in range(m):
        new_cov[n + k, n + k]["value"] = ALPHA_SIGMA**2

    assert np.allclose(new_cov["value"], new_cov["value"].T)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("spectra", data=new_spectra)
        f.create_dataset("covariance", data=new_cov)
        for k, v in aux.items():
            f.create_dataset(k, data=v)
        grp = f.create_group("pkell_windows")
        for k, v in windows.items():
            grp.create_dataset(k, data=v)
        f.create_dataset("rd_fid", data=rd)
        f.create_dataset("zeff_bao", data=zeff)
        f.create_dataset("Hz_fid_bao", data=Hz)
        f.create_dataset("DM_fid_bao", data=DM)
        f.create_dataset("DV_fid_bao", data=DV)


def _fsbao_config(data_path):
    with open(CONFIG_PATH) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    dv = cfg["likelihood"]["RSDPK"]["data_vector"]
    dv["data_vector_info_filename"] = str(data_path)
    dv["spectrum_info"]["alpha_iso"] = {"bins0": ISO_BINS, "bins1": ISO_BINS}
    dv["spectrum_info"]["alpha_par"] = {"bins0": ANISO_BINS, "bins1": ANISO_BINS}
    dv["spectrum_info"]["alpha_perp"] = {"bins0": ANISO_BINS, "bins1": ANISO_BINS}
    return cfg


@pytest.fixture(scope="module")
def fsbao_model(tmp_path_factory):
    from gholax.util.model import Model

    path = tmp_path_factory.mktemp("bao") / "abacus_dr1_rsd_fsbao.h5"
    _build_fsbao_file(path)
    return Model(_fsbao_config(path))


@requires_classy
def test_sound_horizon_vs_class():
    """Aubourg+15 fitting formula agrees with CLASS rs_drag to <0.5%."""
    from classy import Class
    from gholax.theory.bao_alphas import sound_horizon_aubourg
    from tests.conftest import COSMO_PARAMS

    for name, p in COSMO_PARAMS.items():
        cosmo = Class()
        cosmo.set(
            {
                "h": p["H0"] / 100.0,
                "omega_b": p["ombh2"],
                "omega_cdm": p["omch2"],
                "m_ncdm": p["mnu"] / 3,
                "deg_ncdm": 3,
                "N_ncdm": 1,
                "N_ur": 0.00641,
            }
        )
        cosmo.compute()
        rd_class = cosmo.rs_drag()
        cosmo.struct_cleanup()
        rd_fit = float(sound_horizon_aubourg(p["omch2"], p["ombh2"], p["mnu"]))
        assert abs(rd_fit / rd_class - 1.0) < 5e-3, (
            f"{name}: rd_fit={rd_fit:.3f}, rd_class={rd_class:.3f}"
        )


@requires_data
def test_fiducial_alphas_are_unity(fsbao_model):
    """At the reference (fiducial) cosmology every predicted alpha is 1."""
    ref_params = fsbao_model.prior.get_reference_point()
    pred = np.array(
        fsbao_model.predict_model("RSDPK", ref_params, apply_scale_mask=False)
    )
    n_alpha = len(ISO_BINS) + 2 * len(ANISO_BINS)
    alphas = pred[-n_alpha:]
    assert np.allclose(alphas, 1.0, atol=1e-4), f"alphas = {alphas}"


@requires_data
def test_data_model_alignment(fsbao_model):
    """Alpha rows load, land after the P(k) block, and survive masking."""
    like = fsbao_model.likelihoods["RSDPK"]
    dv = like.observed_data_vector
    n_alpha = len(ISO_BINS) + 2 * len(ANISO_BINS)

    assert dv.spectrum_types == ["p_gg_ell", "alpha_iso", "alpha_par", "alpha_perp"]
    assert dv.n_dv == 600 + n_alpha
    # alpha rows are the last block, in type-then-bin order
    tail = dv.spectra[-n_alpha:]
    expected = (
        [(b"alpha_iso", b) for b in ISO_BINS]
        + [(b"alpha_par", b) for b in ANISO_BINS]
        + [(b"alpha_perp", b) for b in ANISO_BINS]
    )
    assert [(r["spectrum_type"], r["zbin0"]) for r in tail] == expected
    # measured alpha values present
    assert np.allclose(tail["value"], 1.0)
    # alphas are exempt from scale cuts: all their indices are in scale_mask
    alpha_idx = np.arange(600, 600 + n_alpha)
    assert np.all(np.isin(alpha_idx, np.array(dv.scale_mask)))
    # joint covariance inverted at masked size
    assert dv.cinv.shape == (dv.n_dv_masked, dv.n_dv_masked)

    ref_params = fsbao_model.prior.get_reference_point()
    pred = np.array(
        fsbao_model.predict_model("RSDPK", ref_params, apply_scale_mask=False)
    )
    assert len(pred) == dv.n_dv


@requires_data
def test_gradient_with_alphas(fsbao_model):
    """log_posterior gradient stays finite and matches finite differences."""
    model = fsbao_model
    param_norm = 0.1 * jnp.ones(len(model.param_names))
    grad_fn = jax.jit(jax.grad(model.log_posterior_scaled_params))
    grad_val = grad_fn(param_norm)
    grad_val.block_until_ready()
    assert jnp.all(jnp.isfinite(grad_val)), "AD gradient contains non-finite values"

    # FD on the cosmology params (the ones the alphas respond to)
    eps = 1e-3
    cosmo_idx = [
        i
        for i, name in enumerate(model.param_names)
        if name in ("As", "omch2", "ombh2", "H0", "w", "wa")
    ]
    rel_errs = []
    for i in cosmo_idx:
        ei = jnp.zeros_like(param_norm).at[i].set(eps)
        fp = float(model.log_posterior_scaled_params(param_norm + ei))
        fm = float(model.log_posterior_scaled_params(param_norm - ei))
        fd = (fp - fm) / (2 * eps)
        ad = float(grad_val[i])
        scale = max(abs(fd), abs(ad), 1.0)
        rel_errs.append(abs(ad - fd) / scale)
        print(f"{model.param_names[i]}: AD={ad:.4f}, FD={fd:.4f}")
    rel_errs = np.array(rel_errs)
    assert np.mean(rel_errs < 0.1) > 0.5, f"rel_errs = {rel_errs}"


@requires_data
def test_save_model_round_trip(fsbao_model, tmp_path):
    """save_model round-trips alpha rows and the scalar rd_fid dataset."""
    like = fsbao_model.likelihoods["RSDPK"]
    ref_params = fsbao_model.prior.get_reference_point()
    params_like = {k: ref_params[k] for k in like.sampled_params}
    out = tmp_path / "fsbao_model.h5"
    like.save_model(str(out), params_like, {})

    with h5py.File(out) as f:
        sp = f["spectra"][:]
        alpha = sp[np.char.startswith(sp["spectrum_type"].astype(str), "alpha")]
        assert len(alpha) == len(ISO_BINS) + 2 * len(ANISO_BINS)
        assert np.allclose(alpha["value"], 1.0, atol=1e-4)
        assert f["rd_fid"].shape == ()


@requires_data
def test_backward_compat_alpha_free():
    """An alpha-free config/file reproduces the plain FS behavior."""
    from gholax.util.model import Model

    with open(CONFIG_PATH) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = Model(copy.deepcopy(cfg))
    like = model.likelihoods["RSDPK"]
    dv = like.observed_data_vector

    assert dv.spectrum_types == ["p_gg_ell"]
    assert dv.n_dv == 600
    ref_params = model.prior.get_reference_point()
    pred = np.array(model.predict_model("RSDPK", ref_params, apply_scale_mask=False))
    assert len(pred) == 600
    assert np.all(np.isfinite(pred))
