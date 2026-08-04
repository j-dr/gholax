"""Validation of the BAO implementation against the published DESI DR2 data.

Unlike tests/test_bao_alphas.py -- which builds a synthetic file whose alphas are
1 by construction and so only exercises plumbing -- these tests use the real DESI
DR2 BAO likelihood (arXiv:2503.14738) and check the physics against CLASS:

  * the Aubourg+15 r_d fitting formula vs CLASS rs_drag over the prior volume
  * D_M / D_H on the z_limber grid that BAOAlphas actually interpolates on
  * a BAO-only data vector builds and predicts (no p_gg_ell block)
  * chi^2 from RSDPK matches a direct numpy/CLASS computation

Build the data file first:  python bin/make_desi_dr2_bao_dv.py
"""

import os

import numpy as np
import pytest

from tests.conftest import requires_classy

C_KMS = 2.99792458e5

REPO = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(REPO, "data", "desi_dr2_bao.h5")
RAW_DIR = os.path.join(REPO, "data", "desi_dr2_bao_raw")
CONFIG_PATH = os.path.join(REPO, "example_configs", "desi_dr2_bao_lcdm_bbn.yaml")

requires_data = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="data/desi_dr2_bao.h5 not available (run bin/make_desi_dr2_bao_dv.py)",
)

# Sum m_nu = 0.06 eV in one eigenstate, N_eff = 3.044 (DESI DR2 baseline)
MNU = 0.06
CLASS_NU = {"N_ur": 2.0308, "N_ncdm": 1, "m_ncdm": MNU, "deg_ncdm": 1}

# DESI DR2 + BBN best fit, and the seven tracer effective redshifts
BESTFIT = {"omch2": 0.1169, "ombh2": 0.02218, "H0": 68.51}
ZEFF = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])


def _class_background(omch2, ombh2, H0):
    from classy import Class

    cosmo = Class()
    cosmo.set({"h": H0 / 100.0, "omega_b": ombh2, "omega_cdm": omch2, **CLASS_NU})
    cosmo.compute()
    return cosmo


@pytest.fixture(scope="module")
def bao_model():
    from gholax.util.model import Model

    return Model(CONFIG_PATH)


@requires_classy
def test_sound_horizon_matches_class():
    """Aubourg+15 r_d agrees with CLASS rs_drag across the LCDM prior volume.

    A fractional error in r_d propagates directly into H_0, so 0.15% here bounds
    the induced H_0 systematic at ~0.1 km/s/Mpc, i.e. under 0.2 sigma.
    """
    from gholax.theory.bao_alphas import sound_horizon_aubourg

    devs = []
    for omch2 in (0.10, 0.115, 0.12, 0.125, 0.14):
        for ombh2 in (0.0210, 0.02218, 0.0235):
            cosmo = _class_background(omch2, ombh2, 67.36)
            rd_class = cosmo.rs_drag()
            cosmo.struct_cleanup()
            cosmo.empty()
            rd_fit = float(sound_horizon_aubourg(omch2, ombh2, MNU))
            devs.append(abs(rd_fit / rd_class - 1))

    assert max(devs) < 1.5e-3, f"max r_d deviation {max(devs):.2e}"


@requires_classy
def test_distances_on_limber_grid_match_class():
    """D_M and D_H agree with CLASS at the DESI z_eff.

    Interpolation is done exactly as BAOAlphas.compute does it -- onto the
    Gauss-Legendre-in-chi ``z_limber`` grid, not the native z grid -- so this
    covers the interpolation error as well as the background integration.
    """
    import jax.numpy as jnp

    from gholax.theory.expansion_history import ExpansionHistory

    params = dict(BESTFIT, mnu=MNU, w=-1.0, wa=0.0)
    eh = ExpansionHistory(zmin=0.0001, zmax=3.0, nz=200)
    state = eh.compute({}, params)

    e_z = np.array(jnp.interp(ZEFF, state["z_limber"], state["e_z_limber"]))
    chi = np.array(jnp.interp(ZEFF, state["z_limber"], state["chi_z_limber"]))
    DM = chi / (params["H0"] / 100.0)
    DH = C_KMS / (params["H0"] * e_z)

    cosmo = _class_background(**BESTFIT)
    DM_ref = np.array([cosmo.angular_distance(z) * (1 + z) for z in ZEFF])
    DH_ref = np.array([1.0 / cosmo.Hubble(z) for z in ZEFF])
    cosmo.struct_cleanup()
    cosmo.empty()

    assert np.abs(DM / DM_ref - 1).max() < 5e-4
    assert np.abs(DH / DH_ref - 1).max() < 5e-4


@requires_data
def test_bao_only_pipeline_builds(bao_model):
    """RSDPK builds and predicts with a data vector that has no p_gg_ell block."""
    like = bao_model.likelihoods["RSDPK"]
    dv = like.observed_data_vector

    # only the expansion history and the BAO module are needed
    assert [type(m).__name__ for m in like.likelihood_pipeline] == [
        "ExpansionHistory",
        "BAOAlphas",
    ]
    assert like.fs_types == []
    assert dv.spectrum_types == ["alpha_iso", "alpha_par", "alpha_perp"]
    # 1 iso (BGS) + 6 par + 6 perp
    assert dv.n_dv == 13
    # alphas are scale-cut exempt, so nothing is masked away
    assert dv.n_dv_masked == 13
    assert dv.cinv.shape == (13, 13)

    pred = np.array(
        bao_model.predict_model("RSDPK", BESTFIT, apply_scale_mask=False)
    )
    assert pred.shape == (13,)
    assert np.all(np.isfinite(pred))
    # near the best fit every alpha should sit within a few percent of unity
    assert np.abs(pred - 1).max() < 0.05

    # a BAO-only vector has no pre-window analogue
    with pytest.raises(ValueError, match="BAO-only"):
        like.predict_model(BESTFIT, {}, apply_window=False)


@requires_data
def test_measured_alphas_match_desi_file():
    """The alphas in the data file reproduce the published D_X/r_d values."""
    import h5py

    mean_path = os.path.join(RAW_DIR, "desi_gaussian_bao_ALL_GCcomb_mean.txt")
    if not os.path.exists(mean_path):
        pytest.skip("raw DESI mean file not cached")

    raw = np.genfromtxt(mean_path, dtype=None, encoding="utf-8",
                        names=["z", "value", "quantity"])
    with h5py.File(DATA_PATH) as f:
        spectra = f["spectra"][:]
        rd_fid = f["rd_fid"][()]
        fid = {
            "alpha_iso": f["DV_fid_bao"][:],
            "alpha_perp": f["DM_fid_bao"][:],
            "alpha_par": f["Hz_fid_bao"][:],
        }

    z_to_bin = {z: i for i, z in enumerate(ZEFF)}
    q_to_type = {
        "DV_over_rs": "alpha_iso",
        "DM_over_rs": "alpha_perp",
        "DH_over_rs": "alpha_par",
    }

    assert len(raw) == len(spectra)
    for row in raw:
        t = q_to_type[str(row["quantity"])]
        b = z_to_bin[float(row["z"])]
        if t == "alpha_par":
            factor = fid[t][b] * rd_fid / C_KMS
        else:
            factor = rd_fid / fid[t][b]
        expected = factor * float(row["value"])
        got = spectra[
            (spectra["spectrum_type"] == t.encode()) & (spectra["zbin0"] == b)
        ]["value"]
        assert len(got) == 1
        assert np.isclose(got[0], expected, rtol=1e-12)


def _desi_reference(omch2, ombh2, H0, rd=None):
    """chi^2 and model vector from CLASS + numpy, sharing no code with gholax.

    If *rd* is given it overrides CLASS's rs_drag, which lets a caller isolate the
    conventions from the accuracy of gholax's r_d fitting formula.
    """
    mean_path = os.path.join(RAW_DIR, "desi_gaussian_bao_ALL_GCcomb_mean.txt")
    cov_path = os.path.join(RAW_DIR, "desi_gaussian_bao_ALL_GCcomb_cov.txt")
    if not (os.path.exists(mean_path) and os.path.exists(cov_path)):
        pytest.skip("raw DESI files not cached")

    raw = np.genfromtxt(mean_path, dtype=None, encoding="utf-8",
                        names=["z", "value", "quantity"])
    cov = np.loadtxt(cov_path)

    cosmo = _class_background(omch2, ombh2, H0)
    if rd is None:
        rd = cosmo.rs_drag()
    model = []
    for row in raw:
        z = float(row["z"])
        DM = cosmo.angular_distance(z) * (1 + z)
        DH = 1.0 / cosmo.Hubble(z)
        q = str(row["quantity"])
        if q == "DM_over_rs":
            model.append(DM / rd)
        elif q == "DH_over_rs":
            model.append(DH / rd)
        else:
            model.append((DM**2 * z * DH) ** (1.0 / 3.0) / rd)
    cosmo.struct_cleanup()
    cosmo.empty()

    model = np.array(model)
    resid = raw["value"] - model
    return float(resid @ np.linalg.inv(cov) @ resid), model


@requires_classy
@requires_data
@pytest.mark.parametrize(
    "omch2,ombh2,H0",
    [(0.1169, 0.02218, 68.51), (0.1200, 0.02237, 67.36), (0.1350, 0.02350, 65.0)],
)
def test_chi2_matches_direct_computation(bao_model, omch2, ombh2, H0):
    """RSDPK's chi^2 matches a direct CLASS + numpy evaluation at the same r_d.

    This is the sharpest check of the alpha conventions, the block ordering, and
    the covariance permutation. Feeding the reference gholax's own r_d removes the
    fitting-formula difference, so what remains is pure bookkeeping plus the tiny
    distance-interpolation error; anything larger is a bug.
    """
    from gholax.theory.bao_alphas import sound_horizon_aubourg

    rd_gholax = float(sound_horizon_aubourg(omch2, ombh2, MNU))
    chi2_ref, _ = _desi_reference(omch2, ombh2, H0, rd=rd_gholax)

    like = bao_model.likelihoods["RSDPK"]
    chi2_gholax = -2.0 * float(
        like.compute({"omch2": omch2, "ombh2": ombh2, "H0": H0})
    )

    assert abs(chi2_gholax - chi2_ref) < 0.05, (
        f"chi2 gholax={chi2_gholax:.4f} ref={chi2_ref:.4f} "
        f"diff={chi2_gholax - chi2_ref:+.4f}"
    )


@requires_classy
@requires_data
@pytest.mark.parametrize(
    "omch2,ombh2,H0",
    [(0.1169, 0.02218, 68.51), (0.1200, 0.02237, 67.36), (0.1350, 0.02350, 65.0)],
)
def test_rd_approximation_is_the_only_difference(bao_model, omch2, ombh2, H0):
    """The gholax/CLASS model difference is a pure 1/r_d rescale.

    Every alpha carries r_d as an overall 1/r_d, so if the Aubourg formula were the
    only approximation, the ratio of the gholax and CLASS model vectors would be a
    constant equal to r_d(CLASS)/r_d(Aubourg). Confirming that -- rather than just
    bounding a chi^2 difference -- pins the discrepancy on the fitting formula and
    rules out a z-dependent error in the distances or a mis-ordered data vector.
    """
    import h5py

    from gholax.theory.bao_alphas import sound_horizon_aubourg

    _, model_class = _desi_reference(omch2, ombh2, H0)

    # convert the reference D_X/r_d vector into gholax's alpha convention
    mean_path = os.path.join(RAW_DIR, "desi_gaussian_bao_ALL_GCcomb_mean.txt")
    raw = np.genfromtxt(mean_path, dtype=None, encoding="utf-8",
                        names=["z", "value", "quantity"])
    with h5py.File(DATA_PATH) as f:
        rd_fid = f["rd_fid"][()]
        fid = {
            "alpha_iso": f["DV_fid_bao"][:],
            "alpha_perp": f["DM_fid_bao"][:],
            "alpha_par": f["Hz_fid_bao"][:],
        }
    q_to_type = {
        "DV_over_rs": "alpha_iso",
        "DM_over_rs": "alpha_perp",
        "DH_over_rs": "alpha_par",
    }
    z_to_bin = {z: i for i, z in enumerate(ZEFF)}
    ref_alpha = {}
    for row, m in zip(raw, model_class):
        t = q_to_type[str(row["quantity"])]
        b = z_to_bin[float(row["z"])]
        factor = (
            fid[t][b] * rd_fid / C_KMS if t == "alpha_par" else rd_fid / fid[t][b]
        )
        ref_alpha[(t, b)] = factor * m

    like = bao_model.likelihoods["RSDPK"]
    spectra = like.observed_data_vector.spectra
    ref_vec = np.array(
        [ref_alpha[(r["spectrum_type"].decode(), r["zbin0"])] for r in spectra]
    )
    pred = np.array(
        bao_model.predict_model(
            "RSDPK", {"omch2": omch2, "ombh2": ombh2, "H0": H0},
            apply_scale_mask=False,
        )
    )

    ratio = pred / ref_vec

    cosmo = _class_background(omch2, ombh2, H0)
    rd_class = cosmo.rs_drag()
    cosmo.struct_cleanup()
    cosmo.empty()
    expected = rd_class / float(sound_horizon_aubourg(omch2, ombh2, MNU))

    # the ratio is constant to within the distance-interpolation error...
    assert ratio.max() - ratio.min() < 5e-4, (
        f"model ratio is not constant: spread {ratio.max() - ratio.min():.2e}"
    )
    # ...and that constant is exactly the r_d ratio
    assert abs(ratio.mean() / expected - 1) < 5e-4, (
        f"ratio {ratio.mean():.7f} != rd_CLASS/rd_Aubourg {expected:.7f}"
    )
