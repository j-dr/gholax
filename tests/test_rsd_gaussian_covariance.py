"""Gaussian covariance for redshift-space P_ell(k) multipoles.

Regression test for audit item A2: ``RedshiftSpaceMultipoles.gaussian_variance``
previously divided by ``self.delta_ell`` and ``self.ell_eff`` -- attributes that
belong to the angular (C_ell) ``TwoPointSpectrum`` class and are never defined
here -- so any real call raised ``AttributeError``. The class defines k-space
binning instead (``ko_eff`` / ``delta_k``), and the Gaussian variance of
P_ell(k) bandpowers follows from mode counting in spherical k-shells:

    N_modes(k) = V_survey * k^2 * delta_k / (2 * pi^2)
    Var[P(k)]  = (P_ac P_bd + P_ad P_bc) / N_modes(k)

For a single-tracer auto spectrum this reduces to 2 (P + 1/nbar)^2 / N_modes.

The test bypasses the HDF5 ``load_data`` path and sets the handful of attributes
that ``gaussian_covariance`` / ``gaussian_variance`` actually read, which is far
less brittle than synthesizing a full data-vector file.
"""
import numpy as np
import pytest

from gholax.data_vector.redshift_space_multipoles import RedshiftSpaceMultipoles


def _make_toy_rsd_dv():
    """Build a minimal RedshiftSpaceMultipoles with attributes set directly."""
    spectrum_type = "p_gg_ell"
    bin_pair = (0, 0)
    ells = (0, 2)
    ko = np.array([0.05, 0.10, 0.15])
    delta_k = 0.01
    v_survey = 1.0e9  # [Mpc/h]^3
    noise = 1.0e3

    dt = np.dtype(
        [
            ("spectrum_type", "S10"),
            ("zbin0", int),
            ("zbin1", int),
            ("ell", int),
            ("separation", float),
            ("value", float),
        ]
    )

    rows = []
    for ell in ells:
        for k in ko:
            # arbitrary smooth, positive P_ell(k)
            val = 1.0e4 / (1.0 + (k / 0.1) ** 2)
            rows.append((spectrum_type, bin_pair[0], bin_pair[1], ell, k, val))
    spectra = np.array(rows, dtype=dt)

    dv = RedshiftSpaceMultipoles.__new__(RedshiftSpaceMultipoles)
    dv.spectra = spectra
    dv.spectrum_types = [spectrum_type]
    dv.n_dv = len(spectra)
    dv.ko_eff = ko
    dv.delta_k = delta_k
    dv.spectrum_info = {
        spectrum_type: {
            "n_dv_per_bin": len(ells) * len(ko),
            "bin_pairs": [bin_pair],
            "separation": spectra["separation"].copy(),
        }
    }
    dv.covariance_info = {
        "v_survey": v_survey,
        spectrum_type: {f"{bin_pair[0]}_{bin_pair[1]}": {"noise": noise}},
    }
    return dv, dict(
        spectrum_type=spectrum_type,
        bin_pair=bin_pair,
        ko=ko,
        delta_k=delta_k,
        v_survey=v_survey,
        noise=noise,
        spectra=spectra,
    )


def test_gaussian_covariance_runs_and_is_psd():
    dv, _ = _make_toy_rsd_dv()

    # Must not raise AttributeError (delta_ell / ell_eff regression).
    cov = dv.gaussian_covariance()

    values = np.asarray(cov["value"])
    assert values.shape == (dv.n_dv, dv.n_dv)

    # symmetric
    assert np.allclose(values, values.T, atol=0, rtol=0)

    # positive semi-definite
    eigvals = np.linalg.eigvalsh(values)
    assert np.all(eigvals >= -1e-8 * np.max(np.abs(eigvals)))

    # diagonal is strictly positive for this toy (nonzero P + noise, finite volume)
    assert np.all(np.diag(values) > 0)


def test_gaussian_variance_matches_mode_counting_formula():
    dv, meta = _make_toy_rsd_dv()
    st = meta["spectrum_type"]
    b0, b1 = meta["bin_pair"]

    var = dv.gaussian_variance(st, st, b0, b1, b0, b1)

    p_wn = meta["spectra"]["value"] + meta["noise"]
    k = meta["spectra"]["separation"]
    n_modes = meta["v_survey"] * k ** 2 * meta["delta_k"] / (2.0 * np.pi ** 2)
    expected = 2.0 * p_wn ** 2 / n_modes  # auto: P_ac P_bd + P_ad P_bc = 2 P_wn^2

    assert var.shape == expected.shape
    assert np.allclose(var, expected, rtol=1e-12, atol=0)


def test_no_reference_to_undefined_binning_attributes():
    """gaussian_variance must not depend on the C_ell binning attributes."""
    dv, meta = _make_toy_rsd_dv()
    st = meta["spectrum_type"]
    b0, b1 = meta["bin_pair"]

    assert not hasattr(dv, "delta_ell")
    assert not hasattr(dv, "ell_eff")

    # Succeeds despite those attributes being absent.
    dv.gaussian_variance(st, st, b0, b1, b0, b1)
