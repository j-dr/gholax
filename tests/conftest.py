import os
import pytest

try:
    from classy import Class as _Class  # noqa: F401
    classy_available = True
except ImportError:
    classy_available = False

requires_classy = pytest.mark.skipif(
    not classy_available, reason="classy not installed"
)


def pytest_addoption(parser):
    parser.addoption(
        "--plot-diagnostics",
        action="store_true",
        default=False,
        help="Save diagnostic plots for spectral equivalence tests to tests/plots/",
    )


@pytest.fixture
def plot_diagnostics(request):
    """Fixture that returns True when --plot-diagnostics is passed."""
    return request.config.getoption("--plot-diagnostics")

COSMO_PARAMS = {
    "standard": {
        "H0": 67.5,
        "ombh2": 0.022,
        "omch2": 0.12,
        "mnu": 0.06,
        "w": -1.0,
    },
    "heavy_nu": {
        "H0": 67.5,
        "ombh2": 0.022,
        "omch2": 0.12,
        "mnu": 0.3,
        "w": -1.0,
    },
    "dark_energy": {
        "H0": 67.5,
        "ombh2": 0.022,
        "omch2": 0.12,
        "mnu": 0.06,
        "w": -0.8,
    },
}

# w0wa test cosmologies — include As, ns for sigma8 calculations
W0WA_COSMO_PARAMS = {
    "wa_zero": {
        "H0": 67.5, "ombh2": 0.022, "omch2": 0.12, "mnu": 0.06,
        "As": 2.1, "ns": 0.965, "w": -1.0, "wa": 0.0,
    },
    "wa_neg": {
        "H0": 67.5, "ombh2": 0.022, "omch2": 0.12, "mnu": 0.06,
        "As": 2.1, "ns": 0.965, "w": -0.9, "wa": -0.5,
    },
    "wa_pos": {
        "H0": 67.5, "ombh2": 0.022, "omch2": 0.12, "mnu": 0.06,
        "As": 2.1, "ns": 0.965, "w": -0.9, "wa": 0.5,
    },
    "wa_large_neg": {
        "H0": 67.5, "ombh2": 0.022, "omch2": 0.12, "mnu": 0.06,
        "As": 2.1, "ns": 0.965, "w": -0.8, "wa": -0.8,
    },
}


def class_instance(params, N_ur=0.0):
    """Run CLASS for background-only quantities and return the Class object."""
    from classy import Class
    boltz = Class()
    boltz.set({
        "h": params["H0"] / 100,
        "omega_b": params["ombh2"],
        "omega_cdm": params["omch2"],
        "N_ur": N_ur,
        "N_ncdm": 1,
        "deg_ncdm": 3,
        "m_ncdm": params["mnu"] / 3,
        "Omega_Lambda": 0.0,
        "w0_fld": params["w"],
        "wa_fld": 0.0,
        "output": "",
    })
    boltz.compute()
    return boltz


def class_instance_sigma8(params, N_ur=0.0):
    """Run CLASS with mPk output for sigma8(z) calculations. Supports wa_fld."""
    from classy import Class
    boltz = Class()
    boltz.set({
        "output": "mPk",
        "P_k_max_h/Mpc": 20.0,
        "z_pk": "0.0,0.5,1.0,2.0",
        "A_s": params["As"] * 1e-9,
        "n_s": params["ns"],
        "h": params["H0"] / 100,
        "omega_b": params["ombh2"],
        "omega_cdm": params["omch2"],
        "N_ur": N_ur,
        "N_ncdm": 1,
        "deg_ncdm": 3,
        "m_ncdm": params["mnu"] / 3,
        "Omega_Lambda": 0.0,
        "w0_fld": params["w"],
        "wa_fld": params.get("wa", 0.0),
        "tau_reio": 0.0568,
    })
    boltz.compute()
    return boltz
