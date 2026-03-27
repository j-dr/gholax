"""Integration test: RSD emulator predictions vs exact (CLASS + velocileptors) at reference point."""

import copy
import os

import numpy as np
import pytest
import yaml

from tests.conftest import requires_classy

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'example_configs', 'abcacus_dr1_rsd.yaml'
)


@requires_classy
def test_rsd_emulator_vs_exact():
    from gholax.util.model import Model

    with open(CONFIG_PATH) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Deepcopy before building models — Model.__init__ mutates cfg in place
    exact_cfg = copy.deepcopy(cfg)
    for module_cfg in exact_cfg.get('theory', {}).values():
        if isinstance(module_cfg, dict):
            module_cfg['use_emulator'] = False
            module_cfg['use_boltzmann'] = True
    for lname, like_cfg in exact_cfg.get('likelihood', {}).items():
        if lname == 'params':
            continue
        if isinstance(like_cfg, dict):
            like_cfg['use_boltzmann'] = True

    # Emulator model (default config)
    emu_model = Model(cfg)
    # Exact model (CLASS + velocileptors)
    exact_model = Model(exact_cfg)

    # Evaluate at prior reference point
    ref_params = emu_model.prior.get_reference_point()

    emu_pred = np.array(
        emu_model.predict_model('RSDPK', ref_params, apply_scale_mask=False)
    )
    exact_pred = np.array(
        exact_model.predict_model('RSDPK', ref_params, apply_scale_mask=False)
    )

    # Compare on scale-masked data vector (excludes unused multipoles)
    like = emu_model.likelihoods['RSDPK']
    dv = like.observed_data_vector
    mask = dv.scale_mask

    emu_masked = emu_pred[mask]
    exact_masked = exact_pred[mask]

    # Fractional difference
    frac_diff = np.abs((emu_masked - exact_masked) / exact_masked)
    max_frac_diff = np.max(frac_diff)
    median_frac_diff = np.median(frac_diff)
    p95_frac_diff = np.percentile(frac_diff, 95)
    print(f"Fractional diff — max: {max_frac_diff:.4e}, median: {median_frac_diff:.4e}, 95th: {p95_frac_diff:.4e}")
    assert max_frac_diff < 0.15, f"Max fractional diff {max_frac_diff:.4e} exceeds 15%"
    assert p95_frac_diff < 0.05, f"95th percentile frac diff {p95_frac_diff:.4e} exceeds 5%"

    # Log-likelihood comparison
    data_masked = np.array(dv.measured_spectra[mask])
    cinv = np.array(dv.cinv)

    diff_emu = data_masked - emu_masked
    diff_exact = data_masked - exact_masked
    emu_ll = float(-0.5 * diff_emu @ cinv @ diff_emu)
    exact_ll = float(-0.5 * diff_exact @ cinv @ diff_exact)
    delta_ll = abs(emu_ll - exact_ll)
    print(f"Log-like — emu: {emu_ll:.2f}, exact: {exact_ll:.2f}, delta: {delta_ll:.2f}")
    assert delta_ll < 50.0, f"Delta log-like {delta_ll:.2f} exceeds 50"
