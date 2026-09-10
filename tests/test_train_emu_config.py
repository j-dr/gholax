"""Emulator configs emitted by training carry the training-box param_ranges."""

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from gholax.training.train_nn_emu_torch import (  # noqa: E402
    _DEFAULT_PARAM_ORDER_SPEC,
    save_emu_config,
    training_param_ranges,
)
from gholax.theory.emulator import _build_range_arrays  # noqa: E402


def test_training_param_ranges_units_and_z():
    rng = np.random.default_rng(0)
    lo = np.array([1.1e-9, 0.93, 0.08, 0.0173, 52.0, -1.56, -2.0, 0.0])
    hi = np.array([3.1e-9, 1.01, 0.16, 0.0272, 82.0, -0.44, -0.3, 2.0])
    P = lo + (hi - lo) * rng.random((500, 8))
    r = training_param_ranges(P, _DEFAULT_PARAM_ORDER_SPEC, scale_As=True)
    assert "z" not in r and set(r) == set(_DEFAULT_PARAM_ORDER_SPEC) - {"z"}
    assert 1.1 <= r["As"][0] < r["As"][1] <= 3.1          # rescaled to 1e-9
    assert 0.93 <= r["ns"][0] < r["ns"][1] <= 1.01
    r_raw = training_param_ranges(P, _DEFAULT_PARAM_ORDER_SPEC, scale_As=False)
    assert r_raw["As"][1] < 1e-8
    _build_range_arrays(r, _DEFAULT_PARAM_ORDER_SPEC)  # consumable by the emulator


def test_save_emu_config_writes_param_ranges(tmp_path):
    out = str(tmp_path / "my_emu")
    ranges = {"ns": [0.93, 1.01], "As": [1.1, 3.1]}
    save_emu_config(out, "pkell0", {}, k=np.array([1e-3, 0.6]), param_ranges=ranges)
    cfg = yaml.safe_load(open(out + "_config.yaml"))
    assert cfg["param_ranges"] == ranges
    assert cfg["param_order_spec"] == _DEFAULT_PARAM_ORDER_SPEC
    save_emu_config(str(tmp_path / "bare"), "pkell0", {})
    assert "param_ranges" not in yaml.safe_load(open(str(tmp_path / "bare_config.yaml")))
