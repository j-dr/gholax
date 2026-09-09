"""Emulator training-box guards: yaml param_ranges -> input clipping in every
emulator class, the Model-level smooth penalty, and the construction-time
prior-support warning."""

import copy
import os
import types
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from gholax.theory.emulator import (
    _DEFAULT_PARAM_ORDER,
    Emulator,
    MultiSpectrumEmulator,
    PijEmulator,
    ScalarEmulator,
)
from gholax.theory.spectral_equivalence import build_equiv_cparam_grid
from gholax.util.model import (
    Model,
    _collect_emulator_ranges,
    _sampled_range_edges,
    _warn_prior_outside_ranges,
)

EMU_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gholax", "theory", "emu_weights",
)
PARAMS = {"H0": 67.5, "ombh2": 0.022, "omch2": 0.12, "mnu": 0.06,
          "w": -1.0, "As": 2.083, "ns": 0.9649}
Z = jnp.array([0.3, 0.5])
NS_COL = _DEFAULT_PARAM_ORDER.index("ns")


def _grid(**over):
    p = {k: jnp.array(v) for k, v in {**PARAMS, **over}.items()}
    return build_equiv_cparam_grid(p, Z, {})


def _load(name):
    return yaml.safe_load(open(os.path.join(EMU_DIR, name)))


def _abs_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    for k in ("pij_bases",):
        if k in cfg:
            cfg[k] = [os.path.join(EMU_DIR, b) for b in cfg[k]]
    for k in ("spec_base", "filebase"):
        if k in cfg:
            cfg[k] = os.path.join(EMU_DIR, cfg[k])
    return cfg


def _write(tmp_path, cfg, name="emu.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def _assert_clip_equivalent(predict):
    hi = float(_load("emu_pk0.yaml")["param_ranges"]["ns"][1])
    out_edge = predict(_grid(ns=hi))
    out_far = predict(_grid(ns=hi + 0.05))
    out_in = predict(_grid())
    assert np.allclose(out_far, out_edge, rtol=1e-6, atol=0)
    assert not np.allclose(out_in, out_edge, rtol=1e-3)
    assert np.isfinite(out_far).all()


def test_pij_clip_equivalence(tmp_path):
    cfg = _load("emu_config_dst_irresum.yaml")
    emu = PijEmulator(_write(tmp_path, _abs_cfg(cfg)), abspath=True)
    assert emu.input_param_order is None
    assert emu.effective_param_order == _DEFAULT_PARAM_ORDER
    assert emu.param_ranges["ns"] == tuple(cfg["param_ranges"]["ns"])
    assert emu.sigma8z_emu.param_ranges == emu.param_ranges
    _assert_clip_equivalent(emu.predict)
    _assert_clip_equivalent(emu.pij_emus[0].predict)
    bare = copy.deepcopy(cfg); bare.pop("param_ranges")
    ref = PijEmulator(_write(tmp_path, _abs_cfg(bare), "bare.yaml"), abspath=True)
    assert ref.param_lo is None and ref.param_ranges == {}
    assert np.allclose(ref.predict(_grid()), emu.predict(_grid()))


def test_scalar_clip_equivalence():
    cfg = _load("emu_sigma8z.yaml")
    emu = ScalarEmulator(_abs_cfg(cfg), data_dir="")
    _assert_clip_equivalent(emu.predict)
    bare = ScalarEmulator(cfg["filebase"])
    assert bare.param_lo is None and bare.param_ranges == {}
    assert np.allclose(bare.predict(_grid()), emu.predict(_grid()))
    override = ScalarEmulator(cfg["filebase"], param_ranges={"ns": [0.9, 0.95]})
    assert override.param_ranges == {"ns": (0.9, 0.95)}
    assert np.allclose(override.predict(_grid()), bare.predict(_grid(ns=0.95)))


def test_multispectrum_clip_and_child(tmp_path):
    cfg = _load("emu_p_density_shape.yaml")
    emu = MultiSpectrumEmulator(_write(tmp_path, _abs_cfg(cfg)), abspath=True)
    assert emu.sigma8z_emu.param_ranges == emu.param_ranges
    order = emu.input_param_order
    ns = order.index("ns")

    def predict(g):
        return emu.predict(g[:, [_DEFAULT_PARAM_ORDER.index(p) for p in order]])

    _assert_clip_equivalent(predict)
    assert emu.param_hi[ns] == pytest.approx(cfg["param_ranges"]["ns"][1])


@pytest.mark.parametrize("name", ["emu_config_dst_irresum.yaml",
                                  "emu_p_density_shape.yaml", "emu_sigma8z.yaml"])
def test_yaml_without_ranges_unchanged(tmp_path, name):
    cfg = _load(name); cfg.pop("param_ranges")
    path = _write(tmp_path, _abs_cfg(cfg))
    if "pij_bases" in cfg:
        emu = PijEmulator(path, abspath=True)
    elif "spec_base" in cfg:
        emu = MultiSpectrumEmulator(path, abspath=True)
    else:
        emu = ScalarEmulator(_abs_cfg(cfg), data_dir="")
    assert emu.param_lo is None and emu.param_ranges == {}
    order = getattr(emu, "input_param_order", None) or _DEFAULT_PARAM_ORDER
    g = _grid()[:, [_DEFAULT_PARAM_ORDER.index(p) for p in order]]
    assert np.isfinite(emu.predict(g)).all()


def test_unknown_range_name_raises():
    cfg = _load("emu_sigma8z.yaml")
    with pytest.raises(ValueError):
        ScalarEmulator(cfg["filebase"], param_ranges={"mnu": [0.01, 0.5]})
    with pytest.raises(ValueError):
        ScalarEmulator(cfg["filebase"], param_ranges={"ns": [1.0, 0.9]})


def test_clip_is_differentiable():
    cfg = _load("emu_sigma8z.yaml")
    emu = ScalarEmulator(_abs_cfg(cfg), data_dir="")
    f = jax.jit(lambda g: jnp.sum(emu.predict(g)))
    g = _grid(ns=1.2)
    grad = jax.grad(f)(g)
    assert np.isfinite(grad).all() and np.all(grad[:, NS_COL] == 0)
    assert np.any(jax.grad(f)(_grid())[:, NS_COL] != 0)
    assert np.isfinite(jax.vmap(emu.predict)(jnp.stack([g, _grid()]))).all()


def _shim(names, ranges, penalty=True, soft=0.05):
    m = types.SimpleNamespace(emulator_range_penalty=penalty,
                              emulator_range_softness=soft)
    m._range_idx, m._range_lo, m._range_hi = _sampled_range_edges(names, ranges)
    return m


def test_range_penalty_zero_inside_negative_outside():
    names = ["As", "ns", "mnu", "b1"]
    ranges = {"ns": (0.93, 1.01), "logmnu": (-2.0, -0.30103), "z": (0.0, 3.0)}
    m = _shim(names, ranges)
    pen = lambda x: Model._range_penalty(m, jnp.asarray(x))
    assert float(pen([2.0, 0.97, 0.06, 1.0])) == 0.0
    assert float(pen([2.0, 0.97, 0.01, 1.0])) == 0.0   # mnu edge via 10**logmnu
    assert float(pen([2.0, 1.02, 0.06, 1.0])) < 0
    assert float(pen([2.0, 0.97, 0.6, 1.0])) < 0
    g = jax.grad(pen)(jnp.array([2.0, 1.02, 0.06, 1.0]))
    assert np.isfinite(g).all() and g[1] < 0 and g[0] == 0
    assert float(Model._range_penalty(_shim(names, ranges, penalty=False),
                                      jnp.array([2.0, 1.5, 0.06, 1.0]))) == 0.0
    assert _sampled_range_edges(["b1"], ranges)[0] is None


def test_collect_emulator_ranges_intersection():
    ns = types.SimpleNamespace
    A = ns(param_ranges={"ns": (0.93, 1.01), "H0": (52.0, 82.0)})
    B = ns(param_ranges={"ns": (0.9, 1.0), "w": (-3.0, 1.0)})
    C = ns(param_ranges={"H0": (55.0, 90.0)})
    like = ns(likelihood_pipeline=[ns(emulator=A), ns(emulators={0: B}),
                                   ns(sigma8_emu=C), ns()])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _collect_emulator_ranges({"l": like})
    assert r == {"ns": (0.93, 1.0), "H0": (55.0, 82.0), "w": (-3.0, 1.0)}
    like2 = ns(likelihood_pipeline=[ns(emulator=ns(param_ranges={}))])
    with pytest.warns(UserWarning, match="without param_ranges"):
        assert _collect_emulator_ranges({"l": like2}) == {}


def test_warning_emitted_and_not_emitted():
    from gholax.sampler.priors import Prior

    cfg = {
        "ns": {"prior": {"dist": "norm", "loc": 0.9649, "scale": 0.042}, "ref": 0.9649},
        "H0": {"prior": {"dist": "uniform", "min": 50.0, "max": 82.0}, "ref": 67.0},
        "As": {"prior": {"dist": "uniform", "min": 1.5, "max": 2.5}, "ref": 2.0},
    }
    prior = Prior(copy.deepcopy(cfg))
    ranges = {"ns": (0.93, 1.01), "H0": (52.0, 82.0), "As": (1.1, 3.1),
              "logmnu": (-2.0, -0.30103)}
    like = types.SimpleNamespace(fixed_params={"logmnu": -1.22, "NA": 0.0})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_prior_outside_ranges(prior, {"l": like}, ranges, 1e-3)
    msgs = [str(x.message) for x in w]
    assert any("prior for ns" in m for m in msgs)
    assert any("prior for H0" in m for m in msgs)
    assert not any("prior for As" in m for m in msgs)
    assert not any("fixed" in m for m in msgs)
    like_bad = types.SimpleNamespace(fixed_params={"mnu": 0.8})
    with pytest.warns(UserWarning, match="fixed mnu"):
        _warn_prior_outside_ranges(prior, {"l": like_bad}, ranges, 1e-3)


def test_spectral_equivalence_drops_wcdm_w_and_As_ranges():
    """wCDM emulators fed w_equiv/As_equiv must not constrain sampled w/As
    when a SpectralEquivalence module is in the pipeline."""
    ns = types.SimpleNamespace
    SpectralEquivalence = type("SpectralEquivalence", (), {})
    wcdm = ns(param_ranges={"w": (-1.56, -0.44), "As": (1.1, 3.1), "ns": (0.93, 1.01)})
    w0wa = ns(param_ranges={"w": (-3.0, 1.0), "wa": (-3.0, 2.0), "As": (1.1, 3.1)})
    like = ns(likelihood_pipeline=[SpectralEquivalence(), ns(emulator=wcdm), ns(emulator=w0wa)])
    r = _collect_emulator_ranges({"l": like})
    assert r["w"] == (-3.0, 1.0) and r["wa"] == (-3.0, 2.0) and r["ns"] == (0.93, 1.01)
    like_no_se = ns(likelihood_pipeline=[ns(emulator=wcdm), ns(emulator=w0wa)])
    assert _collect_emulator_ranges({"l": like_no_se})["w"] == (-1.56, -0.44)
