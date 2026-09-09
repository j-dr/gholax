"""WarmupConfig: legacy-key translation, presence sentinels, validation."""

import warnings

import pytest

from gholax.sampler import warmup as warmup_module
from gholax.sampler.warmup import (
    LEGACY_KEYS,
    SAMPLER_WARMUP_DEFAULTS,
    WarmupConfig,
)

# WarmupConfig field -> the NUTS attribute it replaces.
FIELD_TO_ATTR = {
    "algorithm": "warmup_algorithm",
    "n_steps": "n_steps_warmup",
    "init_file": "warmup_init_file",
    "restart": "restart",
    "diagonal_mass_matrix": "diagonal_mass_matrix",
    "step_size_init": "step_size_init",
    "step_size_search": "step_size_search",
    "target_acceptance_rate": "target_acceptance_rate",
    "window_steps": "pooled_window_steps",
    "max_steps": "pooled_window_max_steps",
    "max_window": "pooled_window_max_window",
    "terminal_steps": "pooled_window_terminal_steps",
    "consecutive_windows": "pooled_window_consecutive_windows",
    "rtol_mass": "pooled_window_rtol_mass",
    "rtol_step": "pooled_window_rtol_step",
    "mass_stat": "pooled_window_mass_stat",
    "mixing_rhat": "pooled_window_mixing_rhat",
    "mixing_quantile": "pooled_window_mixing_quantile",
    "min_tail_steps": "pooled_window_min_tail_steps",
    "allow_unconverged": "pooled_window_allow_unconverged",
    "metric_estimator": "pooled_window_metric_estimator",
    "fisher_cutoff": "pooled_window_fisher_cutoff",
    "fisher_reg": "pooled_window_fisher_reg",
    "dense_rank": "pooled_window_dense_rank",
    "dense_update": "pooled_window_dense_update",
    "max_doublings": "pooled_window_max_doublings",
    "sampling_max_num_doublings": "max_num_doublings",
    "sampling_depth_auto": "max_num_doublings_auto",
    "depth_cap_quantile": "depth_cap_quantile",
    "depth_cap_margin": "depth_cap_margin",
    "depth_cap_margin_saturated": "depth_cap_margin_saturated",
    "stage_steps": "adaptive_warmup_stage_steps",
    "adaptive_max_steps": "adaptive_warmup_max_steps",
    "adaptive_min_steps": "adaptive_warmup_min_steps",
    "adaptive_rtol_mass": "adaptive_warmup_rtol_mass",
    "adaptive_rtol_step": "adaptive_warmup_rtol_step",
}


def _nuts(scfg):
    from gholax.sampler import NUTS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return NUTS({"sampler": {"NUTS": dict(scfg)}})


def _load(c, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return WarmupConfig.from_sampler_config(c, sampler="NUTS", **kw)


LEGACY_CONFIGS = [
    {},
    {"warmup_algorithm": "adaptive_window", "n_steps_warmup": 300},
    {
        "warmup_algorithm": "pooled_window",
        "pooled_window_max_window": 100,
        "adaptive_warmup_rtol_mass": 0.8,
        "adaptive_warmup_rtol_step": 0.3,
        "pooled_window_consecutive_windows": 1,
        "pooled_window_allow_unconverged": True,
    },
    {
        "pooled_window_mass_stat": "max_diag",
        "pooled_window_metric_estimator": "fisher",
        "pooled_window_dense_rank": "auto",
        "pooled_window_dense_update": False,
        "step_size_init": 0.02,
        "max_num_doublings": 6,
        "pooled_window_terminal_steps": 40,
    },
]


@pytest.mark.parametrize("scfg", LEGACY_CONFIGS)
def test_config_reproduces_legacy_attributes(scfg):
    """The loader must reproduce every attribute NUTS.__init__ sets today."""
    nuts = _nuts(scfg)
    for field, attr in FIELD_TO_ATTR.items():
        assert getattr(nuts.warmup_config, field) == getattr(nuts, attr), field


def test_nested_block_equals_legacy_form():
    legacy = LEGACY_CONFIGS[3]
    nested = {"warmup": {LEGACY_KEYS[k]: v for k, v in legacy.items()}}
    assert _load(legacy) == _load(nested)


@pytest.mark.parametrize("key,field", sorted(LEGACY_KEYS.items()))
def test_every_legacy_key_warns_and_maps(key, field):
    if field in ("algorithm", "mass_stat", "metric_estimator", "init_file"):
        value = {
            "algorithm": "adaptive_window",
            "mass_stat": "max_diag",
            "metric_estimator": "fisher",
            "init_file": "prev.json",
        }[field]
    elif field in ("dense_rank",):
        value = 3
    elif field in ("diagonal_mass_matrix", "allow_unconverged", "dense_update"):
        value = False
    elif field in ("mixing_rhat", "mixing_quantile"):
        value = 1.0
    elif isinstance(getattr(WarmupConfig(), field), float):
        value = 0.42
    else:
        value = 7
    warmup_module._WARNED.discard(("NUTS", key))
    with pytest.warns(DeprecationWarning, match=key):
        cfg = WarmupConfig.from_sampler_config({key: value}, sampler="NUTS")
    assert getattr(cfg, field) == value


def test_nested_wins_over_legacy_and_warns():
    with pytest.warns(UserWarning, match="also set"):
        cfg = WarmupConfig.from_sampler_config(
            {"pooled_window_steps": 5, "warmup": {"window_steps": 9}},
            sampler="NUTS",
        )
    assert cfg.window_steps == 9


def test_unknown_nested_key_raises():
    with pytest.raises(ValueError, match="pooled_window_steps"):
        _load({"warmup": {"pooled_window_steps": 5}})


def test_tree_depth_key_rejected_for_mclmc():
    with pytest.raises(ValueError, match="NUTS only"):
        WarmupConfig.from_sampler_config(
            {"warmup": {"depth_cap_quantile": 0.5}},
            sampler="MCLMC",
            tree_depth=False,
        )


def test_mclmc_defaults():
    cfg = WarmupConfig.from_sampler_config({}, sampler="MCLMC", tree_depth=False)
    for field, value in SAMPLER_WARMUP_DEFAULTS["MCLMC"].items():
        assert getattr(cfg, field) == value
    # per-sampler defaults must not count as explicit user settings
    assert cfg.step_size_search is True


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "key,flag",
    [("step_size_init", "step_size_search"),
     ("max_num_doublings", "sampling_depth_auto")],
)
def test_presence_sentinels(key, flag, nested):
    """Absence, not value, drives the derived flags - in both spellings."""
    assert getattr(_load({}), flag) is True
    value = WarmupConfig()  # defaults; setting them explicitly must still flip
    explicit = getattr(value, LEGACY_KEYS[key])
    c = ({"warmup": {LEGACY_KEYS[key]: explicit}} if nested else {key: explicit})
    assert getattr(_load(c), flag) is False


@pytest.mark.parametrize("mass_stat", ["rms_diag", "max_diag"])
@pytest.mark.parametrize("pooled", [None, 0.7])
@pytest.mark.parametrize("adaptive", [None, 0.8])
def test_rtol_precedence(mass_stat, pooled, adaptive):
    c = {"pooled_window_mass_stat": mass_stat}
    if pooled is not None:
        c["pooled_window_rtol_mass"] = pooled
    if adaptive is not None:
        c["adaptive_warmup_rtol_mass"] = adaptive
    expected = (
        pooled
        if pooled is not None
        else (adaptive if adaptive is not None else 0.05)
        if (adaptive is not None or mass_stat == "max_diag")
        else 0.1
    )
    assert _load(c).rtol_mass == expected
    assert _nuts(c).pooled_window_rtol_mass == expected


@pytest.mark.parametrize(
    "scfg,match",
    [
        ({"warmup_algorithm": "pooled_windows"}, "warmup_algorithm"),
        ({"pooled_window_mixing_quantile": 0.0}, "mixing_quantile"),
        ({"pooled_window_mixing_rhat": 0.5}, "mixing_rhat"),
        ({"pooled_window_min_tail_steps": 1}, "min_tail_steps"),
        ({"pooled_window_consecutive_windows": 0}, "consecutive_windows"),
        ({"pooled_window_mass_stat": "nope"}, "mass_stat"),
        ({"pooled_window_metric_estimator": "nope"}, "metric_estimator"),
        ({"pooled_window_dense_rank": -1}, "dense_rank"),
        ({"pooled_window_terminal_steps": 0}, "terminal_steps"),
    ],
)
def test_validation_preserved(scfg, match):
    with pytest.raises(ValueError, match=match):
        _load(scfg)
