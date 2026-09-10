"""SeedingConfig: legacy-key translation, cross-key defaults, validation."""

import warnings

import pytest

from gholax.sampler import seeding as seeding_module
from gholax.sampler.seeding import (
    LEGACY_KEYS,
    SAMPLER_SEEDING_DEFAULTS,
    SeedingConfig,
)

# SeedingConfig field -> the sampler attribute it replaces.
FIELD_TO_ATTR = {v: k for k, v in LEGACY_KEYS.items()}


def _nuts(scfg):
    from gholax.sampler import NUTS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return NUTS({"sampler": {"NUTS": dict(scfg)}})


def _load(c, sampler="NUTS"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return SeedingConfig.from_sampler_config(c, sampler=sampler)


LEGACY_CONFIGS = [
    {},
    {"minimize_and_sample": True, "minimize_n_starts": 8,
     "minimize_start_scale": 1.0},
    {"pathfinder_init": True, "pathfinder_resample": True,
     "pathfinder_n_paths": 3, "pathfinder_elbo_samples": 10,
     "pathfinder_maxiter": 50, "pathfinder_maxcor": 5,
     "pathfinder_start_scale": 0.25, "mass_matrix_init": "fisher_seeds"},
    {"pathfinder_init": False, "mass_matrix_init": "hessian",
     "minimize_and_sample": True},
]


@pytest.mark.parametrize("scfg", LEGACY_CONFIGS)
def test_config_reproduces_legacy_attributes(scfg):
    """The loader must reproduce every attribute the samplers set today."""
    nuts = _nuts(scfg)
    for field, attr in FIELD_TO_ATTR.items():
        assert getattr(nuts.seeding_config, field) == getattr(nuts, attr), field


def test_nested_block_equals_legacy_form():
    legacy = LEGACY_CONFIGS[2]
    nested = {"seeding": {LEGACY_KEYS[k]: v for k, v in legacy.items()}}
    assert _load(legacy) == _load(nested)


@pytest.mark.parametrize("key,field", sorted(LEGACY_KEYS.items()))
def test_every_legacy_key_warns_and_maps(key, field):
    value = {
        "mass_matrix_init": "hessian",
        "minimize_and_sample": True,
        "pathfinder_init": True,
        "pathfinder_resample": True,
        "start_scale": 0.25,
        "pathfinder_start_scale": 0.25,
    }.get(field, 7)
    seeding_module._WARNED.discard(("NUTS", key))
    with pytest.warns(DeprecationWarning, match=key):
        cfg = SeedingConfig.from_sampler_config({key: value}, sampler="NUTS")
    assert getattr(cfg, field) == value


def test_nested_wins_over_legacy_and_warns():
    with pytest.warns(UserWarning, match="also set"):
        cfg = SeedingConfig.from_sampler_config(
            {"minimize_n_starts": 5, "seeding": {"n_starts": 9}}, sampler="NUTS"
        )
    assert cfg.n_starts == 9


def test_unknown_nested_key_raises():
    with pytest.raises(ValueError, match="minimize_n_starts"):
        _load({"seeding": {"minimize_n_starts": 5}})


def test_internal_field_is_not_user_settable():
    with pytest.raises(ValueError, match="metric_fallback"):
        _load({"seeding": {"metric_fallback": "ones"}})


@pytest.mark.parametrize("nested", [False, True])
def test_cross_key_defaults(nested):
    """pathfinder_n_paths/start_scale follow their siblings unless set."""
    c = ({"seeding": {"n_starts": 7, "start_scale": 1.5}} if nested
         else {"minimize_n_starts": 7, "minimize_start_scale": 1.5})
    cfg = _load(c)
    assert cfg.pathfinder_n_paths == 7
    assert cfg.pathfinder_start_scale == 1.5

    explicit = dict(c)
    if nested:
        explicit["seeding"] = dict(c["seeding"], pathfinder_n_paths=2,
                                   pathfinder_start_scale=0.1)
    else:
        explicit.update(pathfinder_n_paths=2, pathfinder_start_scale=0.1)
    cfg = _load(explicit)
    assert (cfg.pathfinder_n_paths, cfg.pathfinder_start_scale) == (2, 0.1)


def test_mass_matrix_init_derivation():
    assert _load({}).mass_matrix_init == "pathfinder"          # NUTS default
    assert _load({"pathfinder_init": False}).mass_matrix_init == "hessian_dense"
    assert _load({}, sampler="MCLMC").mass_matrix_init == "ones"
    assert _load({"mass_matrix_init": "hessian"}).mass_matrix_init == "hessian"


@pytest.mark.parametrize("value", ["pathfinder", "fisher_seeds"])
def test_metric_requiring_pathfinder_raises(value):
    with pytest.raises(ValueError, match=value):
        _load({"mass_matrix_init": value, "pathfinder_init": False})


def test_pathfinder_resample_defaults_false():
    """base.py's getattr default was True, the NUTS config default False.

    The True branch was unreachable (only NUTS ever ran Pathfinder, and it
    always set the attribute); False is the behaviour-preserving choice.
    """
    assert SeedingConfig().pathfinder_resample is False


@pytest.mark.parametrize("sampler", sorted(SAMPLER_SEEDING_DEFAULTS))
def test_per_sampler_defaults(sampler):
    cfg = _load({}, sampler=sampler)
    for field, value in SAMPLER_SEEDING_DEFAULTS[sampler].items():
        assert getattr(cfg, field) == value
    assert cfg.minimize_and_sample is (sampler in ("MCLMC", "Minimize"))


@pytest.mark.parametrize(
    "scfg,match",
    [
        ({"minimize_n_starts": 0}, "minimize_n_starts"),
        ({"pathfinder_n_paths": 0}, "pathfinder_n_paths"),
        ({"pathfinder_elbo_samples": 0}, "pathfinder_elbo_samples"),
        ({"pathfinder_maxcor": 0}, "pathfinder_maxcor"),
    ],
)
def test_validation(scfg, match):
    with pytest.raises(ValueError, match=match):
        _load(scfg)
