"""Tests for model-axis (bin-pair) sharding of the Nx2PT likelihood.

With model_shards > 1, the projection stage (Limber -> window) partitions its
bin-pair axis across the 'model' mesh axis inside jax.shard_map; the posterior
value and gradient must match the unsharded evaluation. Runs on CPU with
forced host devices.
"""

import os

if "jax" not in __import__("sys").modules:
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    )

import jax
import numpy as np
import jax.numpy as jnp
import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'example_configs', 'abacus_3x2_example.yaml'
)


def _build_model(gradient_checkpointing=False):
    """Build the 3x2pt model with the same config fixups as
    test_gradient_checkpointing.py: restore the IA settings the reference was
    recorded with, and fall back to the repo-local data vector copy."""
    from gholax.util.model import Model

    with open(CONFIG_PATH) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cfg["theory"]["RealSpaceIAExpansion"] = {
        "z_evolution_model": "spline",
        "spline_N": 3,
        "spline_Delta": 0.6,
        "scale_by_s8z": True,
        "include_magnification_x_ia": True,
        "no_ia": True,
    }

    local_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    for lname in cfg["likelihood"]:
        cfg["likelihood"][lname]["gradient_checkpointing"] = gradient_checkpointing
        dv_cfg = cfg["likelihood"][lname].get("data_vector", {})
        dv_file = dv_cfg.get("data_vector_info_filename")
        if dv_file is not None and not os.path.exists(dv_file):
            dv_cfg["data_vector_info_filename"] = os.path.join(
                local_data_dir, os.path.basename(dv_file)
            )

    return Model(cfg)


def _mesh(n_chains, model_shards):
    from gholax.util.distributed import build_mesh

    return build_mesh({"n_chains": n_chains, "model_shards": model_shards})


def test_sharded_matches_unsharded():
    model = _build_model()
    param_norm = 0.1 * jnp.ones(len(model.param_names))

    f_ref = model.sharded_log_posterior_scaled_params(_mesh(4, 1))
    logp_ref = float(f_ref(param_norm))
    g_ref = np.array(jax.jit(jax.grad(f_ref))(param_norm))

    for shards in (2, 4):
        model_s = _build_model()
        f_s = model_s.sharded_log_posterior_scaled_params(_mesh(4 // shards, shards))
        logp_s = float(f_s(param_norm))
        assert np.isclose(logp_s, logp_ref, atol=0.5), (
            f"model_shards={shards}: logp {logp_s} vs {logp_ref}"
        )

        g_s = np.array(jax.jit(jax.grad(f_s))(param_norm))
        assert np.all(np.isfinite(g_s))
        rel = np.abs(g_s - g_ref) / np.maximum(np.abs(g_s), np.abs(g_ref)).clip(min=1.0)
        worst = np.argmax(rel)
        print(f"model_shards={shards}: max grad rel err {rel[worst]:.2e} "
              f"({model.param_names[worst]})")
        assert np.max(rel) < 1e-2, (
            f"model_shards={shards}: grad mismatch at {model.param_names[worst]} "
            f"sharded={g_s[worst]:.6f} ref={g_ref[worst]:.6f}"
        )


def test_sharded_with_gradient_checkpointing():
    model = _build_model()
    param_norm = 0.1 * jnp.ones(len(model.param_names))
    f_ref = model.sharded_log_posterior_scaled_params(_mesh(4, 1))
    logp_ref = float(f_ref(param_norm))
    g_ref = np.array(jax.jit(jax.grad(f_ref))(param_norm))

    model_s = _build_model(gradient_checkpointing=True)
    f_s = model_s.sharded_log_posterior_scaled_params(_mesh(2, 2))
    logp_s = float(f_s(param_norm))
    assert np.isclose(logp_s, logp_ref, atol=0.5)

    g_s = np.array(jax.jit(jax.grad(f_s))(param_norm))
    assert np.all(np.isfinite(g_s))
    rel = np.abs(g_s - g_ref) / np.maximum(np.abs(g_s), np.abs(g_ref)).clip(min=1.0)
    print(f"ckpt+shards=2: max grad rel err {np.max(rel):.2e}")
    assert np.max(rel) < 1e-2


def test_rsd_replicated_on_mesh():
    """RSDPK has no sharding modules; it must run replicated on any mesh."""
    from gholax.util.model import Model

    cfg_path = os.path.join(
        os.path.dirname(__file__), '..', 'example_configs', 'abcacus_dr1_rsd.yaml'
    )
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    model = Model(cfg)
    param_norm = 0.1 * jnp.ones(len(model.param_names))
    logp_ref = float(jax.jit(model.log_posterior_scaled_params)(param_norm))

    f_s = model.sharded_log_posterior_scaled_params(_mesh(2, 2))
    logp_s = float(f_s(param_norm))
    assert np.isclose(logp_s, logp_ref, atol=0.5), f"{logp_s} vs {logp_ref}"
