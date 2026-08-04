"""Smoke tests for NUTS/Minimize on the ('chains','model') device mesh.

Runs on CPU with 4 forced host devices. The full NUTS run uses the cheap RSD
model (replicated over the model axis); the model-sharded kernel path is
exercised with single NUTS steps on the 3x2pt model.
"""

import os
import sys

if "jax" not in sys.modules:
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    )

import blackjax
import jax
import numpy as np
import jax.numpy as jnp
import yaml


RSD_CONFIG = os.path.join(
    os.path.dirname(__file__), '..', 'example_configs', 'abcacus_dr1_rsd.yaml'
)


def _rsd_cfg(tmp_path, restart=False, n_chains=2, model_shards=2):
    with open(RSD_CONFIG) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cfg["sampler"] = {
        "NUTS": {
            "n_chains": n_chains,
            "model_shards": model_shards,
            "warmup_algorithm": "adaptive_window",
            "adaptive_warmup_stage_steps": 5,
            "adaptive_warmup_min_steps": 5,
            "adaptive_warmup_max_steps": 10,
            "n_steps_min": 4,
            "n_steps_incr": 2,
            "target_r_minus_one": 1e6,  # stop at n_steps_min
            "random_start": False,
            "restart": restart,
        }
    }
    cfg["output_file"] = str(tmp_path / "mesh_smoke.txt")
    return cfg


def test_nuts_mesh_smoke_and_restart(tmp_path):
    from gholax.util.model import Model
    from gholax.sampler import NUTS

    cfg = _rsd_cfg(tmp_path)
    model = Model(cfg)
    sampler = NUTS(cfg)
    samples, param_names = sampler.run(model, cfg["output_file"])

    n_params = len(model.param_names)
    assert samples.shape[0] == 2, samples.shape
    assert samples.shape[1] >= 4
    assert samples.shape[2] == n_params + 1
    assert param_names[-1] == "log_posterior"
    assert np.all(np.isfinite(np.asarray(samples)))

    for suffix in ("samples_chk.npy", "logposterior_chk.npy",
                   "nuts_warmup_parameters.json"):
        assert os.path.exists(f"{cfg['output_file']}.{suffix}"), suffix

    chk = np.load(f"{cfg['output_file']}.samples_chk.npy")
    assert chk.shape[0] == 2 and chk.shape[2] == n_params

    # restart: resumes from the checkpoint and appends
    cfg_r = _rsd_cfg(tmp_path, restart=True)
    model_r = Model(cfg_r)
    sampler_r = NUTS(cfg_r)
    samples_r, _ = sampler_r.run(model_r, cfg_r["output_file"])
    assert samples_r.shape[1] >= samples.shape[1]
    assert np.all(np.isfinite(np.asarray(samples_r)))


def test_mesh_loop_chain_independence():
    """Chains on different mesh shards must receive distinct keys and evolve
    independently (guards against key replication across the chain axis)."""
    from gholax.util.model import Model
    from gholax.util.distributed import build_mesh
    from gholax.sampler import NUTS

    with open(RSD_CONFIG) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg["sampler"] = {"NUTS": {}}
    model = Model(cfg)

    sampler = NUTS(cfg)
    sampler.mesh = build_mesh({"n_chains": 2, "model_shards": 2})
    model.sharded_log_posterior_scaled_params(sampler.mesh)

    dim = len(model.param_names)
    nuts = blackjax.nuts(
        model.log_posterior_scaled_params,
        inverse_mass_matrix=jnp.ones(dim),
        step_size=1e-4,  # small enough that every step accepts
    )
    positions = jnp.stack([0.1 * jnp.ones(dim), 0.11 * jnp.ones(dim)])
    states = sampler._map_chains(nuts.init)(positions)

    loop = sampler._make_mesh_inference_loop(nuts.step, 3, collect_info=True)
    sample_keys = jax.random.split(jax.random.key(3), 2)
    out, infos = loop(sample_keys, None, states, None)

    pos = np.asarray(out.position)
    assert np.all(np.asarray(infos.acceptance_rate) > 0.9)
    assert not np.allclose(pos[0, -1], positions[0]), "chain 0 did not move"
    assert not np.allclose(pos[1, -1], positions[1]), "chain 1 did not move"
    assert not np.array_equal(pos[0], pos[1]), "chains are identical"


def test_minimize_mesh(tmp_path):
    from gholax.util.model import Model
    from gholax.sampler import Minimize

    with open(RSD_CONFIG) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg["sampler"] = {
        "Minimize": {"n_chains": 2, "model_shards": 2, "random_start": False}
    }
    cfg["output_file"] = str(tmp_path / "mesh_min.txt")

    model = Model(cfg)
    sampler = Minimize(cfg)
    samples, param_names = sampler.run(model, cfg["output_file"])

    assert samples.shape == (2, 1, len(model.param_names) + 1)
    assert np.all(np.isfinite(np.asarray(samples)))
    assert os.path.exists(f"{cfg['output_file']}.minimization_results.json")


def _build_3x2_model():
    from gholax.util.model import Model

    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'example_configs', 'abacus_3x2_example.yaml'
    )
    with open(config_path) as f:
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
        dv_cfg = cfg["likelihood"][lname].get("data_vector", {})
        dv_file = dv_cfg.get("data_vector_info_filename")
        if dv_file is not None and not os.path.exists(dv_file):
            dv_cfg["data_vector_info_filename"] = os.path.join(
                local_data_dir, os.path.basename(dv_file)
            )
    return Model(cfg)


def test_nuts_kernel_step_model_sharded():
    """One NUTS step per chain with the model-sharded 3x2pt posterior inside
    the chain shard_map — the exact structure of the mesh inference loop."""
    from gholax.sampler import NUTS
    from gholax.util.distributed import build_mesh

    model = _build_3x2_model()
    mesh = build_mesh({"n_chains": 2, "model_shards": 2})

    sampler = NUTS({"sampler": {"NUTS": {}}})
    sampler.mesh = mesh

    # Activate sharding (side effect) and get the host-callable posterior.
    jlp = model.sharded_log_posterior_scaled_params(mesh)
    dim = len(model.param_names)
    positions = jnp.stack(
        [0.1 * jnp.ones(dim), 0.12 * jnp.ones(dim)]
    )

    nuts = blackjax.nuts(
        model.log_posterior_scaled_params,
        inverse_mass_matrix=jnp.ones(dim),
        step_size=1e-3,
    )
    states = sampler._map_chains(nuts.init)(positions)
    assert np.all(np.isfinite(np.asarray(states.logdensity)))

    # logdensity from the sharded init must match the wrapped posterior
    for i in range(2):
        lp = float(jlp(positions[i]))
        assert np.isclose(float(states.logdensity[i]), lp, atol=0.5)

    keys = jax.random.split(jax.random.key(7), 2)
    step = sampler._map_chains(nuts.step, chain_axes=(0, 0))
    new_states, infos = step(keys, states)
    assert np.all(np.isfinite(np.asarray(new_states.position)))
    assert np.all(np.isfinite(np.asarray(new_states.logdensity)))
