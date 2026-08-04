"""Tests for the per-likelihood gradient_checkpointing option.

With gradient_checkpointing enabled, each pipeline module is wrapped in
jax.checkpoint (remat), so the backward pass recomputes module intermediates
instead of storing them. The posterior value and gradient must be unchanged;
only memory/compute trade off. Evaluates at normalized params = 0.1, matching
tests/test_3x2pt_log_posterior.py.
"""

import os
import time

import jax
import numpy as np
import jax.numpy as jnp
import yaml


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'example_configs', 'abacus_3x2_example.yaml'
)


def _build_model(gradient_checkpointing):
    from gholax.util.model import Model

    with open(CONFIG_PATH) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # The IA section at HEAD (per_source_bin_spline, spline_N=5) is
    # inconsistent with the config's params list (c_s_0..c_s_2 etc., written
    # for spline/N=3); restore the settings the test reference (126c073) was
    # recorded with so the model builds.
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
        # Fall back to the repo-local copy of the data vector when the
        # configured (NERSC) path is unavailable.
        dv_cfg = cfg["likelihood"][lname].get("data_vector", {})
        dv_file = dv_cfg.get("data_vector_info_filename")
        if dv_file is not None and not os.path.exists(dv_file):
            dv_cfg["data_vector_info_filename"] = os.path.join(
                local_data_dir, os.path.basename(dv_file)
            )

    return Model(cfg)


def test_checkpointing_flag_is_read():
    model = _build_model(True)
    for lname in model.likelihoods:
        assert model.likelihoods[lname].gradient_checkpointing is True

    model = _build_model(False)
    for lname in model.likelihoods:
        assert model.likelihoods[lname].gradient_checkpointing is False


def test_checkpointing_preserves_value_and_gradient():
    model_off = _build_model(False)
    model_on = _build_model(True)

    param_norm = 0.1 * jnp.ones(len(model_off.param_names))

    # Compare the jitted posterior (what the samplers evaluate). Remat re-runs
    # identical ops, but it can change XLA fusion decisions, so allow the same
    # float32 reduction-reordering spread the reference test absorbs (~0.5 on
    # |logp| ~ 2e4).
    logp_off = float(jax.jit(model_off.log_posterior_scaled_params)(param_norm))
    logp_on = float(jax.jit(model_on.log_posterior_scaled_params)(param_norm))
    assert np.isclose(logp_on, logp_off, atol=0.5), (
        f"log_posterior changed with checkpointing: {logp_on} vs {logp_off}"
    )

    def timed_grad(model):
        grad_fn = jax.jit(jax.grad(model.log_posterior_scaled_params))
        t0 = time.perf_counter()
        grad_val = grad_fn(param_norm)
        grad_val.block_until_ready()
        compile_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        grad_val = grad_fn(param_norm)
        grad_val.block_until_ready()
        exec_time = time.perf_counter() - t0
        return np.array(grad_val), compile_time, exec_time

    grad_off, compile_off, exec_off = timed_grad(model_off)
    grad_on, compile_on, exec_on = timed_grad(model_on)

    print(f"\nGradient compile time: off={compile_off:.1f}s on={compile_on:.1f}s")
    print(f"Gradient execution time: off={exec_off:.3f}s on={exec_on:.3f}s")

    assert np.all(np.isfinite(grad_on)), "Checkpointed gradient has non-finite values"

    abs_diff = np.abs(grad_on - grad_off)
    scale = np.maximum(np.abs(grad_on), np.abs(grad_off)).clip(min=1.0)
    rel_err = abs_diff / scale
    worst = np.argmax(rel_err)
    print(f"Max gradient relative error: {rel_err[worst]:.2e} "
          f"({model_off.param_names[worst]})")
    assert np.max(rel_err) < 1e-2, (
        f"Gradient changed with checkpointing: {model_off.param_names[worst]} "
        f"off={grad_off[worst]:.6f} on={grad_on[worst]:.6f}"
    )
