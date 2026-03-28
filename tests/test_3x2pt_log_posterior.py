"""Integration tests for the 3x2pt likelihood pipeline.

Evaluates at normalized params = 0.1 (0.1 prior-sigma from reference)
so that nuisance parameters are non-zero and all model contributions are
exercised.
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

REFERENCE_LOG_POSTERIOR = -10293.5488281250


def _build_model():
    from gholax.util.model import Model

    with open(CONFIG_PATH) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return Model(cfg)


def test_3x2pt_log_posterior():
    model = _build_model()

    param_norm = 0.1 * jnp.ones(len(model.param_names))
    log_post = float(model.log_posterior_scaled_params(param_norm))

    assert np.isclose(log_post, REFERENCE_LOG_POSTERIOR, atol=0.01), (
        f"log_posterior = {log_post}, expected {REFERENCE_LOG_POSTERIOR}"
    )


def test_3x2pt_gradient_timing():
    model = _build_model()

    grad_fn = jax.jit(jax.grad(model.log_posterior_scaled_params))
    param_norm = 0.1 * jnp.ones(len(model.param_names))

    # First call: JIT compilation + execution
    t0 = time.perf_counter()
    grad_val = grad_fn(param_norm)
    grad_val.block_until_ready()
    compile_time = time.perf_counter() - t0

    # Second call: execution only (already compiled)
    t0 = time.perf_counter()
    grad_val = grad_fn(param_norm)
    grad_val.block_until_ready()
    exec_time = time.perf_counter() - t0

    print(f"\nGradient compile time: {compile_time:.1f}s")
    print(f"Gradient execution time: {exec_time:.3f}s")

    assert jnp.all(jnp.isfinite(grad_val)), "Gradient contains non-finite values"
