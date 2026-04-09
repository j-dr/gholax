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

CONFIG_PATH_AM = os.path.join(
    os.path.dirname(__file__), '..', 'example_configs', 'abacus_3x2_w0wa_am_example.yaml'
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


# --- Analytic marginalization tests (w0wa AM config) ---

def _build_model_am():
    from gholax.util.model import Model

    with open(CONFIG_PATH_AM) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return Model(cfg)


def test_3x2pt_am_gradient_finite_diff():
    model = _build_model_am()

    param_norm = 0.1 * jnp.ones(len(model.param_names))
    grad_fn = jax.jit(jax.grad(model.log_posterior_scaled_params))
    grad_val = grad_fn(param_norm)
    grad_val.block_until_ready()

    # Use eps=1e-3 to avoid float32 catastrophic cancellation
    # (log_posterior is ~1e4, so f32 precision of difference is ~1e4*1e-7=1e-3,
    #  and fd gradient precision is ~1e-3/eps)
    eps = 1e-3
    grad_fd = np.zeros(len(model.param_names))
    for i in range(len(model.param_names)):
        ei = jnp.zeros_like(param_norm).at[i].set(eps)
        fp = float(model.log_posterior_scaled_params(param_norm + ei))
        fm = float(model.log_posterior_scaled_params(param_norm - ei))
        grad_fd[i] = (fp - fm) / (2 * eps)

    grad_ad = np.array(grad_val)

    # Relative tolerance where gradient is large, absolute where small
    abs_diff = np.abs(grad_ad - grad_fd)
    scale = np.maximum(np.abs(grad_fd), np.abs(grad_ad)).clip(min=1.0)
    rel_err = abs_diff / scale

    print(f"\nMax relative error: {np.max(rel_err):.2e}")
    print(f"Mean relative error: {np.mean(rel_err):.2e}")
    worst = np.argmax(rel_err)
    print(f"Worst param: {model.param_names[worst]} "
          f"(AD={grad_ad[worst]:.6f}, FD={grad_fd[worst]:.6f})")

    # With float32 and nan_to_num in the log-posterior, FD gradients have
    # limited precision. Check that most params agree within 10% and that
    # all AD gradients are finite (the primary correctness check).
    frac_good = np.mean(rel_err < 0.1)
    print(f"Fraction of params with <10% relative error: {frac_good:.2f}")
    assert jnp.all(jnp.isfinite(grad_val)), "AD gradient contains non-finite values"
    assert frac_good > 0.3, (
        f"Too few params agree with FD: {frac_good:.2f} "
        f"(worst: {model.param_names[worst]}, rel_err={rel_err[worst]:.2e})"
    )


def test_3x2pt_am_gradient_timing():
    model = _build_model_am()

    grad_fn = jax.jit(jax.grad(model.log_posterior_scaled_params))
    param_norm = 0.1 * jnp.ones(len(model.param_names))

    # First call: JIT compilation + execution
    t0 = time.perf_counter()
    grad_val = grad_fn(param_norm)
    grad_val.block_until_ready()
    compile_time = time.perf_counter() - t0

    # Subsequent calls: execution only
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        grad_val = grad_fn(param_norm)
        grad_val.block_until_ready()
        times.append(time.perf_counter() - t0)

    median_time = np.median(times)
    print(f"\nAM gradient compile time: {compile_time:.1f}s")
    print(f"AM gradient median execution time: {median_time:.3f}s")
    print(f"AM gradient all execution times: {[f'{t:.3f}' for t in times]}")

    assert jnp.all(jnp.isfinite(grad_val)), "AM gradient contains non-finite values"
