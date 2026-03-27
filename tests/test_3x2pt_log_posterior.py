"""Integration test: 3x2pt log_posterior at the reference parameter point."""

import os

import numpy as np
import jax.numpy as jnp
import yaml


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'example_configs', 'abacus_3x2_example.yaml'
)

REFERENCE_LOG_POSTERIOR = -2844.9421386719


def test_3x2pt_log_posterior_at_reference():
    from gholax.util.model import Model

    with open(CONFIG_PATH) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    model = Model(cfg)

    ref_params = model.prior.get_reference_point()
    param_array = jnp.array([ref_params[name] for name in model.param_names])

    log_post = float(model.log_posterior(param_array))

    assert np.isclose(log_post, REFERENCE_LOG_POSTERIOR, atol=0.01), (
        f"log_posterior = {log_post}, expected {REFERENCE_LOG_POSTERIOR}"
    )
