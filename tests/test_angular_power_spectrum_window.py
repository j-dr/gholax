"""Regression tests for the fixed angular-power-spectrum window transform."""

import jax
import jax.numpy as jnp
import numpy as np

from gholax.likelihood.window.angular_power_spectrum_window import (
    AngularPowerSpectrumWindow,
)


class _DataVector:
    """Minimal data vector carrying one two-bin set of bandpower windows."""

    def __init__(self):
        window = {
            "0_0": [[0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0],
                      [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.0]],
            "0_1": [[0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0],
                      [0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1]],
            "1_0": [[0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0],
                      [0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1]],
            "1_1": [[0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.0]],
        }
        self.cW = {"c_kk": {key: np.asarray(value, dtype=np.float32)
                              for key, value in window.items()}}


def test_effective_window_matches_interpolate_then_window():
    info = {"c_kk": {"bins0": [0, 1], "bins1": [0, 1], "use_cross": True}}
    window = AngularPowerSpectrumWindow(
        _DataVector(), ["c_kk"], info, n_ell=5, l_max=8
    )
    cl = jnp.array(
        [[1.0, 3.0, 2.0, 6.0, 4.0], [2.0, 1.0, 4.0, 3.0, 5.0],
         [5.0, 2.0, 1.0, 4.0, 3.0], [4.0, 3.0, 2.0, 1.0, 6.0]]
    )

    result = window.compute({"c_kk_mbias": cl}, {})["c_kk_obs"]
    interpolated = jax.vmap(
        lambda values: jnp.interp(jnp.arange(window.l_max), window.ell, values)
    )(cl)
    expected = jnp.einsum(
        "ilm,im->il", window.cW["c_kk"][:, :, : window.l_max], interpolated
    )

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        jax.jit(lambda x: window.compute({"c_kk_mbias": x}, {})["c_kk_obs"])(cl),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )
