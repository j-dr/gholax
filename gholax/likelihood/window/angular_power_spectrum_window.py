from ...util.likelihood_module import LikelihoodModule
from ...data_vector.two_point_spectrum import field_types
from copy import copy
import jax.numpy as jnp
import numpy as np


class AngularPowerSpectrumWindow(LikelihoodModule):
    """Convolve theory C_ell with bandpower window functions.

    Interpolates theory C_ell to integer ell and applies the
    coupling/window matrix to produce observed bandpowers.
    """

    shards_pair_axis = True

    def __init__(
        self,
        observed_data_vector,
        spectrum_types,
        spectrum_info,
        n_ell=200,
        l_max=3001,
        **config,
    ):
        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.n_ell = n_ell
        self.l_max = l_max
        self.cl_tag = config.get("cl_tag", "_mbias")

        self.ell = jnp.logspace(1, jnp.log10(self.l_max), self.n_ell)

        self.all_spectra = {}
        self.output_requirements = {}

        for t in self.spectrum_types:
            self.all_spectra[t] = []
            for ii, i in enumerate(spectrum_info[t]["bins0"]):
                if spectrum_info[t]["use_cross"]:
                    if field_types[t][0] == field_types[t][1]:
                        bins1 = self.spectrum_info[t]["bins1"][ii:]
                    else:
                        bins1 = self.spectrum_info[t]["bins1"][:]
                    for j in bins1:
                        self.all_spectra[t].append((i, j))
                        self.output_requirements[f"{t}_{i}_{j}_obs"] = [
                            f"{t}_{i}_{j}{self.cl_tag}"
                        ]
                else:
                    self.all_spectra[t].append((i, i))
                    self.output_requirements[f"{t}_{i}_{i}_obs"] = [
                        f"{t}_{i}_{i}{self.cl_tag}"
                    ]

        self.cW = copy(self.observed_data_vector.cW)
        # Both the theory ell grid and the bandpower windows are fixed for the
        # lifetime of a likelihood.  Compose their two linear operations once:
        # C_ell(n_ell) -> C_ell(l_max) -> C_b.  This removes the l_max-long
        # interpolated spectrum from every posterior evaluation.
        interp = self._integer_ell_interpolator()
        self.cW_effective = {}
        for t in self.cW:
            bins0 = jnp.unique(
                jnp.array([int(s.split("_")[0]) for s in self.cW[t]])
            ).sort()
            bins1 = jnp.unique(
                jnp.array([int(s.split("_")[1]) for s in self.cW[t]])
            ).sort()
            n_bins0 = len(bins0)
            n_bins1 = len(bins1)
            if self.spectrum_info[t]["use_cross"]:
                cW_t = jnp.zeros(
                    (
                        n_bins0 * n_bins1,
                        self.cW[t][f"{bins0[0]}_{bins1[0]}"].shape[0],
                        self.cW[t][f"{bins0[0]}_{bins1[0]}"].shape[1],
                    )
                )
                counter = 0
                for i in bins0:
                    for j in bins1:
                        try:
                            cW_t = cW_t.at[counter].set(self.cW[t][f"{i}_{j}"])
                        except:
                            cW_t = cW_t.at[counter].set(self.cW[t][f"{j}_{i}"])
                        counter += 1
            else:
                cW_t = jnp.zeros(
                    (
                        n_bins0,
                        self.cW[t][f"{bins0[0]}_{bins1[0]}"].shape[0],
                        self.cW[t][f"{bins0[0]}_{bins1[0]}"].shape[1],
                    )
                )
                for i in bins0:
                    cW_t = cW_t.at[i].set(self.cW[t][f"{i}_{i}"])

            self.cW[t] = cW_t
            self.cW_effective[t] = jnp.einsum(
                "ibl,ln->ibn", cW_t[:, :, : self.l_max], interp
            )

    def _integer_ell_interpolator(self):
        """Fixed map matching ``jnp.interp(arange(l_max), ell, values)``."""
        x = np.asarray(self.ell)
        q = np.arange(self.l_max, dtype=x.dtype)
        n = len(x)
        right = np.searchsorted(x, q, side="right")
        lo = np.clip(right - 1, 0, n - 2)
        hi = lo + 1
        frac = (q - x[lo]) / (x[hi] - x[lo])

        # jnp.interp clamps, rather than extrapolates, outside the knot range.
        below = q <= x[0]
        above = q >= x[-1]
        frac[below] = 0.0
        lo[below] = hi[below] = 0
        frac[above] = 0.0
        lo[above] = hi[above] = n - 1

        interpolation = np.zeros((self.l_max, n), dtype=x.dtype)
        rows = np.arange(self.l_max)
        interpolation[rows, lo] += 1.0 - frac
        interpolation[rows, hi] += frac
        return jnp.asarray(interpolation)

    def compute(self, state, params_values):
        """Convolve theory C_ell with window matrices and write observed bandpowers to state."""
        for t in self.spectrum_types:
            cW_t = self.cW_effective[t]
            if self.model_sharding is not None:
                # C_ell arrives as a local block of the pair axis; use the
                # matching block of the window matrices.
                cW_t = self.model_sharding.slice_local(cW_t)
            state[f"{t}_obs"] = jnp.einsum(
                "iln,in->il", cW_t, state[f"{t}{self.cl_tag}"]
            )

        return state
