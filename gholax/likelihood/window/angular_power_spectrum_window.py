from ...util.likelihood_module import LikelihoodModule
from ...data_vector.two_point_spectrum import field_types
from jax.lax import scan
from copy import copy
import jax.numpy as jnp


class AngularPowerSpectrumWindow(LikelihoodModule):
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

    def compute(self, state, params_values):
        ell_p = jnp.arange(self.l_max)
        for t in self.spectrum_types:
            f = lambda carry, c_l: (carry, jnp.interp(ell_p, self.ell, c_l))
            _, c_l_p = scan(f, 0, state[f"{t}{self.cl_tag}"])
            c_l_p = jnp.einsum("ilm,im->il", self.cW[t][:, :, : self.l_max], c_l_p)
            state[f"{t}_obs"] = c_l_p

        return state
