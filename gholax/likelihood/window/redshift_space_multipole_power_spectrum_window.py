from ...util.likelihood_module import LikelihoodModule
from ...data_vector.redshift_space_multipoles import field_types
from jax.lax import scan
from copy import copy
import jax.numpy as jnp


class RedshiftSpaceMultipolePowerSpectrumWindow(LikelihoodModule):
    def __init__(
        self,
        observed_data_vector,
        spectrum_types,
        spectrum_info,
        kmin=1e-3,
        kmax=0.6,
        nk=200,
        n_ell=3,
        **config,
    ):
        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.logk = jnp.linspace(jnp.log10(kmin), jnp.log10(kmax), nk)
        self.nk = nk
        self.n_ell = n_ell
        self.pl_tag = config.get("pl_tag", "")

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
                            f"{t}_{i}_{j}{self.pl_tag}"
                        ]
                else:
                    self.all_spectra[t].append((i, i))
                    self.output_requirements[f"{t}_{i}_{i}_obs"] = [
                        f"{t}_{i}_{i}{self.pl_tag}"
                    ]

        self.W = copy(self.observed_data_vector.W)

        bins0 = jnp.unique(jnp.array([int(s.split("_")[0]) for s in self.W])).sort()
        bins1 = jnp.unique(jnp.array([int(s.split("_")[1]) for s in self.W])).sort()
        n_bins0 = len(bins0)
        n_bins1 = len(bins1)
        if self.spectrum_info["p_gg_ell"]["use_cross"]:
            W_t = jnp.zeros(
                (
                    n_bins0 * n_bins1,
                    self.W[f"{bins0[0]}_{bins1[0]}"].shape[0],
                    self.W[f"{bins0[0]}_{bins1[0]}"].shape[1],
                )
            )
            counter = 0
            for i in bins0:
                for j in bins1:
                    try:
                        W_t = W_t.at[counter].set(self.W[f"{i}_{j}"])
                    except:
                        W_t = W_t.at[counter].set(self.W[f"{j}_{i}"])
                    counter += 1
        else:
            W_t = jnp.zeros(
                (
                    n_bins0,
                    self.W[f"{bins0[0]}_{bins1[0]}"].shape[0],
                    self.W[f"{bins0[0]}_{bins1[0]}"].shape[1],
                )
            )
            for i in bins0:
                W_t = W_t.at[i].set(self.W[f"{i}_{i}"])

        self.W = W_t

    def compute(self, state, params_values):
        for t in self.spectrum_types:

            def f(carry, p_l):
                p_l_kth = jnp.zeros(
                    (self.n_ell, self.observed_data_vector.kth.shape[0])
                )
                for i in range(self.n_ell):
                    p_l_kth = p_l_kth.at[i, ...].set(
                        jnp.interp(self.observed_data_vector.kth, self.k, p_l[i])
                    )
                return carry, p_l_kth.flatten()

            _, p_l_kth = scan(f, 0, state[f"{t}{self.pl_tag}"])
            p_l_kth = jnp.einsum("ito,it->io", self.W[:, :, :], p_l_kth)
            state[f"{t}_obs"] = p_l_kth.reshape(
                -1, self.n_ell, self.observed_data_vector.ko_eff.shape[0]
            )

        return state
