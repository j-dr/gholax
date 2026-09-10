from ...util.likelihood_module import LikelihoodModule
from ...data_vector.two_point_spectrum import field_types
import jax.numpy as jnp
import numpy as np


class ShearMultiplicativeBias(LikelihoodModule):
    """Apply shear multiplicative bias correction (1+m) to angular power spectra.

    Multiplies C_ell by (1+m_i)(1+m_j) for each bin pair involving shear fields.
    """

    shards_pair_axis = True

    def __init__(self, observed_data_vector, spectrum_types, spectrum_info, **config):
        """Initialize the multiplicative bias module.

        Args:
            observed_data_vector: DataVector instance.
            spectrum_types: List of spectrum type strings.
            spectrum_info: Dict of spectrum configuration info.
            **config: Additional config (cl_tag).
        """
        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.cl_tag = config.get("cl_tag", "_w_lensing_ct")

        self.source_bin_mapping = config.get("source_bin_mapping", {})
        self.all_spectra = {}
        self.output_requirements = {}
        self.indexed_params = {}
        for t in self.spectrum_types:
            self.all_spectra[t] = []
            self.output_requirements[f"{t}_mbias"] = []
            self.indexed_params[t] = []
            if "gamma" in field_types[t][0]:
                for i in range(self.observed_data_vector.nz_s.shape[0]):
                    if i in self.spectrum_info[t]["bins0"]:
                        mi = self.source_bin_mapping.get(i, i)
                        self.output_requirements[f"{t}_mbias"].append(f"m_bias_{mi}")
                        self.indexed_params[t].append(f"m_bias_{mi}")
                    else:
                        self.indexed_params[t].append(f"NA")

            elif "gamma" in field_types[t][1]:
                for i in range(self.observed_data_vector.nz_s.shape[0]):
                    if i in self.spectrum_info[t]["bins1"]:
                        mi = self.source_bin_mapping.get(i, i)
                        self.output_requirements[f"{t}_mbias"].append(f"m_bias_{mi}")
                        self.indexed_params[t].append(f"m_bias_{mi}")
                    else:
                        self.indexed_params[t].append(f"NA")

        for k in self.indexed_params:
            self.indexed_params[k] = np.array(self.indexed_params[k])[:, None]

    def compute(self, state, params_values):
        """Apply multiplicative bias to C_ell for all spectrum types."""
        param_vec = jnp.array(list(params_values.values()))

        for t in self.spectrum_types:
            c_l = state[f"{t}{self.cl_tag}"]
            # Under sharding c_l is a local block of the pair axis; build the
            # per-pair m_bias vectors at the full length recorded by Limber
            # and slice them to the same block.
            if self.model_sharding is not None:
                n_c_l = self.model_sharding.pair_counts[t]
            else:
                n_c_l = c_l.shape[0]

            if "gamma" in field_types[t][0]:
                m_bias = param_vec[self.param_indices[t][:, 0]]
                m_bias = jnp.repeat(m_bias, n_c_l // m_bias.shape[0], 0)
                if self.model_sharding is not None:
                    m_bias = self.model_sharding.slice_local(m_bias)
                state["m_bias_0"] = m_bias
                state["param_vec"] = param_vec
                state["params_values"] = params_values
                c_l = c_l * (1 + m_bias[:, None])
            if "gamma" in field_types[t][1]:
                m_bias = param_vec[self.param_indices[t][:, 0]]
                m_bias = jnp.tile(m_bias, n_c_l // m_bias.shape[0]).flatten()
                if self.model_sharding is not None:
                    m_bias = self.model_sharding.slice_local(m_bias)
                state["m_bias_1"] = m_bias
                c_l = c_l * (1 + m_bias[:, None])

            state[f"{t}_mbias"] = c_l

        return state
