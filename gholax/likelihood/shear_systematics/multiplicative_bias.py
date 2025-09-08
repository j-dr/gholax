from ...util.likelihood_module import LikelihoodModule
from ...data_vector.two_point_spectrum import field_types
import jax.numpy as jnp
import numpy as np


class ShearMultiplicativeBias(LikelihoodModule):
    def __init__(self, observed_data_vector, spectrum_types, spectrum_info, **config):
        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.cl_tag = config.get("cl_tag", "_w_lensing_ct")

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
                        self.output_requirements[f"{t}_mbias"].append(f"m_bias_{i}")
                        self.indexed_params[t].append(f"m_bias_{i}")
                    else:
                        self.indexed_params[t].append(f"NA")

            elif "gamma" in field_types[t][1]:
                for i in range(self.observed_data_vector.nz_s.shape[0]):
                    if i in self.spectrum_info[t]["bins1"]:
                        self.output_requirements[f"{t}_mbias"].append(f"m_bias_{i}")
                        self.indexed_params[t].append(f"m_bias_{i}")
                    else:
                        self.indexed_params[t].append(f"NA")

        for k in self.indexed_params:
            self.indexed_params[k] = np.array(self.indexed_params[k])[:, None]

    def compute(self, state, params_values):
        param_vec = jnp.array(list(params_values.values()))

        for t in self.spectrum_types:
            c_l = state[f"{t}{self.cl_tag}"]
            n_c_l = c_l.shape[0]

            if "gamma" in field_types[t][0]:
                m_bias = param_vec[self.param_indices[t][:, 0]]
                m_bias = jnp.repeat(m_bias, n_c_l // m_bias.shape[0], 0)
                state["m_bias_0"] = m_bias
                state["param_vec"] = param_vec
                state["params_values"] = params_values
                c_l = c_l * (1 + m_bias[:, None])
            if "gamma" in field_types[t][1]:
                m_bias = param_vec[self.param_indices[t][:, 0]]
                m_bias = jnp.tile(m_bias, n_c_l // m_bias.shape[0]).flatten()
                state["m_bias_1"] = m_bias
                c_l = c_l * (1 + m_bias[:, None])

            state[f"{t}_mbias"] = c_l

        return state
