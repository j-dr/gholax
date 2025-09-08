from ...util.likelihood_module import LikelihoodModule
from ...data_vector.two_point_spectrum import field_types
from jax.scipy.integrate import trapezoid
from jax.lax import scan
import jax.numpy as jnp
import numpy as np


class DeltaZ(LikelihoodModule):
    def __init__(
        self,
        observed_data_vector,
        zmin=0,
        zmax=2.0,
        nz=125,
        param_name=None,
        nz_name=None,
        field="gamma",
        zbin_indices=0,
        **config,
    ):
        self.z = jnp.linspace(zmin, zmax, nz)
        self.observed_data_vector = observed_data_vector
        self.field = field
        self.zbin_index = zbin_indices
        self.nz_name = nz_name
        self.output_requirements = {}

        bins = []
        self.do_shift = False

        for spec in self.observed_data_vector.spectrum_info:
            if field in field_types[spec][0]:
                self.do_shift = True
                bins.extend(self.observed_data_vector.spectrum_info[spec]["bins0"])
            if field in field_types[spec][1]:
                self.do_shift = True
                bins.extend(self.observed_data_vector.spectrum_info[spec]["bins1"])

        self.bins = jnp.unique(jnp.array(bins))
        self.bins.sort()
        nz = getattr(self.observed_data_vector, self.nz_name)
        self.nbins = len(self.bins)
        self.nbins_all = nz.shape[0]
        self.param_name_base = param_name
        self.output_requirements[f"{self.nz_name}_shifted"] = [
            f"{param_name}_{i}" for i in self.bins
        ]

        if self.do_shift:
            self.indexed_params = np.array(
                [
                    f"{self.param_name_base}_{i}" if i in self.bins else "NA"
                    for i in range(self.nbins_all)
                ]
            )[:, None]

    def compute(self, state, params_values):
        if self.do_shift:
            param_vec = jnp.array(list(params_values.values()))
            nz = getattr(self.observed_data_vector, self.nz_name)
            delta_z = param_vec[self.param_indices[:, 0]]

            def f(i, xs):
                nz_i, delta_z_i = xs
                nz_i_shifted = jnp.interp(
                    self.z - delta_z_i, self.z, nz_i, left=0, right=0
                )
                nz_i_shifted = nz_i_shifted / trapezoid(nz_i_shifted, x=self.z)
                return i + 1, nz_i_shifted

            _, nz_shifted = scan(f, 0, (nz, delta_z))
            state[f"{self.nz_name}_shifted"] = nz_shifted
        else:
            pass

        return state
