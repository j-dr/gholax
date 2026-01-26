import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from jax.scipy.integrate import trapezoid

from ...data_vector.two_point_spectrum import field_types
from ...util.likelihood_module import LikelihoodModule


class SmailOutlier(LikelihoodModule):
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
        self.alpha = config.get("alpha", 2)
        self.beta = config.get("beta", 0.798)
        self.z0 = config.get("z0", 0.178)
        outlier_distribution_file = config.get("outlier_distribution_file", None)
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
        self.output_requirements[f"{self.nz_name}_shifted"] = []
        for i in self.bins:
            self.output_requirements[f'{self.nz_name}_shifted'].extend(
            [
                f"delta_z_{self.param_name_base}_{i}",
                f"sigma_z_{self.param_name_base}_{i}",
                f"f_out_z_{self.param_name_base}_{i}",
                f"delta_z_out_{self.param_name_base}_{i}",
            ])


        if self.do_shift:
            self.indexed_params = np.array(
                [
                    [
                        f"delta_z_{self.param_name_base}_{i}",
                        f"sigma_z_{self.param_name_base}_{i}",
                        f"f_out_z_{self.param_name_base}_{i}",
                        f"delta_z_out_{self.param_name_base}_{i}",                       
                    ]
                    if i in self.bins
                    else ["NA", "NA", "NA", "NA"]
                    for i in range(self.nbins_all)
                ]
            )
            
            nz_smail = self.z**self.alpha * np.exp(-(self.z/self.z0)**self.beta)
            dz = jnp.diff(self.z, append=self.z[-1])
            norm = np.sum(nz_smail*dz)
            nz_smail /= norm
            cdf = np.cumsum(nz_smail*dz)
            zbin_edges_idx = cdf.searchsorted(np.linspace(0, 1.0, self.nbins_all+1))
            zbin_edges = self.z[zbin_edges_idx]
            self.zbin_edges = jnp.array([zbin_edges[:-1], zbin_edges[1:]]).T
            if outlier_distribution_file is not None:
                outlier_distributions = np.loadtxt(outlier_distribution_file)
                outlier_distributions = jnp.array(outlier_distributions)
            else:
                raise ValueError("outlier_distribution_file must be provided")            
            
            outlier_distributions = jnp.array([np.interp(self.z, outlier_distributions[0,:], outlier_distributions[1+i,:]) for i in range(self.nbins_all)])
            norms = trapezoid(outlier_distributions, x=self.z, axis=-1)
            self.outlier_distributions = outlier_distributions / norms[:,None]
            

    def compute(self, state, params_values):
        param_vec = jnp.array(list(params_values.values()))
        pz_params = param_vec[self.param_indices]
        pz_params = pz_params.at[:,1].set(jnp.where(pz_params[:,1] == 0, 1e-4, pz_params[:,1]))

        def f(i, xs):
            zbin_edges_i, nz_out_i, delta_z_i, sigma_z_i, f_out_i, delta_z_out_i = xs
            nz_true_itg = (self.z[:,None])**self.alpha * jnp.exp(-((self.z[:,None])/self.z0)**self.beta)
            nz_true_itg *= jnp.exp(-(self.z[None,:] - self.z[:, None]) ** 2 / (2 * (sigma_z_i * (1 + self.z[:, None])) ** 2))
            nz_true_itg /= jnp.sqrt(2 * jnp.pi) * sigma_z_i * (1 + self.z[:, None])
            nz_true_itg = jnp.where(((self.z[None,:]) >= zbin_edges_i[1]) | ((self.z[None,:]) < zbin_edges_i[0]), 0, nz_true_itg)
            nz_true_i = trapezoid(nz_true_itg, x=self.z, axis=-1)
            nz_true_i = jnp.interp(self.z - delta_z_i, self.z, nz_true_i, left=0, right=0)
            nz_true_i = nz_true_i / trapezoid(nz_true_i, x=self.z)
            nz_out_i_shifted = jnp.interp(
                self.z - delta_z_out_i, self.z, nz_out_i, left=0, right=0
            )
            nz_out_i_shifted = nz_out_i_shifted / trapezoid(nz_out_i_shifted, x=self.z) 
            nz_true_i = (1 - f_out_i) * nz_true_i + f_out_i * nz_out_i_shifted  
            return i + 1, nz_true_i

        _, nz_shifted = scan(f, 0, (self.zbin_edges, self.outlier_distributions, pz_params[:,0], pz_params[:,1], pz_params[:,2], pz_params[:,3]))
        state[f"{self.nz_name}_shifted"] = nz_shifted

        return state
