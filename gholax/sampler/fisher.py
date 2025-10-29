import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from copy import copy

class Fisher(object):
    def __init__(self, config):
        c = config["sampler"]["Fisher"]
        self.include_prior = c.get("include_prior", True)
        self.n_samples = c.get("n_samples", 20000)
        self.s8_module_index = c.get("s8_module_index", 1)

    def run(self, model, output_file=None):

        param_dict = model.prior.initial_position(random_start=False)
        F = 0
        for like in model.likelihoods:        
            jmp = jax.jit(partial(model.predict_model, like))
            gjmp = jax.jacobian(jmp)
            dmdt = gjmp(param_dict)
            dmdt_mat = np.array(list(dmdt.values()))
            cinv = model.likelihoods[like].observed_data_vector.cinv
            F += np.dot(dmdt_mat, np.dot(cinv, dmdt_mat.T))
        
        if self.include_prior:
            param_names = list(dmdt.keys())
            idx = np.array([list(param_dict.keys()).index(p) for p in param_names])
            prior_sigmas = np.array(model.prior.get_prior_sigmas())[idx]
            
            prior_cov = np.diag(1/prior_sigmas**2)
            F += prior_cov

        Finv = np.linalg.inv(F)
        ref = np.array([list(param_dict.values())[list(param_dict.keys()).index(p)] for p in param_names])        
        bounds = model.prior.get_minimizer_bounds()        
        priors = {p:bounds[list(param_dict.keys()).index(p)] for p in ['As', 'ns', 'H0', 'ombh2', 'omch2']}        

        samples_i = np.random.multivariate_normal(ref, Finv, self.n_samples)
        for i, p in enumerate(priors):
            if i == 0:
                idx = (priors[p][0] < samples_i[:,param_names.index(p)]) & (samples_i[:,param_names.index(p)] < priors[p][1])
            else:
                idx &= (priors[p][0] < samples_i[:,param_names.index(p)]) & (samples_i[:,param_names.index(p)] < priors[p][1])
        
        samples_i = samples_i[idx]
        
        x =  jnp.array([samples_i[:,param_names.index('As')], 
                        samples_i[:,param_names.index('ns')],
                        samples_i[:,param_names.index('H0')],
                        -np.ones_like(samples_i[:,param_names.index('As')]),
                        samples_i[:,param_names.index('ombh2')],
                        samples_i[:, param_names.index('omch2')],
                        np.ones_like(samples_i[:, param_names.index('As')])*-2.,
                        np.zeros_like(samples_i[:, param_names.index('As')])])    

        labels = [model.prior.config[p]['latex'] if 'latex' in model.prior.config[p] else p for p in param_names]
        labels.extend([r'\Omega_m', r'\sigma_8', r'S_8'])
        like = model.likelihoods[list(model.likelihoods.keys())[0]]
        om = (samples_i[:, param_names.index('omch2')]+samples_i[:, param_names.index('ombh2')])/(samples_i[:, param_names.index('H0')]/100)**2
        sigma8 = like.likelihood_pipeline[self.s8_module_index].emulator.predict(x.T)
        s8 = sigma8[:,0] * np.sqrt(om/0.3)    
        samples = np.hstack([samples_i, om[:,None], sigma8, s8[:,None]])
        pnames = copy(param_names)
        pnames.extend(['omegam', 'sigma8_emu', 's8'])    

        return samples, pnames        
        



