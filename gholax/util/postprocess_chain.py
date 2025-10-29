import numpy as np
from getdist import MCSamples
import h5py as h5
import json
import jax.numpy as jnp

def load_samples_checkpoint_nuts(output_file, model, likelihood_name, s8_module_index=1, burn_in_frac=0, ignore_chains=[], smooth_scale=-1):
    """
    Load samples from a NUTS checkpoint file and return a GetDist MCSamples object along with best-fit parameters.
    output_file: str
        Base name of the output files (without extensions).
    model: object
        The model object containing prior and likelihood information.
    likelihood_name: str
        The name of the likelihood to be used from the model.likelihoods dictionary.
    s8_module_index: int
        Index of the module in the likelihood pipeline that computes sigma8.
    burn_in_frac: float
        Fraction of samples to discard as burn-in.
    ignore_chains: list
        List of chain indices to ignore. 
    smooth_scale: float
        Smoothing scale for GetDist plots.
    """
    samples = np.load(f'{output_file}.samples_chk.npy')
    log_posterior = np.load(f'{output_file}.logposterior_chk.npy')
    
    samples_list = []

    params = model.prior.get_reference_point()
    sigmas = model.prior.get_prior_sigmas()
    reference = jnp.array(list(model.prior.get_reference_point().values()))
    names = list(params.keys())
    
    like = model.likelihoods[likelihood_name]
    labels = [model.prior.config[p]['latex'] if 'latex' in model.prior.config[p] else p for p in names]

    with open(f'{output_file}.minimization_results.json', 'r') as fp:
        opt = json.load(fp)
    samples_i = np.array(opt['x_opt'])*sigmas + reference
    cosmo_params = ['As', 'ns', 'H0', 'w', 'ombh2', 'omch2', 'logmnu']
    samps = []
    like = model.likelihoods[likelihood_name]

    for p in cosmo_params:
        if p in names:
            samps.append(samples_i[:, names.index(p)])
        else:
            samps.append(np.ones(len(samples_i)) * like.fixed_params[p])
    
    samps.append(np.zeros(len(samples_i)))
    x =  jnp.array(samps)
    
    om_bf = (samples_i[:, names.index('omch2')]+samples_i[:, names.index('ombh2')])/(samples_i[:, names.index('H0')]/100)**2
    sigma8_bf = like.likelihood_pipeline[s8_module_index].emulator.predict(x.T)
    
    log_post = []
    for i in range(samples.shape[0]):
        if i in ignore_chains: continue
        samples_i = samples[i,:,:]
        samps = []
        for p in cosmo_params:
            if p in names:
                samps.append(samples_i[:, names.index(p)])
            else:
                samps.append(np.ones(len(samples_i)) * like.fixed_params[p])
        
        samps.append(np.zeros(len(samples_i)))
        x =  jnp.array(samps)
        om = (samples_i[:, names.index('omch2')]+samples_i[:, names.index('ombh2')])/(samples_i[:, names.index('H0')]/100)**2
        sigma8 = like.likelihood_pipeline[s8_module_index].emulator.predict(x.T)
        s8 = sigma8[:,0] * np.sqrt(om/0.3)
        log_post.append(log_posterior[i,:])
        samples_i = np.hstack([samples_i, om[:,None], sigma8, s8[:,None]])
        samples_list.append(samples_i)

    names.extend(['omegam', 'sigma8', 's8'])
    labels.extend([r'\Omega_m', r'\sigma_8', r'S_8'])
    samples = np.array(samples_list)
    log_post = np.array(log_post)
    gds = MCSamples(samples=samples[:,:,:], names = names, labels=labels, ignore_rows=burn_in_frac, loglikes=log_post, settings={'smooth_scale_2D':smooth_scale, 'smooth_scale_1D':smooth_scale})

    params_bf_chain = dict(zip(names, np.array(opt['x_opt'][0])*sigmas + reference))    
    params_bf_chain['sigma8'] = sigma8_bf[0]
    params_bf_chain['omegam'] = om_bf[0]
    params_bf_chain['s8'] = sigma8_bf[0] * np.sqrt(om_bf[0]/0.3)

    return gds, params_bf_chain, reference, names, opt

def load_samples_emcee(output_file, model, likelihood_name, s8_module_index=1, smooth_scale=-1, burn_in_frac=0.33):
    sampler_data = h5.File(f'{output_file}.0.samples.h5', 'r')

    nwalkers = sampler_data['mcmc/chain'].shape[1]
    samples = sampler_data['mcmc/chain'][:].reshape(-1,sampler_data['mcmc/chain'].shape[-1])[:-50*nwalkers]
    sampler_data.close()
    
    params = model.prior.get_reference_point()    
    names = list(params.keys())
    labels = [model.prior.config[p]['latex'] if 'latex' in model.prior.config[p] else p for p in names]
    reference = jnp.array(list(model.prior.get_reference_point().values()))
    like = model.likelihoods[likelihood_name]

    samples_i = samples[:,:]
    x =  jnp.array([samples_i[:,names.index('As')], 
                    samples_i[:,names.index('ns')],
                    samples_i[:,names.index('H0')],
                    -np.ones_like(samples_i[:,names.index('As')]),
                    samples_i[:,names.index('ombh2')],
                    samples_i[:, names.index('omch2')],
                    -2*np.ones_like(samples_i[:,names.index('As')]),
                    np.zeros_like(samples_i[:, names.index('As')])])
    
    om = (samples_i[:, names.index('omch2')]+samples_i[:, names.index('ombh2')])/(samples_i[:, names.index('H0')]/100)**2
    sigma8 = like.likelihood_pipeline[s8_module_index].emulator.predict(x.T)
    s8 = sigma8[:,0] * np.sqrt(om/0.3)    
    
    samples = np.hstack([samples_i, om[:,None], sigma8, s8[:,None]])
    labels.extend([r'\Omega_m', r'\sigma_8', r'S_8'])
    names.extend(['omegam', 'sigma8', 's8'])
    
    gds = MCSamples(samples=samples[:,:], ignore_rows=burn_in_frac, names = names, labels=labels,
                    settings={'smooth_scale_2D':smooth_scale, 'smooth_scale_1D':smooth_scale})

    return gds, reference, names, params
    

def load_samples_from_checkpoint(output_file, model, likelihood_name, sampler, s8_module_index=1, burn_in_frac=0, ignore_chains=[], smooth_scale=-1, new_param_order=True):
    
    if sampler == 'NUTS':
        return load_samples_checkpoint_nuts(output_file, model, likelihood_name, s8_module_index=s8_module_index, burn_in_frac=burn_in_frac, ignore_chains=ignore_chains, smooth_scale=smooth_scale)
    elif sampler == 'emcee':
        return load_samples_emcee(output_file, model, likelihood_name, s8_module_index=s8_module_index, smooth_scale=smooth_scale, burn_in_frac=burn_in_frac)
    else:
        raise NotImplementedError(f"Sampler {sampler} not recognized.")
