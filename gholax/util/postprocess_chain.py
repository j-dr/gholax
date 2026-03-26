import numpy as np
from getdist import MCSamples
import h5py as h5
import json
import jax.numpy as jnp

def _build_emu_input(samples_i, names, cosmo_params, like):
    """Build emulator input array from chain samples using the emulator's parameter order.

    Args:
        samples_i: Array of shape (n_samples, n_params) with chain samples.
        names: List of sampled parameter names.
        cosmo_params: List of cosmological parameter names expected by the emulator (excluding 'z').
        like: Likelihood object with fixed_params for non-sampled parameters.

    Returns:
        Array of shape (n_samples, len(cosmo_params) + 1) with z=0 appended.
    """
    samps = []
    for p in cosmo_params:
        if p in names:
            samps.append(samples_i[:, names.index(p)])
        elif p == 'logmnu':
            if 'logmnu' in names:
                samps.append(samples_i[:, names.index('logmnu')])
            elif 'mnu' in names:
                samps.append(np.log10(samples_i[:, names.index('mnu')]))
            else:
                samps.append(np.log10(np.ones(len(samples_i)) * like.fixed_params.get('mnu', 0.06)))
        else:
            samps.append(np.ones(len(samples_i)) * like.fixed_params[p])
    samps.append(np.zeros(len(samples_i)))
    return jnp.array(samps).T


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
    like = model.likelihoods[likelihood_name]
    s8_emu = like.likelihood_pipeline[s8_module_index].emulator
    ipo = getattr(s8_emu, 'input_param_order', ['As', 'ns', 'H0', 'w', 'ombh2', 'omch2', 'logmnu', 'z'])
    # cosmo_params is everything except the trailing 'z'
    cosmo_params = [p for p in ipo if p != 'z']

    x = _build_emu_input(samples_i, names, cosmo_params, like)

    om_bf = (samples_i[:, names.index('omch2')]+samples_i[:, names.index('ombh2')])/(samples_i[:, names.index('H0')]/100)**2
    sigma8_bf = s8_emu.predict(x)

    log_post = []
    for i in range(samples.shape[0]):
        if i in ignore_chains: continue
        samples_i = samples[i,:,:]
        x = _build_emu_input(samples_i, names, cosmo_params, like)
        om = (samples_i[:, names.index('omch2')]+samples_i[:, names.index('ombh2')])/(samples_i[:, names.index('H0')]/100)**2
        sigma8 = s8_emu.predict(x)
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

def load_samples_checkpoint_mh(output_file, model, likelihood_name, s8_module_index=1, burn_in_frac=0, ignore_chains=[], smooth_scale=-1):
    """
    Load samples from a MH checkpoint file and return a GetDist MCSamples object along with best-fit parameters.
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
    like = model.likelihoods[likelihood_name]
    s8_emu = like.likelihood_pipeline[s8_module_index].emulator
    ipo = getattr(s8_emu, 'input_param_order', ['As', 'ns', 'H0', 'w', 'ombh2', 'omch2', 'logmnu', 'z'])
    # cosmo_params is everything except the trailing 'z'
    cosmo_params = [p for p in ipo if p != 'z']

    x = _build_emu_input(samples_i, names, cosmo_params, like)

    om_bf = (samples_i[:, names.index('omch2')]+samples_i[:, names.index('ombh2')])/(samples_i[:, names.index('H0')]/100)**2
    sigma8_bf = s8_emu.predict(x)

    log_post = []
    for i in range(samples.shape[0]):
        if i in ignore_chains: continue
        samples_i = samples[i,:,:]
        x = _build_emu_input(samples_i, names, cosmo_params, like)
        om = (samples_i[:, names.index('omch2')]+samples_i[:, names.index('ombh2')])/(samples_i[:, names.index('H0')]/100)**2
        sigma8 = s8_emu.predict(x)
        s8 = sigma8[:,0] * np.sqrt(om/0.3)
        samples_i = np.hstack([samples_i, om[:,None], sigma8, s8[:,None]])
        samples_list.append(samples_i)

    names.extend(['omegam', 'sigma8', 's8'])
    labels.extend([r'\Omega_m', r'\sigma_8', r'S_8'])
    samples = np.array(samples_list)
    log_post = np.array(log_post)
    gds = MCSamples(samples=samples[:,:,:], names = names, labels=labels, ignore_rows=burn_in_frac, settings={'smooth_scale_2D':smooth_scale, 'smooth_scale_1D':smooth_scale})

    params_bf_chain = dict(zip(names, np.array(opt['x_opt'][0])*sigmas + reference))    
    params_bf_chain['sigma8'] = sigma8_bf[0]
    params_bf_chain['omegam'] = om_bf[0]
    params_bf_chain['s8'] = sigma8_bf[0] * np.sqrt(om_bf[0]/0.3)

    return gds, params_bf_chain, reference, names, opt

def load_samples_emcee(output_file, model, likelihood_name, s8_module_index=1, smooth_scale=-1, burn_in_frac=0.33):
    """Load samples from an emcee checkpoint and return a GetDist MCSamples object.

    Args:
        output_file: Base name of the output files (without extensions).
        model: Model object containing prior and likelihood information.
        likelihood_name: Name of the likelihood in model.likelihoods.
        s8_module_index: Index of the pipeline module that computes sigma8.
        smooth_scale: Smoothing scale for GetDist plots.
        burn_in_frac: Fraction of samples to discard as burn-in.

    Returns:
        Tuple of (MCSamples, reference_values, param_names, reference_point_dict).
    """
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
    s8_emu = like.likelihood_pipeline[s8_module_index].emulator
    ipo = getattr(s8_emu, 'input_param_order', ['As', 'ns', 'H0', 'w', 'ombh2', 'omch2', 'logmnu', 'z'])
    cosmo_params = [p for p in ipo if p != 'z']

    x = _build_emu_input(samples_i, names, cosmo_params, like)

    om = (samples_i[:, names.index('omch2')]+samples_i[:, names.index('ombh2')])/(samples_i[:, names.index('H0')]/100)**2
    sigma8 = s8_emu.predict(x)
    s8 = sigma8[:,0] * np.sqrt(om/0.3)    
    
    samples = np.hstack([samples_i, om[:,None], sigma8, s8[:,None]])
    labels.extend([r'\Omega_m', r'\sigma_8', r'S_8'])
    names.extend(['omegam', 'sigma8', 's8'])
    
    gds = MCSamples(samples=samples[:,:], ignore_rows=burn_in_frac, names = names, labels=labels,
                    settings={'smooth_scale_2D':smooth_scale, 'smooth_scale_1D':smooth_scale})

    return gds, reference, names, params
    

def spot_check_emulator(
    chain_file,
    config_file,
    model,
    likelihood_name,
    sampler,
    n_samples=20,
    seed=42,
    burn_in_frac=0.3,
    ignore_chains=[],
    state_keys=None,
):
    """Spot-check emulator accuracy against exact calculations for a subsample of chain points.

    Reads a NUTS or emcee chain, randomly selects n_samples points, then evaluates
    both the emulator-based model and a freshly instantiated exact model (with
    use_emulator=False and use_boltzmann=True) at each point and returns the comparison.

    Args:
        chain_file: Base path for chain output files (without extensions).
        config_file: Path to the YAML config file used to run the chain.
        model: Instantiated emulator Model object.
        likelihood_name: Key into model.likelihoods to evaluate.
        sampler: Sampler type string ('NUTS' or 'emcee').
        n_samples: Number of chain points to evaluate exactly.
        seed: Random seed for reproducible subsampling.
        burn_in_frac: Fraction of samples to discard as burn-in from the start of each chain.
        ignore_chains: List of NUTS chain indices to skip.
        state_keys: Optional list of state dict keys to extract and return from the
            pipeline for both the emulator and exact models. If None, no state is saved.

    Returns:
        Dict with keys:
            'params': array of shape (n_samples, n_params) in physical parameter space.
            'param_names': list of sampled parameter names.
            'emu_predictions': array of shape (n_samples, n_dv), emulator predictions.
            'exact_predictions': array of shape (n_samples, n_dv), exact predictions.
            'frac_diff': (emu - exact) / exact, shape (n_samples, n_dv).
            'abs_diff': emu - exact, shape (n_samples, n_dv).
            'emu_log_like': array of shape (n_samples,), emulator log-likelihoods.
            'exact_log_like': array of shape (n_samples,), exact log-likelihoods.
            'delta_log_like': emu_log_like - exact_log_like, shape (n_samples,).
            'emu_state': dict mapping each requested state key to a list of per-sample
                values from the emulator model. Only present if state_keys is not None.
            'exact_state': dict mapping each requested state key to a list of per-sample
                values from the exact model. Only present if state_keys is not None.
    """
    import yaml
    from gholax.util.model import Model

    # --- Load chain samples (physical parameter space) ---
    if sampler == 'NUTS':
        raw = np.load(f'{chain_file}.samples_chk.npy')  # (n_chains, n_steps, n_params)
        n_chains, n_steps, _ = raw.shape
        chains = []
        for i in range(n_chains):
            if i in ignore_chains:
                continue
            burn_in_steps = int(n_steps * burn_in_frac)
            chains.append(raw[i, burn_in_steps:, :])
        samples_flat = np.concatenate(chains, axis=0)  # (N_total, n_params)
    elif sampler == 'emcee':
        sampler_data = h5.File(f'{chain_file}.0.samples.h5', 'r')
        nwalkers = sampler_data['mcmc/chain'].shape[1]
        samples_flat = sampler_data['mcmc/chain'][:].reshape(-1, sampler_data['mcmc/chain'].shape[-1])
        sampler_data.close()
        n_burn = int(len(samples_flat) * burn_in_frac)
        samples_flat = samples_flat[n_burn:]
    else:
        raise NotImplementedError(f"Sampler '{sampler}' not recognized.")

    # --- Randomly subsample ---
    rng = np.random.default_rng(seed)
    n_total = len(samples_flat)
    if n_samples > n_total:
        raise ValueError(f"Requested {n_samples} samples but chain only has {n_total} points after burn-in.")
    indices = rng.choice(n_total, size=n_samples, replace=False)
    subsamples = samples_flat[indices]  # (n_samples, n_params)

    # --- Build exact (no-emulator) config ---
    with open(config_file, 'r') as fp:
        cfg_exact = yaml.load(fp, Loader=yaml.SafeLoader)

    for module_cfg in cfg_exact.get('theory', {}).values():
        if isinstance(module_cfg, dict):
            module_cfg['use_emulator'] = False
            module_cfg['use_boltzmann'] = True

    for lname, like_cfg in cfg_exact.get('likelihood', {}).items():
        if lname == 'params':
            continue
        if isinstance(like_cfg, dict):
            like_cfg['use_boltzmann'] = True

    print("Instantiating exact (no-emulator) model. This may take a moment...", flush=True)
    exact_model = Model(cfg_exact)

    # --- Evaluate predictions at each subsampled point ---
    param_names = model.param_names
    like = model.likelihoods[likelihood_name]
    dv = like.observed_data_vector
    data_masked = np.array(dv.measured_spectra[dv.scale_mask])
    cinv = np.array(dv.cinv)

    emu_preds = []
    exact_preds = []
    emu_log_likes = []
    exact_log_likes = []
    if state_keys is not None:
        emu_state = {k: [] for k in state_keys}
        exact_state = {k: [] for k in state_keys}

    for i, sample in enumerate(subsamples):
        params_dict = dict(zip(param_names, sample))

        if state_keys is not None:
            emu_pred, emu_s = model.predict_model(
                likelihood_name, params_dict, apply_scale_mask=False, return_state=True
            )
            exact_pred, exact_s = exact_model.predict_model(
                likelihood_name, params_dict, apply_scale_mask=False, return_state=True
            )
            for k in state_keys:
                emu_state[k].append(np.array(emu_s[k]))
                exact_state[k].append(np.array(exact_s[k]))
        else:
            emu_pred = model.predict_model(
                likelihood_name, params_dict, apply_scale_mask=False
            )
            exact_pred = exact_model.predict_model(
                likelihood_name, params_dict, apply_scale_mask=False
            )

        emu_preds.append(np.array(emu_pred))
        exact_preds.append(np.array(exact_pred))

        diff_emu = data_masked - np.array(emu_pred)[dv.scale_mask]
        diff_exact = data_masked - np.array(exact_pred)[dv.scale_mask]
        emu_ll = float(-0.5 * diff_emu @ cinv @ diff_emu)
        exact_ll = float(-0.5 * diff_exact @ cinv @ diff_exact)
        emu_log_likes.append(emu_ll)
        exact_log_likes.append(exact_ll)

        max_frac = np.max(np.abs((emu_pred - exact_pred) / exact_pred))
        print(f"  [{i+1}/{n_samples}] max |frac diff| = {max_frac:.4e}  delta_log_like = {emu_ll - exact_ll:.4e}", flush=True)

    emu_preds = np.array(emu_preds)
    exact_preds = np.array(exact_preds)
    emu_log_likes = np.array(emu_log_likes)
    exact_log_likes = np.array(exact_log_likes)

    result = {
        'params': subsamples,
        'param_names': param_names,
        'emu_predictions': emu_preds,
        'exact_predictions': exact_preds,
        'frac_diff': (emu_preds - exact_preds) / exact_preds,
        'abs_diff': emu_preds - exact_preds,
        'emu_log_like': emu_log_likes,
        'exact_log_like': exact_log_likes,
        'delta_log_like': emu_log_likes - exact_log_likes,
    }
    if state_keys is not None:
        result['emu_state'] = emu_state
        result['exact_state'] = exact_state
    return result


def load_samples_from_checkpoint(output_file, model, likelihood_name, sampler, s8_module_index=1, burn_in_frac=0, ignore_chains=[], smooth_scale=-1, new_param_order=True):
    """Load MCMC samples from a checkpoint file, dispatching by sampler type.

    Args:
        output_file: Base name of the output files (without extensions).
        model: Model object containing prior and likelihood information.
        likelihood_name: Name of the likelihood in model.likelihoods.
        sampler: Sampler type string ('NUTS' or 'emcee').
        s8_module_index: Index of the pipeline module that computes sigma8.
        burn_in_frac: Fraction of samples to discard as burn-in.
        ignore_chains: List of chain indices to ignore.
        smooth_scale: Smoothing scale for GetDist plots.
        new_param_order: Unused, kept for backwards compatibility.

    Returns:
        Result from the sampler-specific loader (varies by sampler type).
    """
    if sampler == 'NUTS':
        return load_samples_checkpoint_nuts(output_file, model, likelihood_name, s8_module_index=s8_module_index, burn_in_frac=burn_in_frac, ignore_chains=ignore_chains, smooth_scale=smooth_scale)
    elif sampler == 'emcee':
        return load_samples_emcee(output_file, model, likelihood_name, s8_module_index=s8_module_index, smooth_scale=smooth_scale, burn_in_frac=burn_in_frac)
    elif sampler == 'MetropolisHastings':
        return load_samples_checkpoint_mh(output_file, model, likelihood_name, s8_module_index=s8_module_index, burn_in_frac=burn_in_frac, ignore_chains=ignore_chains, smooth_scale=smooth_scale)        
    else:
        raise NotImplementedError(f"Sampler {sampler} not recognized.")
