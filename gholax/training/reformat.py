import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import h5py
import yaml
from yaml import Loader
from gholax.util.model import Model


def load_training_data(key, dataset, downsample=None, spec_idx=None):
    keys = list(dataset.keys())
    keys = [k for k in keys if key in k]
    if downsample is not None:
        keys = keys[::downsample]
    nproc = len(keys)
    print(key)
    print(key in keys)
    for i, k in enumerate(keys):
        rank = k.split('_')[-1]
        d = dataset[k][:]
        p = dataset['params_{}'.format(rank)][:]
        if i == 0:
            if spec_idx is not None:
                size_i = [d.shape[0], d.shape[2], d.shape[3]]
            else:
                size_i = d.shape

            size = [size_i[0] * nproc]
            psize = [size_i[0] * nproc, p.shape[1]]
            [size.append(size_i[j]) for j in range(1, len(size_i))]

            Y = np.zeros(size)
            X = np.zeros(psize)

        X[i * size_i[0]:(i + 1) * size_i[0]] = p

        if spec_idx is not None:
            Y[i * size_i[0]:(i + 1) * size_i[0]] = d[:, spec_idx, ...]
        else:
            Y[i * size_i[0]:(i + 1) * size_i[0]] = d

    return X, Y


def _write_dataset(f, name, data):
    try:
        f.create_dataset(name, data.shape)
    except:
        del f[name]
        f.create_dataset(name, data.shape)
    f[name][:] = data


def _get_target_config(target):
    """Return (likelihood_name, pipeline_idx, data_key, sigma8_key, reformatter_name)."""
    if target == 'p_cleft':
        return ('Nx2PTAngularPowerSpectrum', 6,
                'Nx2PTAngularPowerSpectrum.p_ij_real_space_bias_grid',
                'Nx2PTAngularPowerSpectrum.sigma8_z', 'cleft')
    elif target == 'p_density_shape':
        return ('Nx2PTAngularPowerSpectrum', 9,
                'Nx2PTAngularPowerSpectrum.p_ij_real_space_density_shape_grid',
                'Nx2PTAngularPowerSpectrum.sigma8_z', 'density_shape')
    elif 'shape_shape' in target:
        return ('Nx2PTAngularPowerSpectrum', 10,
                'Nx2PTAngularPowerSpectrum.p_mij_real_space_shape_shape_grid',
                'Nx2PTAngularPowerSpectrum.sigma8_z', 'shape_shape')
    elif target.startswith('pkell'):
        return ('RSDPK', 5,
                'RSDPK.p_ij_ell_no_ap_redshift_space_bias_grid',
                'RSDPK.sigma8_z', 'pkell')
    elif target == 'f_z':
        return ('RSDPK', 3, 'RSDPK.f_z', None, 'f_z')
    elif target == 'sigma8z':
        return ('RSDPK', 3, 'RSDPK.sigma8_z', None, 'sigma8z')
    else:
        raise ValueError(f"Unknown target: {target}. Expected one of: p_cleft, p_density_shape, "
                         f"p{{0,1,2}}_shape_shape, pkell{{0,1,2,3}}, f_z, sigma8z")


def _reformat_spectra_common(training_data, Ptrain_all, Ftrain_all_spec, sigma8_key, z, k):
    """Shared logic for spectra reformatters: filter zeros, build expanded params."""
    _, sigma8_z_all = load_training_data(sigma8_key, training_data)
    idx = np.any(Ptrain_all > 0, axis=1)
    Ptrain = Ptrain_all[idx]
    sigma8_z = sigma8_z_all[idx]
    del Ptrain_all, sigma8_z_all

    Ptrain_z = np.zeros((len(Ptrain) * len(z), Ptrain.shape[1] + 1))
    Ptrain_z[:, -1] = sigma8_z.flatten()

    for j in range(len(z)):
        Ptrain_z[j::len(z), :-1] = Ptrain

    return idx, Ptrain_z


def _reformat_cleft(training_data, data_key, sigma8_key, z, k):
    nspec = 19
    Ptrain_all, Ftrain_all_spec = load_training_data(data_key, training_data)
    idx, Ptrain_z = _reformat_spectra_common(training_data, Ptrain_all, Ftrain_all_spec, sigma8_key, z, k)

    _write_dataset(training_data, 'params', Ptrain_z)

    Ftrain = Ftrain_all_spec[idx]
    Ftrain = Ftrain.reshape(-1, len(k) * nspec)
    Ftrain /= Ptrain_z[:, -1, None] ** 2

    _write_dataset(training_data, 'p_cleft', Ftrain)
    del Ftrain, Ftrain_all_spec


def _reformat_density_shape(training_data, data_key, sigma8_key, z, k):
    nspec = 21
    Ptrain_all, Ftrain_all_spec = load_training_data(data_key, training_data)
    idx, Ptrain_z = _reformat_spectra_common(training_data, Ptrain_all, Ftrain_all_spec, sigma8_key, z, k)

    _write_dataset(training_data, 'params', Ptrain_z)

    Ftrain = Ftrain_all_spec[idx]
    Ftrain = np.einsum('iskz->izsk', Ftrain).reshape(-1, len(k) * nspec)
    Ftrain /= Ptrain_z[:, -1, None] ** 2

    _write_dataset(training_data, 'p_density_shape', Ftrain)
    del Ftrain, Ftrain_all_spec


def _reformat_shape_shape(training_data, data_key, sigma8_key, z, k):
    nspec = 13
    nm = 3
    Ptrain_all, Ftrain_all_spec = load_training_data(data_key, training_data)

    idx = None
    Ptrain_z = None
    for m in range(nm):
        print(m, flush=True)
        Ftrain_all = Ftrain_all_spec[:, m, :, :, :]  # imskz
        if m == 0:
            idx, Ptrain_z = _reformat_spectra_common(training_data, Ptrain_all, Ftrain_all, sigma8_key, z, k)
            _write_dataset(training_data, 'params', Ptrain_z)

        Ftrain = Ftrain_all[idx]
        Ftrain = np.einsum('iskz->izsk', Ftrain).reshape(-1, len(k) * nspec)
        Ftrain /= Ptrain_z[:, -1, None] ** 2

        _write_dataset(training_data, f'p{m}_shape_shape', Ftrain)
        del Ftrain

    del Ftrain_all_spec


def _reformat_pkell(training_data, data_key, sigma8_key, z, k):
    nspec = 13
    nell = 4
    Ptrain_all, Ftrain_all_spec = load_training_data(data_key, training_data)

    idx = None
    Ptrain_z = None
    for l in range(nell):
        print(l, flush=True)
        Ftrain_all = Ftrain_all_spec[:, :, l, :, :]  # islkz
        if l == 0:
            idx, Ptrain_z = _reformat_spectra_common(training_data, Ptrain_all, Ftrain_all, sigma8_key, z, k)
            _write_dataset(training_data, 'params', Ptrain_z)

        Ftrain = Ftrain_all[idx]
        Ftrain = np.einsum('iskz->izsk', Ftrain).reshape(-1, len(k) * nspec)
        Ftrain /= Ptrain_z[:, -1, None] ** 2

        _write_dataset(training_data, f'pkell{l}', Ftrain)
        del Ftrain

    del Ftrain_all_spec


def _reformat_scalar(training_data, data_key, z, output_name, param_name):
    Ptrain_all, Ftrain_all = load_training_data(data_key, training_data)
    idx = np.any(Ptrain_all > 0, axis=1)
    Ptrain = Ptrain_all[idx]
    Ftrain = Ftrain_all[idx]

    Ptrain_z = np.zeros((len(Ptrain) * len(z), Ptrain.shape[1] + 1))
    Ptrain_z[:, -1] = np.tile(z, len(Ptrain))
    for j in range(len(z)):
        Ptrain_z[j::len(z), :-1] = Ptrain

    Ftrain = Ftrain.flatten()

    _write_dataset(training_data, param_name, Ptrain_z)
    _write_dataset(training_data, output_name, Ftrain)


_RAW_KEY_TO_TARGETS = {
    ('Nx2PTAngularPowerSpectrum', 'p_ij_real_space_bias_grid'): [
        {'target': 'p_cleft', 'param_dataset': 'params'},
    ],
    ('Nx2PTAngularPowerSpectrum', 'p_ij_real_space_density_shape_grid'): [
        {'target': 'p_density_shape', 'param_dataset': 'params'},
    ],
    ('Nx2PTAngularPowerSpectrum', 'p_mij_real_space_shape_shape_grid'): [
        {'target': f'p{m}_shape_shape', 'param_dataset': 'params'} for m in range(3)
    ],
    ('Nx2PTAngularPowerSpectrum', 'sigma8_z'): [],
    ('RSDPK', 'p_ij_ell_no_ap_redshift_space_bias_grid'): [
        {'target': f'pkell{l}', 'param_dataset': 'params'} for l in range(4)
    ],
    ('RSDPK', 'sigma8_z'): [
        {'target': 'sigma8z', 'param_dataset': 'params_sigma8z'},
    ],
    ('RSDPK', 'f_z'): [
        {'target': 'f_z', 'param_dataset': 'params_f_z'},
    ],
}

_TRAINING_DEFAULTS = {
    'n_epochs': 500,
    'n_hidden': [100, 100, 100],
    'n_pcs': 10,
    'use_asinh': True,
    'scale_by_std': True,
    'learning_rate': [5e-3, 1e-3, 5e-4, 1e-4],
    'batch_size': [320, 640, 1280, 2560],
}


def generate_training_configs(generation_config_filename, output_dir=None, **training_overrides):
    """Generate training config YAML files for all targets in a generation config.

    Args:
        generation_config_filename: Path to the YAML config used with generate_training_data.
        output_dir: Directory for output configs and weights. Defaults to the
            directory containing generation_config_filename.
        **training_overrides: Override any training hyperparameter default
            (e.g. n_epochs=1000, n_pcs=20).

    Returns:
        List of paths to the generated training config files.
    """
    generation_config_filename = os.path.abspath(generation_config_filename)
    with open(generation_config_filename, 'r') as fp:
        info = yaml.load(fp, Loader=Loader)

    emu_info = info['emulate']
    raw_training_filename = emu_info['output_filename']

    if output_dir is None:
        output_dir = os.path.dirname(generation_config_filename)
    os.makedirs(output_dir, exist_ok=True)

    training_params = {**_TRAINING_DEFAULTS, **training_overrides}

    config_paths = []
    for likelihood, raw_keys in emu_info['likelihood'].items():
        if isinstance(raw_keys, str):
            raw_keys = [raw_keys]
        for raw_key in raw_keys:
            targets = _RAW_KEY_TO_TARGETS.get((likelihood, raw_key), None)
            if targets is None:
                print(f"Warning: no target mapping for ({likelihood}, {raw_key}), skipping")
                continue
            if not targets:
                continue

            for entry in targets:
                target = entry['target']
                config = {
                    'raw_training_filename': raw_training_filename,
                    'generation_config': generation_config_filename,
                    'training_filename': raw_training_filename,
                    'output_path': os.path.join(output_dir, f'{target}_emu'),
                    'target': target,
                    'param_dataset': entry['param_dataset'],
                    **training_params,
                }
                config_path = os.path.join(output_dir, f'train_{target}.yaml')
                with open(config_path, 'w') as fp:
                    yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
                config_paths.append(config_path)
                print(f"Wrote {config_path}")

    return config_paths


def _link_rank_files(master_filename):
    """Create a master HDF5 file with external links to per-rank files.

    Detects rank files matching ``{master_filename}.{N}`` and creates the
    master file with ``{key}_{rank}`` external links, replicating the
    linking step from ``generate_training_data``.

    Returns True if linking was performed, False if no rank files found.
    """
    import glob as globmod

    rank_files = sorted(globmod.glob(f"{master_filename}.*"))
    # Filter to only numeric suffixes (rank files)
    rank_files = [f for f in rank_files
                  if f.rsplit('.', 1)[-1].isdigit()]
    if not rank_files:
        return False

    nproc = len(rank_files)
    print(f"Master file missing or empty; linking {nproc} rank files...",
          flush=True)

    # Read dataset keys from the first rank file
    with h5py.File(rank_files[0], 'r') as fp:
        dataset_keys = list(fp.keys())

    with h5py.File(master_filename, 'w') as fp:
        for k in dataset_keys:
            for n in range(nproc):
                rank_path = f"{master_filename}.{n}"
                fp[f"{k}_{n}"] = h5py.ExternalLink(rank_path, k)

    print(f"Linked {nproc} rank files into {master_filename}", flush=True)
    return True


def reformat(training_data_filename, config_filename, target):
    """Reformat raw training data in-place for a given target quantity.

    Args:
        training_data_filename: Path to the raw HDF5 file from generate_training_data.
        config_filename: Path to the YAML config used for data generation.
        target: Target quantity string (e.g. 'p_cleft', 'pkell0', 'f_z', etc.).

    Returns:
        The training_data_filename (reformatted datasets are written in-place).
    """
    # Auto-detect missing master file and link rank files if needed
    needs_linking = False
    if not os.path.exists(training_data_filename):
        needs_linking = True
    else:
        with h5py.File(training_data_filename, 'r') as fp:
            if len(fp.keys()) == 0:
                needs_linking = True
    if needs_linking:
        if not _link_rank_files(training_data_filename):
            raise FileNotFoundError(
                f"No training data found: {training_data_filename} does not "
                f"exist and no rank files ({training_data_filename}.N) found")

    lik_name, pipe_idx, data_key, sigma8_key, reformatter = _get_target_config(target)

    model = Model(config_filename)
    pipeline_module = model.likelihoods[lik_name].likelihood_pipeline[pipe_idx]
    z = pipeline_module.z
    k = getattr(pipeline_module, 'k', None)

    training_data = h5py.File(training_data_filename, 'r+')
    print(training_data_filename, flush=True)

    if reformatter == 'cleft':
        _reformat_cleft(training_data, data_key, sigma8_key, z, k)
    elif reformatter == 'density_shape':
        _reformat_density_shape(training_data, data_key, sigma8_key, z, k)
    elif reformatter == 'shape_shape':
        _reformat_shape_shape(training_data, data_key, sigma8_key, z, k)
    elif reformatter == 'pkell':
        _reformat_pkell(training_data, data_key, sigma8_key, z, k)
    elif reformatter == 'f_z':
        _reformat_scalar(training_data, data_key, z, 'f_z', 'params_f_z')
    elif reformatter == 'sigma8z':
        _reformat_scalar(training_data, data_key, z, 'sigma8z', 'params_sigma8z')

    training_data.close()
    print(f"Reformatting complete for target '{target}'", flush=True)
    return training_data_filename


def generate_training_configs_cli():
    """CLI entry point: generate-training-configs config.yaml [output_dir]"""
    if len(sys.argv) < 2:
        print("Usage: generate-training-configs <generation_config.yaml> [output_dir]")
        sys.exit(1)
    generation_config = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_training_configs(generation_config, output_dir=output_dir)
