#!/usr/bin/env python
"""Convert emulator weights from JSON to HDF5 format.

Usage:
    python bin/json2h5.py input.json --n_pcs N [output.h5]

The JSON format stores all PCA components; --n_pcs trims to only the
components used during training. Scalar vs spectrum emulators are
detected automatically (spectrum files contain a non-trivial 'v' key).
"""
import argparse
import json
import os

import h5py
import numpy as np


def json2h5(input_path, output_path, n_pcs):
    with open(input_path, 'r') as f:
        d = json.load(f)

    # Detect scalar emulator: no v key, or v is None
    is_scalar = d.get('v') is None

    with h5py.File(output_path, 'w') as f:
        g = f.create_group('W')
        for i, w in enumerate(d['W']):
            g.create_dataset(f'W_{i}', data=np.array(w, dtype=np.float32))

        g = f.create_group('b')
        for i, b in enumerate(d['b']):
            g.create_dataset(f'b_{i}', data=np.array(b, dtype=np.float32))

        g = f.create_group('alphas')
        for i, a in enumerate(d['alphas']):
            g.create_dataset(f'alphas_{i}', data=np.array([a], dtype=np.float32))

        g = f.create_group('betas')
        for i, b in enumerate(d['betas']):
            g.create_dataset(f'betas_{i}', data=np.array([b], dtype=np.float32))

        g = f.create_group('pc_mean')
        g.create_dataset('pc_mean_0', data=np.array(d['pc_mean'][:n_pcs], dtype=np.float32))

        g = f.create_group('pc_sigmas')
        g.create_dataset('pc_sigmas_0', data=np.array(d['pc_sigmas'][:n_pcs], dtype=np.float32))

        if d.get('param_mean') is not None:
            g = f.create_group('param_mean')
            g.create_dataset('param_mean_0', data=np.array(d['param_mean'], dtype=np.float32))

        if d.get('param_sigmas') is not None:
            g = f.create_group('param_sigmas')
            g.create_dataset('param_sigmas_0', data=np.array(d['param_sigmas'], dtype=np.float32))

        if not is_scalar:
            v = np.array(d['v'], dtype=np.float32)
            g = f.create_group('v')
            g.create_dataset('v_0', data=v[:, :n_pcs])

            if d.get('mean') is not None:
                g = f.create_group('mean')
                g.create_dataset('mean_0', data=np.array(d['mean'], dtype=np.float32))

            if d.get('sigmas') is not None:
                g = f.create_group('sigmas')
                g.create_dataset('sigmas_0', data=np.array(d['sigmas'], dtype=np.float32))

            if d.get('fstd') is not None:
                g = f.create_group('fstd')
                g.create_dataset('fstd_0', data=np.array(d['fstd'], dtype=np.float32))

    print(f"Converted {input_path} -> {output_path}")
    print(f"  scalar={is_scalar}, n_pcs={n_pcs}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert emulator JSON weights to HDF5')
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('output', nargs='?', default=None, help='Output HDF5 file (default: replace .json with .h5)')
    parser.add_argument('--n_pcs', type=int, required=True, help='Number of PCA components to keep')
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.input.replace('.json', '.h5')

    json2h5(args.input, output, args.n_pcs)


if __name__ == '__main__':
    main()
