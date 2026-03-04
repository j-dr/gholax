#!/usr/bin/env python
"""Convert lsstypes likelihood files to gholax RedshiftSpaceMultipoles HDF5 format."""

import sys
import os
import numpy as np
import h5py as h5
import yaml
import lsstypes as types


def main():
    with open(sys.argv[1], 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    likelihood_files = cfg['likelihood_files']
    ells = cfg.get('ells', [0, 2, 4])
    spectrum_type = cfg.get('spectrum_type', 'p_gg_ell')
    auto_only = cfg.get('auto_only', True)
    output_file = cfg['output_file']

    n_bins = len(likelihood_files)
    n_ell = len(ells)

    # --- Load fiducial data ---
    fid = cfg['fiducial']
    z_fid = np.loadtxt(fid['z_fid'])
    chiz_fid = np.loadtxt(fid['chiz_fid'])
    hz_fid = np.loadtxt(fid['hz_fid'])
    nz_d = np.loadtxt(fid['nz_file'])

    assert len(z_fid) == n_bins, f"z_fid length {len(z_fid)} != n_bins {n_bins}"
    assert len(chiz_fid) == n_bins, f"chiz_fid length {len(chiz_fid)} != n_bins {n_bins}"
    assert len(hz_fid) == n_bins, f"hz_fid length {len(hz_fid)} != n_bins {n_bins}"
    assert nz_d.shape[1] == n_bins + 1, (
        f"nz_d has {nz_d.shape[1]} cols, expected {n_bins + 1}"
    )

    # --- Read lsstypes likelihood files ---
    ko = None
    kth = None
    per_bin_data = []

    for b, filepath in enumerate(likelihood_files):
        print(f"Reading bin {b}: {filepath}")
        likelihood = types.read(filepath)
        data = likelihood.data
        window = likelihood.wmatrix

        # Extract spectra per ell
        bin_pk = []
        bin_k = []
        for ell in ells:
            pole = data.get(ells=ell)
            k = np.array(pole.coords('k'))
            pk = np.array(pole.value())
            bin_k.append(k)
            bin_pk.append(pk)

        # All ells should share the same observed k grid
        for i in range(1, n_ell):
            assert np.allclose(bin_k[0], bin_k[i]), (
                f"k grids differ between ell={ells[0]} and ell={ells[i]} in bin {b}"
            )

        ko_bin = bin_k[0]

        # Extract covariance (2D numpy array)
        cov_bin = np.array(likelihood.covariance)

        # Extract window matrix: lsstypes shape (n_ko*n_ell, n_kth*n_ell)
        W_lss = np.array(window.value())
        kth_bin = np.array(window.theory.get(0).coords('k'))

        # Validate k grids across bins
        if ko is None:
            ko = ko_bin
            kth = kth_bin
        else:
            assert np.allclose(ko, ko_bin), f"ko grids differ at bin {b}"
            assert np.allclose(kth, kth_bin), f"kth grids differ at bin {b}"

        per_bin_data.append({
            'pk': bin_pk,
            'cov': cov_bin,
            'W': W_lss,
        })

    n_ko = len(ko)
    n_kth = len(kth)
    n_dv_per_bin = n_ko * n_ell
    n_dv = n_bins * n_dv_per_bin

    # --- Build /spectra structured array ---
    # Ordering: for each bin pair -> for each ell -> for each k
    dt_spectra = np.dtype([
        ('spectrum_type', 'S10'),
        ('zbin0', np.int64),
        ('zbin1', np.int64),
        ('ell', np.int64),
        ('separation', np.float64),
        ('value', np.float64),
    ])

    spectra = np.zeros(n_dv, dtype=dt_spectra)
    counter = 0
    for b in range(n_bins):
        for li, ell in enumerate(ells):
            spectra['spectrum_type'][counter:counter + n_ko] = spectrum_type
            spectra['zbin0'][counter:counter + n_ko] = b
            spectra['zbin1'][counter:counter + n_ko] = b
            spectra['ell'][counter:counter + n_ko] = ell
            spectra['separation'][counter:counter + n_ko] = ko
            spectra['value'][counter:counter + n_ko] = per_bin_data[b]['pk'][li]
            counter += n_ko

    # --- Build /covariance structured array ---
    dt_cov = np.dtype([
        ('spectrum_type0', 'S10'),
        ('spectrum_type1', 'S10'),
        ('zbin00', np.int64),
        ('zbin01', np.int64),
        ('zbin10', np.int64),
        ('zbin11', np.int64),
        ('ell0', np.int64),
        ('ell1', np.int64),
        ('separation0', np.float64),
        ('separation1', np.float64),
        ('value', np.float64),
    ])

    cov = np.zeros((n_dv, n_dv), dtype=dt_cov)

    # Fill metadata via broadcasting: row fields from spectra[i], col fields from spectra[j]
    cov['spectrum_type0'][:] = spectra['spectrum_type'][:, np.newaxis]
    cov['spectrum_type1'][:] = spectra['spectrum_type'][np.newaxis, :]
    cov['zbin00'][:] = spectra['zbin0'][:, np.newaxis]
    cov['zbin01'][:] = spectra['zbin1'][:, np.newaxis]
    cov['zbin10'][:] = spectra['zbin0'][np.newaxis, :]
    cov['zbin11'][:] = spectra['zbin1'][np.newaxis, :]
    cov['ell0'][:] = spectra['ell'][:, np.newaxis]
    cov['ell1'][:] = spectra['ell'][np.newaxis, :]
    cov['separation0'][:] = spectra['separation'][:, np.newaxis]
    cov['separation1'][:] = spectra['separation'][np.newaxis, :]

    # Fill covariance values — block diagonal (one block per bin)
    for b in range(n_bins):
        start = b * n_dv_per_bin
        end = start + n_dv_per_bin
        cov['value'][start:end, start:end] = per_bin_data[b]['cov']

    assert np.allclose(cov['value'], cov['value'].T, atol=1e-16), \
        "Covariance matrix is not symmetric"

    cov_flat = cov.flatten()

    # --- Transpose and store window matrices ---
    # lsstypes: (n_ko*n_ell, n_kth*n_ell)  rows=observed, cols=theory
    # gholax:   (n_kth*n_ell, n_ko*n_ell)  rows=theory,   cols=observed
    windows = {}
    for b in range(n_bins):
        W_gholax = per_bin_data[b]['W'].T
        windows[f'{b}_{b}'] = W_gholax

    # --- Write output HDF5 ---
    with h5.File(output_file, 'w') as f:
        f.create_dataset('spectra', data=spectra)
        f.create_dataset('covariance', data=cov_flat)
        f.create_dataset('z_fid', data=z_fid)
        f.create_dataset('chiz_fid', data=chiz_fid)
        f.create_dataset('hz_fid', data=hz_fid)
        f.create_dataset('nz_d', data=nz_d)
        f.create_dataset('kth_rsd', data=kth)
        f.create_dataset('ko_rsd', data=ko)

        grp = f.create_group('pkell_windows')
        for key, W in windows.items():
            grp.create_dataset(key, data=W)

    # --- Summary ---
    size_mb = os.path.getsize(output_file) / 1e6
    print(f"\nWrote {output_file}")
    print(f"  n_bins:  {n_bins}")
    print(f"  n_ko:    {n_ko}")
    print(f"  n_kth:   {n_kth}")
    print(f"  n_ell:   {n_ell}")
    print(f"  n_dv:    {n_dv}")
    print(f"  size:    {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
