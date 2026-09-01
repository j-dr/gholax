"""
Generate a minimal synthetic HDF5 data vector for testing nx2pt_configspace.
Produces a file with w_theta, gamma_t, xi_plus, xi_minus spectra,
Gaussian n(z) for lens and source bins, and a diagonal covariance.

Run with:
    python generate_dummy_datavector.py
Output:
    dummy_configspace_dv.h5
"""
import numpy as np
import h5py

# ---- Configuration ----
N_LENS_BINS   = 2
N_SOURCE_BINS = 2
N_THETA       = 20   # angular bins
N_Z           = 100  # z grid points for n(z)

# Theta bin centers: 10 to 300 arcmin in radians, log-spaced
theta_centers = np.deg2rad(np.logspace(np.log10(10/60), np.log10(300/60), N_THETA))

# Redshift grid
z_grid = np.linspace(0.0, 3.0, N_Z)

# ---- n(z): simple Gaussians ----
def gaussian_nz(z, z_mean, sigma):
    nz = np.exp(-0.5 * ((z - z_mean) / sigma)**2)
    nz /= np.trapz(nz, z)
    return nz

# Lens n(z): two bins at z=0.3, 0.5
lens_means  = [0.3, 0.5]
lens_sigmas = [0.05, 0.05]

# Source n(z): two bins at z=0.7, 1.0
source_means  = [0.7, 1.0]
source_sigmas = [0.1, 0.1]

# Format: first column = z, remaining = n(z) per bin
nz_d = np.zeros((N_Z, N_LENS_BINS + 1))
nz_d[:, 0] = z_grid
for i, (zm, zs) in enumerate(zip(lens_means, lens_sigmas)):
    nz_d[:, i+1] = gaussian_nz(z_grid, zm, zs)

nz_s = np.zeros((N_Z, N_SOURCE_BINS + 1))
nz_s[:, 0] = z_grid
for i, (zm, zs) in enumerate(zip(source_means, source_sigmas)):
    nz_s[:, i+1] = gaussian_nz(z_grid, zm, zs)

# ---- Build spectra array ----
dt = np.dtype([
    ("spectrum_type", "S10"),
    ("zbin0", int),
    ("zbin1", int),
    ("separation", float),
    ("value", float),
])

rows = []

# w_theta: lens auto-correlations only
for i in range(N_LENS_BINS):
    for th in theta_centers:
        rows.append((b"w_theta", i, i, th, 1e-4))

# gamma_t: all lens x source pairs
for i in range(N_LENS_BINS):
    for j in range(N_SOURCE_BINS):
        for th in theta_centers:
            rows.append((b"gamma_t", i, j, th, 1e-5))
    
# Load theory model vector
# Theory prediction for xi+/xi- at reference cosmology:
# As=2.107e-9, omch2=0.11923, ombh2=0.022447, H0=67.7, ns=0.9649, w=-1.0, mnu=0.06eV
# Computed with gholax using CLASS + CLEFT bias model, no IA
# Lens bins: z~0.3, 0.5 (sigma=0.05); Source bins: z~0.7, 1.0 (sigma=0.1)
# Theta range: 10-300 arcmin, 20 log-spaced bins
# Order: xi+ (pairs 00, 01, 11) then xi- (pairs 00, 01, 11), 20 theta bins each
theory = np.array([
    4.70388111e-06, 4.15300246e-06, 3.69423703e-06, 3.34294931e-06,
    2.92238256e-06, 2.59460120e-06, 2.25510829e-06, 1.95271730e-06,
    1.67967909e-06, 1.43154369e-06, 1.20614540e-06, 1.00901028e-06,
    8.33729849e-07, 6.81672141e-07, 5.49799654e-07, 4.37615837e-07,
    3.43258574e-07, 2.64720281e-07, 2.00499296e-07, 1.50646880e-07,
    6.65867897e-06, 5.84020957e-06, 5.17101558e-06, 4.67206047e-06,
    4.05603968e-06, 3.59020631e-06, 3.10247441e-06, 2.67244309e-06,
    2.28662129e-06, 1.93747319e-06, 1.62182976e-06, 1.34825704e-06,
    1.10624166e-06, 8.98000375e-07, 7.18600375e-07, 5.67266029e-07,
    4.41094267e-07, 3.36986423e-07, 2.52624296e-07, 1.87556038e-07,
    1.11233073e-05, 9.64181878e-06, 8.46438919e-06, 7.62346437e-06,
    6.53648162e-06, 5.75338321e-06, 4.91960039e-06, 4.19704166e-06,
    3.55643025e-06, 2.98154301e-06, 2.46626814e-06, 2.02702551e-06,
    1.64205709e-06, 1.31573844e-06, 1.03796439e-06, 8.07208152e-07,
    6.17917485e-07, 4.63917256e-07, 3.40555004e-07, 2.48142171e-07,
    8.63323559e-07, 7.54657277e-07, 7.69296911e-07, 7.87836101e-07,
    7.22607066e-07, 7.20132624e-07, 6.80198259e-07, 6.46104524e-07,
    6.15286965e-07, 5.81338669e-07, 5.43768437e-07, 5.08809897e-07,
    4.71604022e-07, 4.34975111e-07, 3.97721427e-07, 3.61257533e-07,
    3.25705902e-07, 2.91155271e-07, 2.58150489e-07, 2.28126938e-07,
    1.30702031e-06, 1.12007471e-06, 1.14239434e-06, 1.16944380e-06,
    1.06255133e-06, 1.06024040e-06, 9.97441890e-07, 9.44175466e-07,
    8.96217346e-07, 8.43516329e-07, 7.85611838e-07, 7.32223300e-07,
    6.75573699e-07, 6.20224314e-07, 5.64209133e-07, 5.09768429e-07,
    4.57070698e-07, 4.06206689e-07, 3.57932409e-07, 3.14306622e-07,
    2.43734365e-06, 2.03596587e-06, 2.07754041e-06, 2.12367841e-06,
    1.90279033e-06, 1.89926349e-06, 1.77390705e-06, 1.66826179e-06,
    1.57385618e-06, 1.47090213e-06, 1.35909104e-06, 1.25746435e-06,
    1.15042028e-06, 1.04719182e-06, 9.43725825e-07, 8.44438644e-07,
    7.49607205e-07, 6.58999182e-07, 5.74060241e-07, 4.99568947e-07,
])
print(f"theory shape: {theory.shape}")  # should be (120,)
xi_plus_theory  = theory[:60].reshape(3, N_THETA)
xi_minus_theory = theory[60:].reshape(3, N_THETA)

# xi_plus: upper triangle of source bins
pair_idx = 0
for i in range(N_SOURCE_BINS):
    for j in range(i, N_SOURCE_BINS):
        for t_idx, th in enumerate(theta_centers):
            rows.append((b"xi_plus", i, j, th, float(xi_plus_theory[pair_idx, t_idx])))
        pair_idx += 1

# xi_minus: upper triangle of source bins
pair_idx = 0
for i in range(N_SOURCE_BINS):
    for j in range(i, N_SOURCE_BINS):
        for t_idx, th in enumerate(theta_centers):
            rows.append((b"xi_minus", i, j, th, float(xi_minus_theory[pair_idx, t_idx])))
        pair_idx += 1

spectra = np.array(rows, dtype=dt)
n_dv = len(spectra)
print(f"Total data vector length: {n_dv}")

# ---- Diagonal covariance ----
# Simple diagonal with 10% relative errors
cov_dt = np.dtype([
    ("spectrum_type0", "S10"),
    ("spectrum_type1", "S10"),
    ("zbin00", int),
    ("zbin01", int),
    ("zbin10", int),
    ("zbin11", int),
    ("separation0", float),
    ("separation1", float),
    ("value", float),
])
cov = np.zeros((n_dv, n_dv), dtype=cov_dt)
for ii in range(n_dv):
    for jj in range(n_dv):
        cov[ii, jj]["spectrum_type0"] = spectra[ii]["spectrum_type"]
        cov[ii, jj]["spectrum_type1"] = spectra[jj]["spectrum_type"]
        cov[ii, jj]["zbin00"]  = spectra[ii]["zbin0"]
        cov[ii, jj]["zbin01"]  = spectra[ii]["zbin1"]
        cov[ii, jj]["zbin10"]  = spectra[jj]["zbin0"]
        cov[ii, jj]["zbin11"]  = spectra[jj]["zbin1"]
        cov[ii, jj]["separation0"] = spectra[ii]["separation"]
        cov[ii, jj]["separation1"] = spectra[jj]["separation"]
        if ii == jj:
            cov[ii, jj]["value"] = (1.0 * abs(spectra[ii]["value"]))**2 + 1e-30

# ---- Write HDF5 ----
outfile = "dummy_configspace_dv.h5"
with h5py.File(outfile, "w") as f:
    f.create_dataset("spectra",    data=spectra)
    f.create_dataset("covariance", data=cov.flatten())
    f.create_dataset("nz_d",       data=nz_d)
    f.create_dataset("nz_s",       data=nz_s)

print(f"Written: {outfile}")
print(f"  spectra shape:    {spectra.shape}")
print(f"  covariance shape: {cov.shape}")
print(f"  nz_d shape:       {nz_d.shape}")
print(f"  nz_s shape:       {nz_s.shape}")
print(f"\nSpectrum types included:")
for t in [b"w_theta", b"gamma_t", b"xi_plus", b"xi_minus"]:
    n = np.sum(spectra["spectrum_type"] == t)
    print(f"  {t.decode()}: {n} rows")
