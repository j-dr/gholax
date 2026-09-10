"""Acoustic scale theta_* = r_s(z_*) / D_M(z_*) in pure JAX.

Both integrals use the full background (photons, massless + massive
neutrinos as N_MASSIVE=3 degenerate species like ExpansionHistory, w0wa
dark energy). z_* is the Hu & Sugiyama 1996 fit plus a
small correction calibrated to CAMB 2.0 / Recfast (the recombination code
behind the DESI DR2 compressed CMB prior). Validated against CAMB over
omega_b h^2 in [0.0210, 0.0238], omega_c h^2 in [0.100, 0.135], H0 in
[62, 72] and three (w0, wa): theta_* to 4e-5 (mean offset, 1e-6 scatter).
"""
import jax
import jax.numpy as jnp
from scipy.special import roots_laguerre

from .linear_growth import (
    neutrino_density_ratio, dark_energy_density,
    _T_NU0_EV, _OMEGA_GAMMA_H2, _OMEGA_NU_REL_H2_PER_SPECIES,
)

C_KMS = 2.99792458e5
_GL_NODES, _GL_WEIGHTS = [jnp.asarray(x) for x in roots_laguerre(64)]
N_EFF = 3.044
# degenerate massive species sharing sum m_nu; matches ExpansionHistory
N_MASSIVE = 3


def z_star_hs96(ombh2, omh2):
    """Hu & Sugiyama 1996 photon-decoupling redshift fit."""
    g1 = 0.0783 * ombh2**-0.238 / (1 + 39.5 * ombh2**0.763)
    g2 = 0.560 / (1 + 21.1 * ombh2**1.81)
    return 1048 * (1 + 0.00124 * ombh2**-0.738) * (1 + g1 * omh2**g2)


def z_star(ombh2, omch2, mnu):
    """z_* calibrated to CAMB/Recfast: HS96 minus a linear correction in
    (omega_b, omega_cb); residual 0.007 in z_* (4e-6 in theta_*)."""
    zs = z_star_hs96(ombh2, ombh2 + omch2 + mnu / 93.14)
    wm = ombh2 + omch2
    return zs - (
        1.99937 - 0.100976 * (ombh2 - 0.0224) / 0.001
        - 0.0115137 * (wm - 0.142) / 0.01
    )


def z_drag(ombh2, omch2):
    """Baryon drag redshift: Aizpuru+21 eq. A2 fit (CLASS+HyRec calibrated)
    plus the constant +0.083 CAMB/Recfast offset (scatter <1e-3 over the
    DESI BAO+CMB posterior). omega_m here is omega_b + omega_c."""
    wm = ombh2 + omch2
    zd = (1 + 428.169 * ombh2**0.256459 * wm**0.616388 + 925.56 * wm**0.751615) / wm**0.714129
    return zd + 0.083


def _a2E(H0, ombh2, omch2, w0, wa, mnu, n_massive=N_MASSIVE, nnu=N_EFF):
    """a^2 E(a) with photons, massless neutrinos, n_massive degenerate
    massive species sharing mnu, and w0wa dark energy (CAMB conventions).
    Returned in this form so nothing overflows float32 (or its gradient)
    deep in the radiation era."""
    h = H0 / 100.0
    omega_rad = (_OMEGA_GAMMA_H2 + (nnu - n_massive) * _OMEGA_NU_REL_H2_PER_SPECIES) / h**2
    omega_cb = (ombh2 + omch2) / h**2
    m_per = mnu / n_massive
    omega_nu_rel = n_massive * _OMEGA_NU_REL_H2_PER_SPECIES / h**2

    def omega_nu_massive_a4(a):
        y = m_per * a / _T_NU0_EV
        return omega_nu_rel * neutrino_density_ratio(y, _GL_NODES, _GL_WEIGHTS)

    omega_de = 1.0 - omega_cb - omega_nu_massive_a4(1.0) - omega_rad

    def a2E(a):
        return jnp.sqrt(
            omega_rad + omega_cb * a + omega_nu_massive_a4(a)
            + omega_de * dark_energy_density(a, w0, wa) * a**4
        )
    return a2E


def sound_horizon(H0, ombh2, omch2, w0, wa, mnu, z, n_int=4096, n_massive=N_MASSIVE):
    """Comoving sound horizon r_s(z) in Mpc, integrated on a log(a) grid."""
    a2E = _a2E(H0, ombh2, omch2, w0, wa, mnu, n_massive)
    la = jnp.linspace(jnp.log(1e-8), -jnp.log1p(z), n_int)
    a = jnp.exp(la)
    R = 3.0 * ombh2 * a / (4.0 * _OMEGA_GAMMA_H2)
    cs = C_KMS / jnp.sqrt(3.0 * (1.0 + R))
    # int c_s da / (a^2 H) = int c_s a dln(a) / (a^2 E H0)
    return jnp.trapezoid(cs * a / (jax.vmap(a2E)(a) * H0), la)


def comoving_distance(H0, ombh2, omch2, w0, wa, mnu, z, n_int=4096, n_massive=N_MASSIVE):
    """Comoving distance D_M(z) in Mpc (flat), integrated on a log(1+z) grid."""
    a2E = _a2E(H0, ombh2, omch2, w0, wa, mnu, n_massive)
    lz = jnp.linspace(0.0, jnp.log1p(z), n_int)
    a = jnp.exp(-lz)
    # int dz / E = int (1+z) dln(1+z) / E = int a dln(1+z) / (a^2 E)
    return C_KMS / H0 * jnp.trapezoid(a / jax.vmap(a2E)(a), lz)


def sound_horizon_drag(H0, ombh2, omch2, w0=-1.0, wa=0.0, mnu=0.06, n_int=4096, n_massive=N_MASSIVE):
    """r_d = r_s(z_drag) in Mpc from the integral; +3.4e-5 vs CAMB/Recfast
    with 3e-7 scatter over the DESI DR2 BAO+CMB posterior."""
    return sound_horizon(H0, ombh2, omch2, w0, wa, mnu, z_drag(ombh2, omch2), n_int, n_massive)


def theta_star(H0, ombh2, omch2, w0=-1.0, wa=0.0, mnu=0.06, n_int=4096, n_massive=N_MASSIVE):
    """Acoustic angular scale theta_* (not 100 theta_*). n_massive=3 degenerate
    species (ExpansionHistory convention); pass 1 for CAMB/DESI-chain
    convention (differs by -1.9e-4 relative, 0.8 Planck sigma)."""
    zs = z_star(ombh2, omch2, mnu)
    return (sound_horizon(H0, ombh2, omch2, w0, wa, mnu, zs, n_int, n_massive)
            / comoving_distance(H0, ombh2, omch2, w0, wa, mnu, zs, n_int, n_massive))
