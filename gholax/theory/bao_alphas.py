from ..util.likelihood_module import LikelihoodModule
import jax.numpy as jnp
import numpy as np

# speed of light in km/s
C_KMS = 2.99792458e5


def sound_horizon_aubourg(omch2, ombh2, mnu):
    """Sound horizon at the drag epoch, r_d, in Mpc.

    Aubourg et al. 2015 (arXiv:1411.1074, eq. 16) fitting formula, accurate
    to ~0.02% for cosmologies near Planck. Pure JAX, differentiable in all
    inputs; insensitive to (w0, wa) since r_d is set in the early universe.

    Args:
        omch2: Physical cold dark matter density omega_c h^2.
        ombh2: Physical baryon density omega_b h^2.
        mnu: Sum of neutrino masses in eV.

    Returns:
        r_d in Mpc.
    """
    omega_nu = mnu / 93.14
    omega_cb = omch2 + ombh2
    return (
        55.154
        * jnp.exp(-72.3 * (omega_nu + 0.0006) ** 2)
        / (omega_cb**0.25351 * ombh2**0.12807)
    )


class BAOAlphas(LikelihoodModule):
    """Predict BAO dilation parameters (alphas) from the expansion history.

    Conventions (all quantities physical, fiducials from the data file):
        alpha_par  = (Hz_fid * rd_fid) / (H(zeff) * rd)
        alpha_perp = (D_M(zeff) * rd_fid) / (DM_fid * rd)
        alpha_iso  = (D_V(zeff) / rd) / (DV_fid / rd_fid)
    with H = H0 * E(z) in km/s/Mpc, D_M = chi/h in Mpc (flat; chi_z state is
    Mpc/h), D_V = (D_M^2 c z / H)^(1/3) in Mpc, and rd in Mpc from
    sound_horizon_aubourg (or boltzmann_results.rs_drag() in boltzmann mode).

    Writes one state key per requested alpha type, ``{type}_obs``, an array
    indexable by raw tracer bin id so GaussianLikelihood.get_model_from_state
    can gather it exactly like a windowed spectrum block.
    """

    # fiducial dataset (loaded into spectrum_info by the data vector) per type
    _fid_keys = {
        "alpha_iso": "DV_fid_bao",
        "alpha_par": "Hz_fid_bao",
        "alpha_perp": "DM_fid_bao",
    }

    def __init__(self, spectrum_info, alpha_types, use_boltzmann=False, **config):
        """Initialize from the data vector's spectrum_info.

        Args:
            spectrum_info: The data vector's spectrum_info dict; alpha entries
                must already carry bins0, zeff_bao, rd_fid, and the per-type
                fiducial array (loaded by load_requirements).
            alpha_types: List of alpha spectrum types present in the data
                vector (subset of alpha_iso, alpha_par, alpha_perp).
            use_boltzmann: If True, take r_d from the live CLASS instance in
                state (non-differentiable); otherwise use the Aubourg formula.
        """
        self.alpha_types = list(alpha_types)
        self.use_boltzmann = use_boltzmann

        self.bins = {}
        self.zeff = {}
        self.fid = {}
        rd_fid = None
        for t in self.alpha_types:
            info = spectrum_info[t]
            self.bins[t] = [int(b) for b in info["bins0"]]
            bins = np.array(self.bins[t])
            # aux arrays are indexed by absolute tracer bin id
            self.zeff[t] = jnp.asarray(info["zeff_bao"])[bins]
            self.fid[t] = jnp.asarray(info[self._fid_keys[t]])[bins]
            rd_fid = float(info["rd_fid"])
        self.rd_fid = rd_fid

        self.output_requirements = {}
        if self.use_boltzmann:
            self.output_requirements["rd"] = ["boltzmann_results"]
        else:
            self.output_requirements["rd"] = ["omch2", "ombh2", "mnu"]
        for t in self.alpha_types:
            self.output_requirements[f"{t}_obs"] = ["rd", "H0", "chi_z", "e_z"]

    def compute(self, state, params_values):
        """Compute r_d and the alpha predictions, writing them to state."""
        if self.use_boltzmann:
            rd = state["boltzmann_results"].rs_drag()
        else:
            rd = sound_horizon_aubourg(
                params_values["omch2"], params_values["ombh2"], params_values["mnu"]
            )
        state["rd"] = rd

        h = params_values["H0"] / 100.0
        for t in self.alpha_types:
            e_z_eff = jnp.interp(self.zeff[t], state["z_limber"], state["e_z_limber"])
            chi_z_eff = jnp.interp(
                self.zeff[t], state["z_limber"], state["chi_z_limber"]
            )
            H = params_values["H0"] * e_z_eff
            DM = chi_z_eff / h
            if t == "alpha_par":
                alpha = (self.fid[t] * self.rd_fid) / (H * rd)
            elif t == "alpha_perp":
                alpha = (DM * self.rd_fid) / (self.fid[t] * rd)
            else:
                DV = (DM**2 * C_KMS * self.zeff[t] / H) ** (1.0 / 3.0)
                alpha = (DV / rd) / (self.fid[t] / self.rd_fid)

            obs = jnp.zeros(max(self.bins[t]) + 1)
            state[f"{t}_obs"] = obs.at[jnp.array(self.bins[t])].set(alpha)

        return state
