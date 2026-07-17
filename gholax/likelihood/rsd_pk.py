from ..theory.boltzmann import Boltzmann
from ..theory.expansion_history import ExpansionHistory
from ..theory.linear_growth_rate import LinearGrowthRate
from ..theory.linear_growth import LinearGrowth
from ..theory.linear_power_spectrum import LinearPowerSpectrum
from ..theory.spectral_equivalence import SpectralEquivalence
from ..theory.redshift_space_biased_tracer_spectra import (
    RedshiftSpaceBiasedTracerSpectra,
    RedshiftSpaceBiasExpansion,
)
from .window.redshift_space_multipole_power_spectrum_window import (
    RedshiftSpaceMultipolePowerSpectrumWindow,
)
from .gaussian_likelihood import GaussianLikelihood
from ..data_vector.redshift_space_multipoles import (
    RedshiftSpaceMultipoles,
    field_types,
    ALPHA_TYPES,
)
from ..theory.bao_alphas import BAOAlphas
import jax.numpy as jnp
from jax.lax import scan


class RSDPK(GaussianLikelihood):
    """Redshift-space distortion power spectrum multipole likelihood.

    Assembles a pipeline of theory modules to predict P_ell(k) multipoles
    and compare against observed data.
    """

    def __init__(self, config):
        """Initialize the RSD power spectrum likelihood from config."""
        c = config["likelihood"]["RSDPK"]

        self.zmin_proj = c.get("zmin_proj", 0.0001)
        self.zmax_proj = c.get("zmax_proj", 3.0)
        self.nz_proj = c.get("nz_proj", 200)
        self.zmin_pk = c.get("zmin_pk", 0.0001)
        self.zmax_pk = c.get("zmax_pk", 3.0)
        self.nz_pk = c.get("nz_pk", 30)
        self.kmin = c.get("kmin", 1e-3)
        self.kmax = c.get("kmax", 0.6 + 1e-3)
        self.nk = c.get("nk", 200)
        self.use_boltzmann = c.get("use_boltzmann", False)
        lens_bin_mapping = c.get("lens_bin_mapping", {})

        self.observed_data_vector = RedshiftSpaceMultipoles(
            zmin=self.zmin_proj,
            zmax=self.zmax_proj,
            nz=self.nz_proj,
            **c["data_vector"],
        )
        self.observed_data_vector.load_data()

        config_theory = config.get("theory", {})
        spectrum_types = self.observed_data_vector.spectrum_types
        spectrum_info = self.observed_data_vector.spectrum_info
        # BAO alphas are predicted directly from the expansion history; only
        # the full-shape types go through the bias expansion and window.
        alpha_types = [t for t in spectrum_types if t in ALPHA_TYPES]
        fs_types = [t for t in spectrum_types if t not in ALPHA_TYPES]
        if self.use_boltzmann:
            self.likelihood_pipeline = [
                Boltzmann(),
                LinearPowerSpectrum(
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    **config_theory.get("LinearPowerSpectrum", {}),
                ),
            ]
        else:
            self.likelihood_pipeline = []

        spectral_equiv_modules = []
        if "SpectralEquivalence" in config_theory:
            z_pk = jnp.linspace(self.zmin_pk, self.zmax_pk, self.nz_pk)
            spectral_equiv_modules.append(
                SpectralEquivalence(
                    z=z_pk,
                    **config_theory["SpectralEquivalence"],
                )
            )

        bao_modules = []
        if alpha_types:
            bao_modules.append(
                BAOAlphas(
                    spectrum_info,
                    alpha_types,
                    use_boltzmann=self.use_boltzmann,
                    **config_theory.get("BAOAlphas", {}),
                )
            )

        self.likelihood_pipeline.extend(
            [
                ExpansionHistory(
                    zmin=self.zmin_proj,
                    zmax=self.zmax_proj,
                    nz=self.nz_proj,
                    **config_theory.get("ExpansionHistory", {}),
                ),
                *bao_modules,
                *spectral_equiv_modules,
                LinearGrowthRate(
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    **config_theory.get("LinearGrowthRate", {}),
                ),
                LinearGrowth(
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    **config_theory.get("LinearGrowth", {}),
                ),
                RedshiftSpaceBiasedTracerSpectra(
                    spectrum_info["p_gg_ell"]["z_fid"],
                    spectrum_info["p_gg_ell"]["hz_fid"],
                    spectrum_info["p_gg_ell"]["chiz_fid"],
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    **config_theory.get("RedshiftSpaceBiasedTracerSpectra", {}),
                ),
                RedshiftSpaceBiasExpansion(
                    self.observed_data_vector,
                    fs_types,
                    spectrum_info,
                    spectrum_info["p_gg_ell"]["z_fid"],
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    lens_bin_mapping=lens_bin_mapping,
                    **config_theory.get("RedshiftSpaceBiasExpansion", {}),
                ),
                RedshiftSpaceMultipolePowerSpectrumWindow(
                    self.observed_data_vector,
                    fs_types,
                    spectrum_info,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    **config.get("RedshiftSpaceMultipolePowerSpectrumWindow", {}),
                ),
            ]
        )

        self.n_modules = len(self.likelihood_pipeline)

        # calling this builds the dependency information
        super(RSDPK, self).__init__(c, config["likelihood"].get('params', {}))
        self._build_all_spectra(field_types)

    def get_model_from_state_no_window(self, state):
        """Extract the pre-window theory P_ell(k) predictions from the state.

        BAO alpha types are skipped: they are scalar observables with no
        pre-window analog.
        """
        self.k_no_window = jnp.linspace(0, 0.6, 600)
        window_module = self.likelihood_pipeline[-1]
        k_theory = window_module.k
        model = []
        for t in self.observed_data_vector.spectrum_types:
            if t in ALPHA_TYPES:
                continue
            pl_pre = state[f"{t}{window_module.pl_tag}"]

            def f(carry, pl):
                # pl shape: (n_ell_multipoles, nk)
                def interp_one(carry2, pl_single):
                    return (carry2, jnp.interp(self.k_no_window, k_theory, pl_single))
                _, pl_interp = scan(interp_one, 0, pl)
                return (carry, pl_interp.flatten())

            _, pl_interp = scan(f, 0, pl_pre[self.all_spectra[t]])
            model.append(pl_interp.flatten())

        model = jnp.hstack(model)
        return model
