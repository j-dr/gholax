from ..theory.boltzmann import Boltzmann
from ..theory.expansion_history import ExpansionHistory
from ..theory.linear_growth_rate import LinearGrowthRate
from ..theory.linear_growth import LinearGrowth
from ..theory.linear_power_spectrum import LinearPowerSpectrum
from ..theory.redshift_space_biased_tracer_spectra import (
    RedshiftSpaceBiasedTracerSpectra,
    RedshiftSpaceBiasExpansion,
)
from .window.redshift_space_multipole_power_spectrum_window import (
    RedshiftSpaceMultipolePowerSpectrumWindow,
)
from .gaussian_likelihood import GaussianLikelihood
from ..data_vector.redshift_space_multipoles import RedshiftSpaceMultipoles, field_types
import jax.numpy as jnp
from jax.lax import scan


class RSDPK(GaussianLikelihood):
    def __init__(self, config):
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

        self.likelihood_pipeline.extend(
            [
                ExpansionHistory(
                    zmin=self.zmin_proj,
                    zmax=self.zmax_proj,
                    nz=self.nz_proj,
                    **config_theory.get("ExpansionHistory", {}),
                ),
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
                    spectrum_types,
                    spectrum_info,
                    spectrum_info["p_gg_ell"]["z_fid"],
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    **config_theory.get("RedshiftSpaceBiasExpansion", {}),
                ),
                RedshiftSpaceMultipolePowerSpectrumWindow(
                    self.observed_data_vector,
                    spectrum_types,
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
        self.all_spectra = {}

        for t in self.observed_data_vector.spectrum_types:
            self.all_spectra[t] = []
            for ii, i in enumerate(spectrum_info[t]["bins0"]):
                if spectrum_info[t]["use_cross"]:
                    if field_types[t][0] == field_types[t][1]:
                        bins1 = spectrum_info[t]["bins1"][ii:]
                    else:
                        bins1 = spectrum_info[t]["bins1"][:]

                    for j in bins1:
                        self.all_spectra[t].append(
                            i * spectrum_info[t]["n_bins1_tot"] + j
                        )
                else:
                    self.all_spectra[t].append(i)
            self.all_spectra[t] = jnp.array(self.all_spectra[t])

    def get_model_from_state(self, state):
        dv = self.observed_data_vector
        model = []
        for t in dv.spectrum_types:
            f = lambda carry, i: (carry, state[f"{t}_obs"][i])
            _, m_t = scan(f, 0, self.all_spectra[t])
            model.append(m_t.flatten())

        model = jnp.hstack(model)

        return model
