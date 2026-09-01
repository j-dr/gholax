from ..theory.boltzmann import Boltzmann
from ..theory.expansion_history import ExpansionHistory
from ..theory.linear_growth import LinearGrowth
from ..theory.linear_power_spectrum import LinearPowerSpectrum
from ..theory.real_space_biased_tracer_spectra import (
    RealSpaceMatterPowerSpectrum,
    RealSpaceBiasedTracerSpectra,
    RealSpaceBiasExpansion,
)
from ..theory.ia import DensityShapeIA, ShapeShapeIA, RealSpaceIAExpansion
from .projection.limber import Limber
from .projection.kernels import ProjectionKernels
from .projection.delta_z import DeltaZ
from .projection.lensing_counterterm import LensingCounterterm
from .window.AngularPowerSpectrum_to_CorrelationFunction import AngularPowerSpectrum_to_CorrelationFunction
from .shear_systematics.multiplicative_bias import ShearMultiplicativeBias
from .gaussian_likelihood import GaussianLikelihood
from ..data_vector.two_point_spectrum import TwoPointSpectrum, field_types
import jax.numpy as jnp
from jax.lax import scan


class Nx2PTCorrelationFunction(GaussianLikelihood):
    def __init__(self, config):
        c = config["likelihood"]["Nx2PTCorrelationFunction"]

        self.zmin_proj = c.get("zmin_proj", 0.0001)
        self.zmax_proj = c.get("zmax_proj", 3.0)
        self.nz_proj = c.get("nz_proj", 200)
        self.zmin_pk = c.get("zmin_pk", 0.0001)
        self.zmax_pk = c.get("zmax_pk", 3.0)
        self.nz_pk = c.get("nz_pk", 50)
        self.kmin = c.get("kmin", 1e-3)
        self.kmax = c.get("kmax", 3.9355007545577743)
        self.nk = c.get("nk", 200)
        self.use_boltzmann = c.get("use_boltzmann", False)

        self.observed_data_vector = TwoPointSpectrum(
            zmin=self.zmin_proj,
            zmax=self.zmax_proj,
            nz=self.nz_proj,
            **c["data_vector"],
        )
        self.observed_data_vector.load_data()

        config_theory = config.get("theory", {})
        config_proj = config.get("projection", {})
        n_ell = config_proj.get("n_ell", 200)
        l_max = config_proj.get("l_max", 3001)
        self.k_cutoff = config_proj.get("k_cutoff", 2)

        spectrum_types = self.observed_data_vector.spectrum_types
        spectrum_info = self.observed_data_vector.spectrum_info

        type_map = {
            "w_theta": "c_dd",
            "gamma_t": "c_dk",
            "xi_plus": "c_kk",
            "xi_minus": "c_kk",
        }
        fourier_spectrum_types = []        # without c_dd, for RealSpaceBiasExpansion
        fourier_spectrum_types_limber = [] # with c_dd, for Limber/ProjectionKernels
        fourier_spectrum_info = {}
        
        for t in spectrum_types:
            ft = type_map[t]
            if ft not in fourier_spectrum_types_limber:
                fourier_spectrum_types_limber.append(ft)
                fourier_spectrum_info[ft] = dict(spectrum_info[t])
                fourier_spectrum_info[ft]["use_cross"] = True
                if ft == "c_dd":
                    bins = list(fourier_spectrum_info[ft]["bins0"])
                    fourier_spectrum_info[ft]["bin_pairs"] = [(i, j) for i in bins for j in bins]
                else:
                    fourier_spectrum_types.append(ft)
        
        # Update observed_data_vector for Limber/ProjectionKernels
        for ft in fourier_spectrum_types_limber:
            if ft not in self.observed_data_vector.spectrum_info:
                self.observed_data_vector.spectrum_info[ft] = dict(fourier_spectrum_info[ft])
            self.observed_data_vector.spectrum_info[ft]["use_cross"] = True
            if ft == "c_dd":
                bins = list(fourier_spectrum_info[ft]["bins0"])
                self.observed_data_vector.spectrum_info[ft]["bin_pairs"] = [(i, j) for i in bins for j in bins]
        
        
        # Add dummy c_dd to satisfy RealSpaceBiasExpansion's hardcoded check
        # without triggering p_gg computation
        if "c_dd" not in fourier_spectrum_info:
            fourier_spectrum_info["c_dd"] = {"use_cross": False, "bin_pairs": []}
        if "c_dk" not in fourier_spectrum_info:
            fourier_spectrum_info["c_dk"] = {"use_cross": False, "bin_pairs": []}

        self.configspace_spectrum_types = list(self.observed_data_vector.spectrum_types)
        # Update spectrum_types on data vector so ProjectionKernels computes the right kernels
        self.observed_data_vector.spectrum_types = (
            list(self.observed_data_vector.spectrum_types) + fourier_spectrum_types
        )

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
                LinearGrowth(
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    **config_theory.get("LinearGrowth", {}),
                ),
                DeltaZ(
                    self.observed_data_vector,
                    zmin=self.zmin_proj,
                    zmax=self.zmax_proj,
                    nz=self.nz_proj,
                    param_name="delta_z_source",
                    nz_name="nz_s",
                    **config_proj.get("DeltaZ", {}),
                ),
                ProjectionKernels(
                    self.observed_data_vector,
                    zmin=self.zmin_proj,
                    zmax=self.zmax_proj,
                    nz=self.nz_proj,
                    shifted_nz="nz_s",
                    **config_proj.get("ProjectionKernels", {}),
                ),
                RealSpaceBiasedTracerSpectra(
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    save_pij=False,
                    save_p_11_separately=True,
                    **config_theory.get("RealSpaceBiasedTracerSpectra", {}),
                ),  # this one saves just pmm
                RealSpaceMatterPowerSpectrum(
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    **config_theory.get("RealSpaceMatterPowerSpectrum", {}),
                ),  # this one saves p_ij
                RealSpaceBiasExpansion(
                    self.observed_data_vector,
                    fourier_spectrum_types,
                    fourier_spectrum_info,
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    k_cutoff=self.k_cutoff,
                    **config_theory.get("RealSpaceBiasExpansion", {}),
                ),
                DensityShapeIA(
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    **config_theory.get("DensityShapeIA", {}),
                ),
                ShapeShapeIA(
                    zmin=self.zmin_pk,
                    zmax=self.zmax_pk,
                    nz=self.nz_pk,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    **config_theory.get("ShapeShapeIA", {}),
                ),
                Limber(
                    self.observed_data_vector,
                    fourier_spectrum_types_limber,
                    fourier_spectrum_info,
                    zmin_proj=self.zmin_proj,
                    zmax_proj=self.zmax_proj,
                    nz_proj=self.nz_proj,
                    zmin_pk=self.zmin_pk,
                    zmax_pk=self.zmax_pk,
                    nz_pk=self.nz_pk,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    n_ell=n_ell,
                    l_max=l_max,
                    **config_proj.get("Limber", {}),
                ),
                
                LensingCounterterm(
                    self.observed_data_vector,
                    fourier_spectrum_types_limber,
                    fourier_spectrum_info,
                    zmin_proj=self.zmin_proj,
                    zmax_proj=self.zmax_proj,
                    nz_proj=self.nz_proj,
                    zmin_pk=self.zmin_pk,
                    zmax_pk=self.zmax_pk,
                    nz_pk=self.nz_pk,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    k_cutoff=self.k_cutoff,
                    n_ell=n_ell,
                    l_max=l_max,
                    **config_proj.get("LensingCounterterm", {}),
                ),
                ShearMultiplicativeBias(
                    self.observed_data_vector,
                    fourier_spectrum_types_limber,
                    fourier_spectrum_info,
                    **config_proj.get("ShearMultiplicativeBias", {}),
                ),
                AngularPowerSpectrum_to_CorrelationFunction(
                    self.observed_data_vector,
                    spectrum_types,
                    spectrum_info,
                    n_ell=n_ell,
                    l_max=l_max,
                    **config_proj.get("AngularPowerSpectrum_to_CorrelationFunction", {}),
                ),
            ]
        )

        no_ia = config_theory.get("RealSpaceIAExpansion", {}).get("no_ia", False)
        if not no_ia:
            self.likelihood_pipeline.extend(
                RealSpaceIAExpansion(
                            self.observed_data_vector,
                            fourier_spectrum_types_limber,
                            fourier_spectrum_info,
                            zmin=self.zmin_pk,
                            zmax=self.zmax_pk,
                            nz=self.nz_pk,
                            kmin=self.kmin,
                            kmax=self.kmax,
                            nk=self.nk,
                            **config_theory.get("RealSpaceIAExpansion", {}),
                        ),
            )

        self.n_modules = len(self.likelihood_pipeline)

        # calling this builds the dependency information
        super(Nx2PTCorrelationFunction, self).__init__(c, config["likelihood"].get('params', {}))
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
                        if t == "w_theta" and i != j:
                            continue
                        self.all_spectra[t].append((i, j))
                else:
                    self.all_spectra[t].append((i, i))

    def get_model_from_state(self, state):
        cf_keys = {
            "w_theta": "w",
            "gamma_t": "gamma",
            "xi_plus":  "xi_pos",
            "xi_minus": "xi_neg",
        }
        model = []
        for t in self.configspace_spectrum_types:
            cf_key = cf_keys[t]
            for (i, j) in self.all_spectra[t]:
                model.append(state[f"{cf_key}_{i}_{j}_obs"])
        return jnp.hstack(model)
