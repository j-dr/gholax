import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit
from jax.lax import scan
import yaml
from .likelihood import Likelihood

class GaussianLikelihood(Likelihood):
    """Abstract base class for Gaussian likelihoods.

    Manages a pipeline of LikelihoodModules, parameter classification
    (sampled/fixed/derived), analytic marginalization over linear nuisance
    parameters via the Woodbury identity, and dependency graph resolution.
    """

    def __init__(self, config, shared_params={}):
        """Initialize the Gaussian likelihood.

        Args:
            config: Likelihood configuration dict (from YAML).
            shared_params: Parameters shared between likelihoods (e.g., cosmology).
        """
        self.analytic_marginalization = config.get("analytic_marginalization", True)
        self.include_am_priors = config.get("include_am_priors", True)
        self.include_am_determinant = config.get("include_am_determinant", True)

        if self.analytic_marginalization:
            self.linear_params_filename = config["linear_params_filename"]

            self.linear_params_dict = yaml.load(
                open(self.linear_params_filename), Loader=yaml.SafeLoader
            )
            self.linear_params_names = np.array(
                [key for key in self.linear_params_dict.keys()]
            )
            self.linear_params_means = {
                key: float(self.linear_params_dict[key]["mean"])
                for key in self.linear_params_dict.keys()
            }
            self.linear_params_stds = jnp.array(
                [
                    self.linear_params_dict[key]["std"]
                    for key in self.linear_params_dict.keys()
                ]
            )
            self.Nlin = len(self.linear_params_dict)
        else:
            self.linear_params_dict = {}
            self.linear_params_means = {}
            self.linear_params_names = jnp.zeros(0)
            self.linear_params_stds = jnp.zeros(0)
            self.Nlin = 0

        # linear_params_dict must be set before super().__init__, which calls
        # _augment_free_params() to inject the linear params into free_params.
        super().__init__(config, shared_params)

    def _augment_free_params(self):
        """Add analytically marginalized linear nuisance params to free_params."""
        self.free_params.update(self.linear_params_dict)

    def setup_training_requirements(self, required_data):
        """Determine pipeline modules and parameters needed for training data generation.

        Args:
            required_data: List of state keys required for training.

        Returns:
            Tuple of (ordered list of required modules, list of required parameter names).
        """
        required_modules = {}

        def get_module_dependencies_recursive(requirement, likelihood_pipeline):
            deps = []
            for module in likelihood_pipeline:
                if requirement in module.original_requirements:
                    deps.append(module)
                    for r in module.original_requirements[requirement]:
                        deps.extend(
                            get_module_dependencies_recursive(r, likelihood_pipeline)
                        )
            return deps

        for r in required_data:
            required_modules[r] = get_module_dependencies_recursive(
                r, self.likelihood_pipeline
            )

        required_modules_all = []
        required_params_all = []
        for r in required_modules:
            for m in required_modules[r]:
                if m not in required_modules_all:
                    required_modules_all.append(m)
                    for p in m.required_params:
                        if p not in required_params_all:
                            required_params_all.append(p)
        # put modules in correct order
        idx = []
        for m in required_modules_all:
            idx.append(self.likelihood_pipeline.index(m))
        idx = np.sort(idx)
        required_modules_all = [self.likelihood_pipeline[i] for i in idx]

        self.training_pipeline = required_modules_all
        return required_modules_all, required_params_all

    def predict_model(
        self, params, params_am, return_state=False, apply_scale_mask=True,
        apply_window=True,
    ):
        """Run the full pipeline and return the model prediction vector.

        Args:
            params: Dict of sampled parameter values.
            params_am: Dict of analytically marginalized parameter values.
            return_state: If True, also return the full pipeline state.
            apply_scale_mask: If True, apply scale cuts to the prediction.
            apply_window: If True, apply window function convolution.

        Returns:
            Model prediction vector, or (prediction, state) if return_state.
        """
        params_all = params.copy()
        params_all.update(params_am)
        state, params_dict = self.setup_state_params(params_all)

        pipeline = self.likelihood_pipeline if apply_window else self.likelihood_pipeline[:-1]
        for module in pipeline:
            state = module.compute(state, params_dict)

        if apply_window:
            model = self.get_model_from_state(state)
        else:
            model = self.get_model_from_state_no_window(state)

        if apply_scale_mask and apply_window:
            model = model[self.observed_data_vector.scale_mask]

        if return_state:
            return model, state
        else:
            return model

    def generate_training_data(self, params):
        """Run the training pipeline and return the state with computed observables."""
        params_all = params.copy()
        state, params_dict = self.setup_state_params(params_all)

        for module in self.training_pipeline:
            state = module.compute(state, params_dict)

        return state

    def _build_all_spectra(self, field_types):
        """Build the per-spectrum-type index arrays used to gather the model
        vector out of the pipeline state.

        Args:
            field_types: The field_types dict of the data-vector module the
                subclass observes (they differ per data-vector type).
        """
        spectrum_info = self.observed_data_vector.spectrum_info
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
        """Extract the windowed model vector from the pipeline state."""
        dv = self.observed_data_vector
        model = []
        for t in dv.spectrum_types:
            def f(carry, i):
                return (carry, state[f"{t}_obs"][i])
            _, m_t = scan(f, 0, self.all_spectra[t])
            model.append(m_t.flatten())

        model = jnp.hstack(model)

        return model

    def get_model_from_state_no_window(self, state):
        """Extract the model prediction before window convolution (subclass override)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_model_from_state_no_window"
        )

    def compute_am(self, params):
        """Compute the log-likelihood with analytic marginalization over linear params.

        Uses the Woodbury identity to marginalize over linear nuisance parameters.

        Args:
            params: Dict of sampled parameter values.

        Returns:
            Tuple of (log-likelihood, non-marginalized chi2, template Jacobian).
        """
        params_am = self.linear_params_means

        @jit
        def predict_model_linear_pars(params_am):
            return self.predict_model(params, params_am)

        model = self.predict_model(params, params_am)
        templates = jacfwd(predict_model_linear_pars)(params_am)
        templates = jnp.array(list(templates.values()))

        diff = (
            self.observed_data_vector.measured_spectra[
                self.observed_data_vector.scale_mask
            ]
            - model
        )

        Va = jnp.dot(jnp.dot(templates, self.observed_data_vector.cinv), diff)
        Lab = jnp.dot(jnp.dot(templates, self.observed_data_vector.cinv), templates.T)

        if self.include_am_priors:
            Lab = jnp.add(Lab, jnp.diag(1.0 / self.linear_params_stds**2))

        Lab_inv = jnp.linalg.inv(Lab)

        # Compute the modified chi2
        lnL = -0.5 * jnp.dot(
            diff, jnp.dot(self.observed_data_vector.cinv, diff)
        )  # this is the "bare" lnL
        lnL += 0.5 * jnp.dot(
            Va, jnp.dot(Lab_inv, Va)
        )  # improvement in chi2 due to changing linear params

        lnL_nonmarg = jnp.copy(-2 * lnL)

        if self.include_am_determinant:
            lnL += -0.5 * jnp.log(jnp.linalg.det(Lab)) + 0.5 * self.Nlin * jnp.log(
                2 * jnp.pi
            )  # volume factor from the determinant

        return lnL, lnL_nonmarg, templates

    def compute_noam(self, params):
        """Compute the log-likelihood without analytic marginalization."""
        params_am = {}
        model = self.predict_model(params, params_am)

        diff = (
            self.observed_data_vector.measured_spectra[
                self.observed_data_vector.scale_mask
            ]
            - model
        )

        chi2 = jnp.dot(diff, jnp.dot(self.observed_data_vector.cinv, diff))
        lnL = -0.5 * chi2

        return lnL

    def compute(self, params):
        """Compute the log-likelihood, dispatching to AM or non-AM implementation."""
        if self.analytic_marginalization:
            lnL, lnL_nonmarg, templates = self.compute_am(params)
        else:
            lnL = self.compute_noam(params)

        return lnL

    def save_model(self, filename, params, params_am):
        """Save the model prediction to file via the data vector."""
        model = self.predict_model(params, params_am, apply_scale_mask=False)
        self.observed_data_vector.save_data_vector(filename, model)
        

