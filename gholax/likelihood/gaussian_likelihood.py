import jax.numpy as jnp
import numpy as np
from jax import jacobian, jit
from copy import copy
import yaml
import abc

class GaussianLikelihood(metaclass=abc.ABCMeta):
    def __init__(self, config, shared_params={}):
        self.analytic_marginalization = config.get("analytic_marginalization", True)
        self.include_am_priors = config.get("include_am_priors", True)
        self.include_am_determinant = config.get("include_am_determinant", True)
        self.output_requirements = {}

        if self.analytic_marginalization:
            self.linear_params_filename = config["linear_params_filename"]

            self.linear_params_dict = yaml.load(
                open(self.linear_params_filename), Loader=yaml.SafeLoader
            )
            self.linear_params_names = np.array(
                [key for key in self.linear_params_dict.keys()]
            )
            self.linear_params_means = {
                key: self.linear_params_dict[key]["mean"]
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

        #set up other parameters
        l_pars = copy(config["params"])
        l_pars.update(shared_params) #include parameters shared between likelihoods, e.g. cosmology.
        
        self.fixed_params = {}
        self.derived_params = {}
        self.sampled_params = {}
        
        # if 'prior' not in params, fix it to value/ref/derived.
        for p in l_pars:
            if "prior" not in l_pars[p]:
                if 'value' in l_pars[p]:
                    self.fixed_params[p] = l_pars[p]["value"]
                elif 'ref' in l_pars[p]:
                    self.fixed_params[p] = l_pars[p]["ref"]
                elif 'derived' in l_pars[p]:
                    self.derived_params[p] = l_pars[p]["derived"]
                else:
                    raise (
                                ValueError(
                                    f"Must specify either a prior, ref/value or derived for parameter {p}."
                                )
                            )
                if p in config['params']:
                    config["params"].pop(p)
            elif p not in config["params"]:
                config["params"][p] = l_pars[p]
        
        self.sampled_params = config["params"]
        self.free_params = copy(self.sampled_params)
        self.free_params.update(self.linear_params_dict)
        
        self.setup_params()
        self.build_dependency_graph()

    def setup_params(self):
        #parse arguments to derived parameters 
        for p in self.derived_params:
            fstr = self.derived_params[p]
            args = fstr.split(':')[0].split('lambda')[1]
            args = [a.strip() for a in args.split(',')]
            self.output_requirements[p] = args
            self.derived_params[p] = eval(self.derived_params[p])
            
        self.fixed_params.update({"NA": 0.0})
        self.n_free_params = len(self.free_params)
        self.n_fixed_params = len(self.fixed_params)

        # ensure ordering is the same as when sampling
        param_dict = dict(zip(self.free_params.keys(), jnp.zeros(self.n_free_params)))
        _, param_dict = self.setup_state_params(param_dict)

        self.free_param_names = list(param_dict.keys())

        for module in self.likelihood_pipeline:
            if hasattr(module, "indexed_params"):
                if type(module.indexed_params) is dict:
                    param_indices = {}
                    for k in module.indexed_params:
                        if type(module.indexed_params[k]) is list:
                            param_indices[k] = []
                            for l in range(len(module.indexed_params[k])):  # noqa: E741
                                param_indices[k].append(
                                    jnp.zeros(
                                        module.indexed_params[k][l].shape, dtype=int
                                    )
                                )
                                for i in range(param_indices[k][l].shape[0]):
                                    for j in range(param_indices[k][l].shape[1]):
                                        param_indices[k][l] = (
                                            param_indices[k][l]
                                            .at[i, j]
                                            .set(
                                                self.free_param_names.index(
                                                    module.indexed_params[k][l][i, j]
                                                )
                                            )
                                        )

                        else:
                            param_indices[k] = jnp.zeros(
                                module.indexed_params[k].shape, dtype=int
                            )
                            for i in range(param_indices[k].shape[0]):
                                for j in range(param_indices[k].shape[1]):
                                    param_indices[k] = (
                                        param_indices[k]
                                        .at[i, j]
                                        .set(
                                            self.free_param_names.index(
                                                module.indexed_params[k][i, j]
                                            )
                                        )
                                    )
                else:
                    param_indices = jnp.zeros(module.indexed_params.shape, dtype=int)
                    for i in range(param_indices.shape[0]):
                        for j in range(param_indices.shape[1]):
                            param_indices = param_indices.at[i, j].set(
                                self.free_param_names.index(module.indexed_params[i, j])
                            )

                module.param_indices = param_indices

    def build_dependency_graph(self):
        state_dependencies = {}
        for module in self.likelihood_pipeline:
            state_dependencies.update(module.output_requirements)
        
        state_dependencies.update(self.output_requirements)

        for module in self.likelihood_pipeline:
            module.get_dependencies(state_dependencies)

            
    def setup_training_requirements(self, required_data):
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

    def setup_state_params(self, params_values):
        for p in self.fixed_params:
            params_values[p] = jnp.array(self.fixed_params[p])
        
        for p in self.derived_params:
            params_values[p] = jnp.array(self.derived_params[p](*[params_values[arg] for arg in self.output_requirements[p]]))

        # sort param dict by keys so that always have predetermined
        # param ordering in all likelihood modules. Important
        # because jit can change order otherwise.
        param_dict = {}
        sorted_keys = list(params_values.keys())
        sorted_keys.sort()
        for p in sorted_keys:
            param_dict[p] = params_values[p]

        state = {}
        state["derived"] = {}

        return state, param_dict

    def predict_model(
        self, params, params_am, return_state=False, apply_scale_mask=True
    ):
        params_all = params.copy()
        params_all.update(params_am)
        state, params_dict = self.setup_state_params(params_all)

        for module in self.likelihood_pipeline:
            state = module.compute(state, params_dict)

        model = self.get_model_from_state(state)

        if apply_scale_mask:
            model = model[self.observed_data_vector.scale_mask]

        if return_state:
            return model, state
        else:
            return model

    def generate_training_data(self, params):
        params_all = params.copy()
        state, params_dict = self.setup_state_params(params_all)

        for module in self.training_pipeline:
            state = module.compute(state, params_dict)

        return state

    @abc.abstractmethod
    def get_model_from_state(
        self,
        state,
    ):
        return

    def compute_am(self, params):
        params_am = self.linear_params_means

        @jit
        def predict_model_linear_pars(params_am):
            return self.predict_model(params, params_am)

        model = self.predict_model(params, params_am)
        templates = jacobian(predict_model_linear_pars)(params_am)
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
        if self.analytic_marginalization:
            lnL, lnL_nonmarg, templates = self.compute_am(params)
        else:
            lnL = self.compute_noam(params)

        return lnL

    def save_model(self, filename, params, params_am):
        model = self.predict_model(params, params_am, apply_scale_mask=False)
        self.observed_data_vector.save_data_vector(filename, model)
        

