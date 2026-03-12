import jax.numpy as jnp
import numpy as np
from jax import jacobian, jit
from copy import copy
import yaml
import abc

class Likelihood(metaclass=abc.ABCMeta):
    """Abstract base class for likelihoods.

    Manages a pipeline of LikelihoodModules, parameter classification
    (sampled/fixed/derived), and dependency graph resolution.
    """

    def __init__(self, config, shared_params={}):
        """Initialize the likelihood.

        Args:
            config: Likelihood configuration dict (from YAML).
            shared_params: Parameters shared between likelihoods (e.g., cosmology).
        """
        self.output_requirements = {}
        config_params = config.get("params",{})
        l_pars = copy(config_params)
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
                if p in config_params:
                    config_params.pop(p)
            elif p not in config_params:
                config_params[p] = l_pars[p]
        
        self.sampled_params = config_params
        self.free_params = copy(self.sampled_params)
        
        self.setup_params()
        self.build_dependency_graph()

    def setup_params(self):
        """Parse derived parameters, build parameter index maps for pipeline modules."""
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
        """Resolve transitive dependencies across all pipeline modules."""
        state_dependencies = {}
        for module in self.likelihood_pipeline:
            state_dependencies.update(module.output_requirements)
        
        state_dependencies.update(self.output_requirements)

        for module in self.likelihood_pipeline:
            module.get_dependencies(state_dependencies)
         
    def setup_state_params(self, params_values):
        """Merge fixed/derived params and sort, returning (state, param_dict)."""
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






        

