import jax.numpy as jnp
import numpy as np
from copy import copy
from jax import lax
import abc


class LikelihoodModule(object):
    def __init__(self):
        """
        Initialize module. Set what inputs are required for each output,
        including both parameters and variables that are expected to be
        passed through the state.
        """
        self.output_requirements = {}
        pass

    def get_dependencies(self, state_dependencies):
        self.original_requirements = copy(self.output_requirements)

        def get_dependencies_recurse(state_dependencies, requirements):
            deps = []
            for k in requirements:
                if k in state_dependencies:
                    deps.extend(
                        get_dependencies_recurse(
                            state_dependencies, state_dependencies[k]
                        )
                    )
                else:
                    deps.extend([k])

            return deps

        self.required_params = []

        for k in self.output_requirements:
            self.output_requirements[k] = list(
                np.unique(
                    get_dependencies_recurse(state_dependencies, state_dependencies[k])
                )
            )
            self.required_params.extend(self.output_requirements[k])

        self.required_params = np.unique(self.required_params)

    def check_cache(self, state, params_values):
        """
        Check to see whether parameters required for this module
        have changed from the last time this module was called.
        If they have then need to recompute.

        Args:
            params_values dict: Dictionary of parameter values

        Returns:
            bool : Whether the required parameters for this module
                   are the same as the last call.
        """

        params = jnp.array([params_values[p] for p in self.required_params])
        last_params = jnp.array([state["last_params"][p] for p in self.required_params])
        return jnp.any(params != last_params)

    def get_required_inputs(self, state):
        requirements = {r: state[r] for r in self.required_inputs}

        return requirements

    @abc.abstractmethod
    def compute(self, state, params_values):
        """Calculates stuff.

        Args:
            state dict: Inputs are contained here and outputs are written.
            params_values dict: Dictionary of parameter values.
        """
        pass

    def check_cache_and_compute(self, state, params_values):
        """Calculates stuff, assuming that there are things in the cache.
        Default method just calls compute. Overload method to do more.

        Args:
            state dict: Inputs are contained here and outputs are written.
            params_values dict: Dictionary of parameter values.
        """
        return self.compute(state, params_values)
