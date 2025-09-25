import abc


class DataVector(metaclass=abc.ABCMeta):
    def __init__(self, config):
        pass

    @abc.abstractmethod
    def load_requirements(self, data_type):
        pass

    @abc.abstractmethod
    def load_data(self):
        """Loads the required data."""
        pass

    @abc.abstractmethod
    def load_covariance_matrix(self):
        pass

    @abc.abstractmethod
    def save_data_vector(self, model, filename):
        """Saves the model to a file.

        Args:
            model (array-like): The model data to be saved.
            filename (str): The name of the file where the model will be saved.
        """
        pass