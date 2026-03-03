import abc


class DataVector(metaclass=abc.ABCMeta):
    """Abstract base class for observed data vectors.

    Subclasses must implement methods for loading the data vector, its
    auxiliary requirements (e.g. redshift distributions), the covariance
    matrix, and saving a model prediction to disk.
    """

    def __init__(self, config):
        pass

    @abc.abstractmethod
    def load_requirements(self, data_type):
        """Load auxiliary data required by the data vector (e.g. n(z), windows)."""
        pass

    @abc.abstractmethod
    def load_data(self):
        """Load the observed data vector from disk."""
        pass

    @abc.abstractmethod
    def load_covariance_matrix(self):
        """Load and invert the covariance matrix for the data vector."""
        pass

    @abc.abstractmethod
    def save_data_vector(self, model, filename):
        """Saves the model to a file.

        Args:
            model (array-like): The model data to be saved.
            filename (str): The name of the file where the model will be saved.
        """
        pass
