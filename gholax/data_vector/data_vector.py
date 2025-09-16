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
