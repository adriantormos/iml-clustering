import abc
import numpy as np


class Algorithm(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'evaluate') and
                callable(subclass.evaluate) and
                hasattr(subclass, 'save') and
                callable(subclass.save) or
                NotImplemented)

    @abc.abstractmethod
    def __init__(self, config, output_path, verbose):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def train(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def evaluate(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError('Method not implemented in interface class')