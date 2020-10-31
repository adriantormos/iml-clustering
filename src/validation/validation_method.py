import abc
import numpy as np


class ValidationMethod(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'evaluate') and
                callable(subclass.evaluate) or
                NotImplemented)

    @abc.abstractmethod
    def __init__(self, config, output_path, verbose, algorithm):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def evaluate(self, values):
        raise NotImplementedError('Method not implemented in interface class')