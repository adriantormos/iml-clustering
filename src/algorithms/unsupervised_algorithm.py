import abc
import numpy as np
from src.algorithms.algorithm import Algorithm


class UnsupervisedAlgorithm(Algorithm):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'run') and
                callable(subclass.run) or
                NotImplemented)

    # Main methods

    @abc.abstractmethod
    def __init__(self, config, output_path, verbose):
        raise NotImplementedError('Method not implemented in interface class')

    def train(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        output_labels = self.run(values)
        return self.merge_labels(labels, output_labels)

    @abc.abstractmethod
    def run(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented in interface class')

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Unsupervised learning not implements evaluate method')

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError('Method not implemented in interface class')

    # Auxiliary methods

    def merge_labels(self, labels, output_labels):
        if len(labels) != len(output_labels):
            raise Exception('The unsupervised algorithm extracted less classes than the ones in the dataset')
        unique1, counts1 = np.unique(labels, return_counts=True)
        counts1, unique1 = zip(*sorted(zip(counts1, unique1), reverse=True))
        unique2, counts2 = np.unique(output_labels, return_counts=True)
        counts2, unique2 = zip(*sorted(zip(counts2, unique2), reverse=True))
        if len(unique1) != len(unique2):
            raise Exception('The unsupervised algorithm extracted less classes than the ones in the dataset')
        #print(unique1, counts1)
        #print(unique2, counts2)
        result_labels = np.zeros(len(labels))
        for index, _class1 in enumerate(unique1):
            _class2 = unique2[index]
            result_labels[output_labels == _class2] = _class1
        #print(labels)
        #print(output_labels)
        #print(result_labels)
        return result_labels
