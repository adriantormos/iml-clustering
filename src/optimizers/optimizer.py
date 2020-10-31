import abc


class Optimizer(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'run') and
                callable(subclass.run) or
                NotImplemented)

    @abc.abstractmethod
    def __init__(self, optimizer_config, algorithm_config, output_path, verbose):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def run(self, values, labels):
        raise NotImplementedError('Method not implemented in interface class')