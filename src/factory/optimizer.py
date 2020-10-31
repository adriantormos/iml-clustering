from src.optimizers.optimizer import Optimizer
from src.optimizers.types.unsupervised import UnsupervisedOptimizer


class OptimizerFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_optimizer(optimizer_config, algorithm_config, output_path, verbose) -> Optimizer:
        name = optimizer_config['name']
        if name == 'unsupervised':
            optimizer = UnsupervisedOptimizer(optimizer_config, algorithm_config, output_path, verbose)
        else:
            raise Exception('The optimizers with name ' + name + ' does not exist')
        if issubclass(type(optimizer), Optimizer):
            return optimizer
        else:
            raise Exception('The optimizers does not follow the interface definition')
