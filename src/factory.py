from src.data.dataset import Dataset
from src.data.types.Kropt import KroptDataset
from src.algorithms.algorithm import Algorithm
from src.algorithms.types.kmeans import KmeansAlgorithm
from src.algorithms.types.bisecting_kmeans import BisectingKmeansAlgorithm
from src.algorithms.types.dbscan import DBSCANAlgorithm


class Factory():

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_dataset(config, verbose) -> Dataset:
        name = config['name']
        if name == 'kropt':
            dataset = KroptDataset(config, verbose)
        else:
            raise Exception('The dataset with name ' + name + ' does not exist')
        if issubclass(dataset, Dataset):
            return dataset

    @staticmethod
    def select_algorithm(config, output_path, verbose) -> Algorithm:
        name = config['name']
        if name == 'kmeans':
            algorithm = KmeansAlgorithm(config, output_path, verbose)
        elif name == 'bisecting_kmeans':
            algorithm = BisectingKmeansAlgorithm(config, output_path, verbose)
        elif name == 'dbscan':
            algorithm = DBSCANAlgorithm(config, output_path, verbose)
        else:
            raise Exception('The algorithm with name ' + name + ' does not exist')
        if issubclass(algorithm, Algorithm):
            return algorithm
