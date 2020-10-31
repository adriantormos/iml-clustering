from src.data.dataset import Dataset
from src.data.types.kropt import KroptDataset
from src.data.types.hypothyroid import HypothyroidDataset
from src.data.types.breast import BreastDataset
from src.data.types.test import TestDataset
from src.algorithms.algorithm import Algorithm
from src.algorithms.types.kmeans import KMeansAlgorithm
from src.algorithms.types.kmedians import KMediansAlgorithm
from src.algorithms.types.bisecting_kmeans import BisectingKMeansAlgorithm
from src.algorithms.types.dbscan import DBSCANAlgorithm
from src.algorithms.types.fcm import FCMAlgorithm


class Factory():

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_dataset(config, verbose) -> Dataset:
        name = config['name']
        if name == 'kropt':
            dataset = KroptDataset(config, verbose)
        elif name == 'hypothyroid':
            dataset = HypothyroidDataset(config, verbose)
        elif name == 'breast':
            dataset = BreastDataset(config, verbose)
        elif name == 'test':
            dataset = TestDataset(config, verbose)
        else:
            raise Exception('The dataset with name ' + name + ' does not exist')
        if issubclass(type(dataset), Dataset):
            return dataset
        else:
            raise Exception('The dataset does not follow the interface definition')

    @staticmethod
    def select_algorithm(config, output_path, verbose) -> Algorithm:
        name = config['name']
        if name == 'kmeans':
            algorithm = KMeansAlgorithm(config, output_path, verbose)
        elif name == 'bisecting_kmeans':
            algorithm = BisectingKMeansAlgorithm(config, output_path, verbose)
        elif name == 'kmedians':
            algorithm = KMediansAlgorithm(config, output_path, verbose)
        elif name == 'dbscan':
            algorithm = DBSCANAlgorithm(config, output_path, verbose)
        elif name == 'fcm':
            algorithm = FCMAlgorithm(config, output_path, verbose)
        else:
            raise Exception('The algorithm with name ' + name + ' does not exist')
        if issubclass(type(algorithm), Algorithm):
            return algorithm
        else:
            raise Exception('The algorithm does not follow the interface definition')
