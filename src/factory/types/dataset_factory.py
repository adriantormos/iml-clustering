from src.data.dataset import Dataset
from src.data.types.kropt import KroptDataset
from src.data.types.hypothyroid import HypothyroidDataset
from src.data.types.breast import BreastDataset


class DatasetFactory:

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
        else:
            raise Exception('The dataset with name ' + name + ' does not exist')
        if issubclass(dataset, Dataset):
            return dataset
