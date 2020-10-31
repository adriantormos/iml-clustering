from src.data.dataset import Dataset
import numpy as np


class TestDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        super(TestDataset, self).__init__(config, verbose)
        self.values = np.asarray([[0, 0.1, 0.2], [0.9, 0.8, 0.7], [0.05, 0.1, 0.1]])
        self.labels = np.asarray([0, 0, 1])

    def get_raw_data(self) -> (np.ndarray, np.ndarray):
        return self.values, self.labels

    def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
        return self.values, self.labels
