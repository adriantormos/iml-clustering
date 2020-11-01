from src.data.dataset import Dataset
from src.auxiliary.file_methods import load_arff
import numpy as np
import pandas as pd


class BreastDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        super(BreastDataset, self).__init__(config, verbose)
        self.data, self.meta = load_arff('breast-w')
        self.data = pd.DataFrame(self.data)
        self.verbose = verbose
        self.preprocessed_data = self.preprocess_dataset()

    def get_raw_data(self) -> (np.ndarray, np.ndarray):
        values = self.data.columns[:-1].to_numpy()
        labels = self.data.columns[-1].to_numpy()
        return values, labels

    def get_raw_dataframe(self) -> pd.DataFrame:
        return self.data

    def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
        values = self.preprocessed_data[:,:-1]
        labels = self.preprocessed_data[:,-1]
        return values, labels

    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        return self.data

    # Auxiliary methods

    def preprocess_dataset(self):
        if self.verbose:
            print('Started data preprocessing')

        data = self.data

        data.replace([np.inf, -np.inf], np.nan)
        df = data.dropna()  # Remove NaN elements
        if self.verbose:
            print('    df with no Nan=', np.shape(df))

        type_list = np.array(self.meta.types())

        nominal_bool = (type_list == 'numeric')
        nominal_columns = np.array(self.meta.names())[nominal_bool]

        X = df[nominal_columns].to_numpy()

        # Scaling
        mean_vec = np.matrix(np.mean(X, axis=0))
        n, m = X.shape
        M = np.repeat(mean_vec, n, axis=0)
        M = np.array(M)
        Xc = (X - M)  # Xc = X centered
        # sd = 1
        sd = np.std(Xc, axis=0)
        Xcs = Xc / sd  # Xcs = X centered and scaled

        # Problem with division by 0
        Xcs = np.nan_to_num(Xcs)

        if self.verbose:
            print('Finished data preprocessing')

        return Xcs
