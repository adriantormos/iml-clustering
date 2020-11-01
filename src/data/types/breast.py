from src.data.dataset import Dataset
from src.auxiliary.file_methods import load_arff
import numpy as np
import pandas as pd
from src.auxiliary.preprocessing_methods import min_max_normalize


class BreastDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        super(BreastDataset, self).__init__(config, verbose)
        self.data, self.meta = load_arff('breast-w')
        self.data = pd.DataFrame(self.data)
        self.verbose = verbose
        self.features = self.data.columns[:-1]
        self.classes_to_numerical = config['classes_to_numerical']
        self.class_feature = 'Class'
        self.preprocessed_data = self.preprocess_dataset()

    def get_raw_data(self) -> (np.ndarray, np.ndarray):
        values = self.data.loc[:, self.data.columns != self.class_feature].to_numpy()
        labels = self.data[self.class_feature].to_numpy()
        return values, labels

    def get_raw_dataframe(self) -> pd.DataFrame:
        return self.data

    def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
        values = self.preprocessed_data.loc[:, self.preprocessed_data.columns != self.class_feature].to_numpy()
        labels = self.preprocessed_data[self.class_feature].to_numpy()
        return values, labels

    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        return self.preprocessed_data

    # Auxiliary methods

    def preprocess_dataset(self):
        data = self.data

        if self.verbose:
            print('Started data preprocessing')

        # Delete features with more than half of samples with NaN values
        if self.verbose:
            nan_count = data.isnull().sum().sum()
            print('    ', 'Total number of NaNs: ', nan_count, '; relative: ',
                  (nan_count * 100) / (len(data.index) * len(data.columns)), '%')

        columns_to_drop = []
        for feature_index in data.columns:
            nan_count = data[feature_index].isnull().sum()
            if nan_count > (len(data.index) / 2):
                columns_to_drop.append(feature_index)
        data.drop(columns=columns_to_drop, inplace=True)
        self.features = data.columns[:-1]
        if self.verbose:
            print('    ', 'Deleted because of too many NaN values the features with name:', columns_to_drop)

        # replace the NaN values by the mean and normalize
        for feature_index in self.features:
            feature = data[feature_index]

            # replace the NaN values by the mean
            nan_indexes = data.index[feature.isnull()].tolist()
            feature = feature.to_numpy()
            feature_without_nans = np.delete(feature, nan_indexes)
            mean = np.mean(feature_without_nans)
            feature[nan_indexes] = mean

            # do normalization
            normalized_feature = min_max_normalize(feature)
            data[feature_index] = normalized_feature

        # Convert classes to numerical
        data[self.class_feature] = data[self.class_feature].str.decode("utf-8")
        data[self.class_feature] = data[self.class_feature].map(self.classes_to_numerical)

        if self.verbose:
            print('Finished data preprocessing')

        return data
