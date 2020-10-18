from src.data.dataset import Dataset
from src.auxiliary.file_methods import load_arff
from src.auxiliary.preprocessing_methods import min_max_normalize
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class HypothyroidDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        super(HypothyroidDataset, self).__init__(config, verbose)
        self.data, _ = load_arff('hypothyroid')
        self.data = pd.DataFrame(self.data)
        self.balance = config['balance'] # Not implemented yet
        self.only_numerical = config['only_numerical']
        self.class_feature = 'Class'
        self.numerical_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
        self.nominal_features = [name for name in self.data.columns if name not in self.numerical_features + [self.class_feature]]
        self.verbose = verbose

    def get_raw_data(self) -> (np.ndarray, np.ndarray):
        if self.only_numerical:
            values = self.data[self.numerical_features].to_numpy()
        else:
            values = self.data.loc[:, self.data.columns != self.class_feature].to_numpy()
        labels = self.data[self.class_feature].to_numpy()
        return values, labels

    def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
        data = self.data

        if self.verbose:
            print('Started data preprocessing')

        # Delete features with more than half of samples with NaN values
        if self.verbose:
            nan_count = self.data.isnull().sum().sum()
            print('    ', 'Total number of NaNs: ', nan_count, '; relative: ',
            (nan_count * 100) / (len(data.index) * len(data.columns)), '%')

        columns_to_drop = []
        for feature_index in self.data.columns:
            nan_count = self.data[feature_index].isnull().sum()
            if nan_count > (len(self.data.index)/2):
                columns_to_drop.append(feature_index)
        self.data.drop(columns=columns_to_drop, inplace=True)
        self.numerical_features = [name for name in self.numerical_features if name not in columns_to_drop]
        self.nominal_features = [name for name in self.nominal_features if name not in columns_to_drop]
        if self.verbose:
            print('    ','Deleted because of too many NaN values the features with name:', columns_to_drop)

        # Numerical features -> replace the NaN values by the mean and normalize
        for feature_index in self.numerical_features:
            feature = self.data[feature_index]
            nan_indexes = self.data.index[feature.isnull()].tolist()
            feature = feature.to_numpy()
            feature_without_nans = np.delete(feature, nan_indexes)
            mean = np.mean(feature_without_nans)
            feature[nan_indexes] = mean
            normalized_feature = min_max_normalize(feature)
            self.data[feature_index] = normalized_feature

        # Nominal features -> replace the NaN values by the median
        if not self.only_numerical:
            for feature_index in self.numerical_features:
                feature = self.data[feature_index]
                nan_indexes = self.data.index[feature.isnull()].tolist()
                feature = feature.to_numpy()
                feature_without_nans = np.delete(feature, nan_indexes)
                median = np.median(feature_without_nans)
                feature[nan_indexes] = median
                self.data[feature_index] = feature

        if self.verbose:
            print('Finished data preprocessing')

        if self.only_numerical:
            values = self.data[self.numerical_features].to_numpy()
        else:
            values = self.data.loc[:, self.data.columns != self.class_feature].to_numpy()
        labels = data[self.class_feature].to_numpy()
        return values, labels
