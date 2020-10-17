from scipy.io.arff import loadarff
import pandas as pd
from preprocessing_kropt import transform_krops_col_to_numeric
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_arff(dataset_name: str):
    f = open(os.path.abspath(os.path.join('..', 'datasets', 'all', dataset_name + '.arff')))
    data, metadata = loadarff(f)
    return data, metadata


def get_processed_kropt(balance=False):
    data, metadata = load_arff('kropt')
    data = pd.DataFrame(data)
    columns = data.columns

    ss = MinMaxScaler()
    le = LabelEncoder()

    # Transform all columns to numeric values
    for col in columns:
        if col != 'game':
            data[col] = transform_krops_col_to_numeric(data[col], col)
    game_col = list(data['game'])
    data = data.drop(columns=['game'])
    data = ss.fit(data).transform(data)
    data = pd.DataFrame(data, columns=columns[:-1])
    data.insert(6, 'game', game_col)

    if balance:
        a = data[data['game'] == b'zero']
        data = data.append([data[data['game'] == b'zero']] * 75, ignore_index=True)
        data = data.append([data[data['game'] == b'one']] * 30, ignore_index=True)
        data = data.append([data[data['game'] == b'two']] * 10, ignore_index=True)
        data = data.append([data[data['game'] == b'three']] * 30, ignore_index=True)
        data = data.append([data[data['game'] == b'four']] * 12, ignore_index=True)
        data = data.append([data[data['game'] == b'five']] * 6, ignore_index=True)
        data = data.append([data[data['game'] == b'six']] * 6, ignore_index=True)
        data = data.append([data[data['game'] == b'seven']] * 5, ignore_index=True)
        data = data.append([data[data['game'] == b'eight']] * 2, ignore_index=True)
        data = data.append([data[data['game'] == b'nine']], ignore_index=True)
        data = data.append([data[data['game'] == b'ten']], ignore_index=True)
        data = data.append([data[data['game'] == b'fifteen']], ignore_index=True)
        data = data.append([data[data['game'] == b'sixteen']] * 8, ignore_index=True)
        print(data['game'].value_counts())

    le.fit(data['game'])
    data['game'] = le.transform(data['game'])

    return data


if __name__ == '__main__':
    data, metadata = load_arff('cmc')
    data = pd.DataFrame(data)
    pass
