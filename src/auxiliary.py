from scipy.io.arff import loadarff
import pandas as pd

import os


def load_arff(dataset_name: str):
    f = open(os.path.abspath(os.path.join('..', 'datasets', 'all', dataset_name + '.arff')))
    data, metadata = loadarff(f)
    return data, metadata


if __name__ == '__main__':
    data, metadata = load_arff('cmc')
    data = pd.DataFrame(data)
    pass
