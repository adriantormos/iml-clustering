from src.algorithms.algorithm import Algorithm
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import time

class DBSCANAlgorithm(Algorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        self.eps = config['eps']
        self.metric = config['metric']
        self.algorithm = config['algorithm']
        self.verbose = verbose

    def train(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray: # Revise this code
        #data = np.concatenate((values, labels), axis=0)
        return DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit_predict(values)

    def evaluate(self, values: np.ndarray) -> np.ndarray: # Revise this code
        return DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit_predict(values)

    def save(self):
        raise NotImplementedError('Method not implemented in interface class')
