from src.algorithms.unsupervised_algorithm import UnsupervisedAlgorithm
from src.algorithms.types.kmeans import KMeansAlgorithm
import numpy as np
from scipy.spatial import distance


class KMediansAlgorithm(UnsupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        self.n_clusters = config['n_clusters']
        self.max_iter = config['max_iter']
        self.init_centroids = config['init_centroids'] if 'init_centroids' in config else None
        self.verbose = verbose
        self.kmeans = KMeansAlgorithm({'n_clusters': config['n_clusters'],
                                       'max_iter': config['max_iter'],
                                       'init_centroids': config['init_centroids'] if 'init_centroids' in config
                                       else None,
                                       'maximization_function': 'median'},
                                      output_path, verbose)

    def run(self, values: np.ndarray) -> np.ndarray: # Unsupervised learning
        return self.kmeans.run(values)

    def save(self):
        raise NotImplementedError('Method not implemented in interface class')

    # Auxiliary methods

    def get_nearest_centroid(self, value, centroids: np.ndarray):
        return np.argmin(distance.cdist(np.array([value]), centroids, 'euclidean'))
