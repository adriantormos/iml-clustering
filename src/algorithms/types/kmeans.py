from src.algorithms.unsupervised_algorithm import UnsupervisedAlgorithm
import numpy as np
from scipy.spatial import distance
import time


class KMeansAlgorithm(UnsupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        self.n_clusters = config['n_clusters']
        self.max_iter = config['max_iter']
        self.init_centroids = config['init_centroids'] if 'init_centroids' in config else 'random'
        self.verbose = verbose
        self.maximization_function = 'average'
        if 'maximization_function' in config:
            self.maximization_function = config['maximization_function']

    def run(self, values: np.ndarray) -> np.ndarray: # Unsupervised learning
        has_converged = False

        if self.verbose:
            print('Starting k-means.' if self.maximization_function == 'average' else 'Starting k-medians.',
                  'Maximum {} iterations'.format(self.max_iter))
            start_time = time.time()

        # Initialize centroids
        if self.init_centroids == 'random':
            centroids: np.ndarray = np.array([values[i] for i in np.random.choice(len(values),
                                                                                  size=self.n_clusters, replace=False)])
        elif self.init_centroids is 'kmeans++':
            pass
        else:
            centroids: np.ndarray = np.array(self.init_centroids)

        it_counter = 0
        for i in range(self.max_iter):
            # Compute nearest centroid of each sample
            labels: np.ndarray = np.array([self.get_nearest_centroid(sample, centroids) for sample in values])

            new_centroids = []
            # Recompute centroids
            if self.maximization_function == 'average':
                new_centroids: np.ndarray = np.array([np.average(values[labels == cluster_id], axis=0)
                                                      for cluster_id in range(self.n_clusters)])
            elif self.maximization_function == 'median':
                new_centroids: np.ndarray = np.array([np.median(values[labels == cluster_id], axis=0)
                                                      for cluster_id in range(self.n_clusters)])

            if self.verbose:
                it_counter = i
                if i % 50 == 0:
                    print('Iteration {} of {}'.format(it_counter + 1, self.max_iter))

            # Convergence condition
            if np.all(np.equal(new_centroids, centroids)):
                has_converged = True
                break
            centroids = new_centroids

        if self.verbose:
            print('Finished k-means.' if self.maximization_function == 'average' else 'Finished k-medians.',
                  '{} iterations performed in'.format(it_counter + 1),
                  '{0:.3f} seconds.'.format(time.time() - start_time),
                  'Algorithm converged.' if has_converged else 'Algorithm did not converge.')

        return labels

    def save(self):
        raise NotImplementedError('Method not implemented in interface class')

    # Auxiliary methods

    def get_nearest_centroid(self, value, centroids: np.ndarray):
        return np.argmin(distance.cdist(np.array([value]), centroids, 'euclidean'))
