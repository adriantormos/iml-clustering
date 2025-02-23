from src.algorithms.unsupervised_algorithm import UnsupervisedAlgorithm
from src.algorithms.types.kmeans import KMeansAlgorithm
import numpy as np
import time


class BisectingKMeansAlgorithm(UnsupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        self.n_clusters = config['n_clusters']
        self.max_iter = config['max_iter']
        self.verbose = verbose
        self.kmeans = KMeansAlgorithm({'n_clusters': 2, 'max_iter': self.max_iter}, output_path, verbose)

    def run(self, values: np.ndarray):
        if self.verbose:
            print('Starting bisecting k-means.', 'Maximum {} iterations'.format(self.max_iter))
            start_time = time.time()

        # Base case
        if self.n_clusters == 1:
            print('Finished bisecting k-means in {0:.3f} seconds.'.format(time.time() - start_time))
            return np.zeros(len(values))

        # Initial separation
        labels: np.ndarray = self.kmeans.run(values)
        found_clusters = 2

        if self.verbose:
            print('Found 2 clusters of {}.'.format(self.n_clusters))

        while found_clusters < self.n_clusters:
            # Compute sum of square distances
            ssd_distances: np.ndarray = np.array([self.ssd(values[labels == i]) for i in range(found_clusters)])

            # Get cluster with largest intra-cluster distance
            worst_cluster = np.argmax(ssd_distances)
            rows_from_worst_cluster = values[labels == worst_cluster]

            # Perform k-means with worst cluster
            labels_to_merge = self.kmeans.run(rows_from_worst_cluster)
            labels = self.merge_labels_aux(labels, labels_to_merge, worst_cluster, [worst_cluster, found_clusters])

            found_clusters += 1

            if self.verbose:
                print('Found {} clusters of {}'.format(found_clusters, self.n_clusters))

        if self.verbose:
            print('Finished bisecting k-means in {0:.3f} seconds.'.format(time.time() - start_time))

        return labels

    def save(self):
        raise NotImplementedError('Method not implemented in interface class')

    # Auxiliary methods

    def ssd(self, values: np.ndarray):
        return np.sum((values - np.average(values, axis=0)) ** 2)

    def merge_labels_aux(self, og: np.ndarray, new: np.ndarray, num_to_replace: int, nums_to_insert: list):
        iter_new = 0
        # Iterate over og. When finding a value equal to num_to_replace, replace by nums_to_insert
        # according to the first non-visited element of new
        for i in range(len(og)):
            if og[i] == num_to_replace:
                # If new[iter_new] == 0 insert nums_to_insert[0]
                # If new[iter_new] == 1 insert nums_to_insert[1]
                # etc.
                og[i] = nums_to_insert[new[iter_new]]
                iter_new += 1
        return og
