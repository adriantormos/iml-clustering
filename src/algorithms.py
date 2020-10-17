import pandas as pd
import numpy as np
from random import sample
from scipy.spatial import distance
import time
from typing import List


def get_nearest_centroid(x, centroids: pd.DataFrame):
    x_array = np.repeat([x], [len(centroids)], axis=0)
    return np.argmin(distance.cdist(x_array, centroids, 'euclidean'))


def kmeans(X: pd.DataFrame, n_clusters: int, max_iter: int, init_centroids: list = None,
           verbose: bool = False) -> np.ndarray:
    has_converged = False
    if verbose:
        print('Starting k-means.', 'Maximum {} iterations'.format(max_iter))
        start_time = time.time()

    x: np.ndarray = np.array(X)

    # Initialize centroids
    if init_centroids is None:
        centroids: np.ndarray = np.array(X.sample(n=n_clusters))
    else:
        centroids: np.ndarray = np.array(init_centroids)

    it_counter = 0
    for i in range(max_iter):
        # Compute nearest centroid of each sample
        labels: np.ndarray = np.array([get_nearest_centroid(row, centroids) for row in x])

        # Recompute centroids
        new_centroids: np.ndarray = np.array([np.average(x[labels == i], axis=0) for i in range(n_clusters)])

        if verbose:
            it_counter = i
            if i % 50 == 0:
                print('Iteration {} of {}'.format(it_counter + 1, max_iter))

        # Convergence condition
        if np.all(np.equal(new_centroids, centroids)):
            has_converged = True
            break
        centroids = new_centroids

    if verbose:
        print('Finished k-means.', '{} iterations performed in'.format(it_counter + 1),
              '{0:.3f} seconds.'.format(time.time() - start_time),
              'Algorithm converged.' if has_converged else 'Algorithm did not converge.')

    return labels
