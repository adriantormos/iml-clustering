import pandas as pd
import numpy as np
from random import sample
from scipy.spatial import distance
from typing import List


def get_nearest_centroid(x, centroids: pd.DataFrame):
    x_array = np.repeat([x], [len(centroids)], axis=0)
    return np.argmin(distance.cdist(x_array, centroids, 'euclidean'))


def kmeans(X: pd.DataFrame, n_clusters: int, max_iter: int, verbose: bool = False) -> np.ndarray:
    if verbose:
        print('Starting k-means')

    x: np.ndarray = np.array(X)

    # Initialize centroids
    centroids: np.ndarray = np.array(X.sample(n=n_clusters))

    has_converged = False

    it_counter = 0
    for i in range(max_iter):
        # Compute nearest centroid of each sample
        labels: np.ndarray = np.array([get_nearest_centroid(row, centroids) for row in x])

        # Recompute centroids
        new_centroids: np.ndarray = np.array([np.average(x[labels == i], axis=0) for i in range(n_clusters)])

        if verbose and i % 50 == 0:
            print('Iteration {} of {}'.format(it_counter + 1, max_iter))
        if verbose:
            it_counter = i

        # Convergence condition
        if np.all(np.equal(new_centroids, centroids)):
            has_converged = True
            break
        centroids = new_centroids

    if verbose:
        print('Finished k-means.', '{} iterations performed.'.format(it_counter + 1),
              'Algorithm converged.' if has_converged else 'Algorithm did not converge.')

    return labels
