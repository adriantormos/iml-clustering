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


def ssd(x: np.ndarray):
    return np.sum((x - np.average(x, axis=0))**2)


def merge_labels(og: np.ndarray, new: np.ndarray, num_to_replace: int, nums_to_insert: list):
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


def bisecting_kmeans(X: pd.DataFrame, n_clusters: int, max_iter: int, verbose: bool = False) -> np.ndarray:

    if verbose:
        print('Starting bisecting k-means.', 'Maximum {} iterations'.format(max_iter))
        start_time = time.time()

    x: np.ndarray = np.array(X)

    # Base case
    if n_clusters == 1:
        print('Finished bisecting k-means in {0:.3f} seconds.'.format(time.time() - start_time))
        return np.zeros(len(X))

    # Initial separation
    labels: np.ndarray = kmeans(X, n_clusters=2, max_iter=max_iter, verbose=verbose)
    found_clusters = 2

    if verbose:
        print('Found 2 clusters of {}.'.format(n_clusters))

    while found_clusters < n_clusters:
        # Compute sum of square distances
        ssd_distances: np.ndarray = np.array([ssd(x[labels == i]) for i in range(found_clusters)])

        # Get cluster with largest intra-cluster distance
        worst_cluster = np.argmax(ssd_distances)
        rows_from_worst_cluster = X[labels == worst_cluster]

        # Perform k-means with worst cluster
        labels_to_merge = kmeans(rows_from_worst_cluster, n_clusters=2, max_iter=max_iter, verbose=verbose)
        labels = merge_labels(labels, labels_to_merge, worst_cluster, [worst_cluster, found_clusters])

        found_clusters += 1

        if verbose:
            print('Found {} clusters of {}'.format(found_clusters, n_clusters))

    if verbose:
        print('Finished bisecting k-means in {0:.3f} seconds.'.format(time.time() - start_time))

    return labels
