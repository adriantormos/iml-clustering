from src.algorithms.algorithm import Algorithm
from src.auxiliary.file_methods import print_pretty_json
import numpy as np
import time


class FCMAlgorithm(Algorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        self.config = config
        self.n_clusters = config['n_clusters']
        self.max_iter = config['max_iter']
        self.finish_threshold = config['finish_threshold']
        self.fuzziness = config['fuzziness'] # m parameter in the paper
        distance_metric = config['distance_metric']
        initial_matrix_computation = config['initial_matrix_computation']
        defuzzification_method = config['defuzzification_method']
        objective_computation = config['objective_computation'] # A matrix in the paper
        self.verbose = verbose
        if distance_metric == 'euclidean':
            self.compute_distance_between_points = self.euclidean_distance_between_points
        else:
            raise Exception('Distance metric not implemented')
        if initial_matrix_computation == 'random':
            self.compute_initial_matrix = self.random_compute_initial_matrix
        else:
            raise Exception('Initial matrix computation not implemented')
        if defuzzification_method == 'max_value':
            self.compute_defuzzification = self.max_value_defuzzification
        else:
            raise Exception('Defuzzification method not implemented')
        if objective_computation == 'euclidean':
            self.compute_objective = self.euclidean_objective
        else:
            raise Exception('Objective computation not implemented')
        if self.fuzziness <= 1:
            raise Exception('Algorithm only implemented for fuzziness > 1')

    def train(self, values: np.ndarray, labels=None) -> np.ndarray: # Unsupervised learning
        if self.verbose:
            print('Starting FCM. Config:')
            print_pretty_json(self.config)
            start_time = time.time()

        belonging_matrix = self.compute_initial_matrix(values.shape[0])

        index = 0
        goal = None
        has_converged = False
        while not has_converged and index < self.max_iter:
            centroids = self.compute_centroids(values, belonging_matrix)
            self.update_matrix(centroids, values, belonging_matrix)
            score = self.compute_objective(values, centroids, belonging_matrix)
            if goal and (abs(goal - score) < self.finish_threshold):
                print(goal, score)
                has_converged = True
            goal = score

            if self.verbose:
                if index % 50 == 0:
                    print('Iteration {} of {}'.format(index + 1, self.max_iter))
            index += 1

        if self.verbose:
            print('Finished FCM',
                  '{} iterations performed in'.format(index),
                  '{0:.3f} seconds.'.format(time.time() - start_time),
                  'Algorithm converged.' if has_converged else 'Algorithm did not converge.')

        return self.compute_defuzzification(belonging_matrix)

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented')

    def save(self):
        raise NotImplementedError('Method not implemented')

    # Auxiliary methods

    def random_compute_initial_matrix(self, number_values):
        matrix = np.zeros((number_values, self.n_clusters))
        for index in range(number_values):
            # random initialization for each value, assuring that all arrays sum 1
            matrix[index] = np.random.dirichlet(np.ones(self.n_clusters), size=1)[0]
        return matrix

    def compute_centroids(self, values, belonging_matrix):
        centroids = np.zeros((self.n_clusters, values.shape[1]))
        for index_cluster in range(self.n_clusters):
            # the centroid of a cluster is the mean of all points, weighted by their degree of belonging to the cluster
            weighted_sum = 0
            weights_sum = 0
            for index_matrix in range(values.shape[0]):
                weighted_sum += belonging_matrix[index_matrix][index_cluster]**self.fuzziness * values[index_matrix]
                weights_sum += belonging_matrix[index_matrix][index_cluster]**self.fuzziness
            centroids[index_cluster] = weighted_sum / weights_sum
        return centroids

    def update_matrix(self, centroids, values, belonging_matrix):
        distances_matrix = np.zeros((values.shape[0], self.n_clusters))

        for index_matrix in range(values.shape[0]):
            for index_centroid in range(self.n_clusters):
                distances_matrix[index_matrix, index_centroid] = self.compute_distance_between_points(values[index_matrix], centroids[index_centroid])

        for index_matrix in range(values.shape[0]):
            for index_centroid in range(self.n_clusters):
                sum = 0
                for index_centroid_2 in range(self.n_clusters):
                    sum += (distances_matrix[index_matrix, index_centroid] / distances_matrix[index_matrix, index_centroid_2])**(2/(self.fuzziness - 1))
                belonging_matrix[index_matrix,index_centroid] = sum**(-1)

    def euclidean_objective(self, values, centroids, belonging_matrix):
        score = 0
        for index_matrix in range(values.shape[0]):
            for index_centroid in range(self.n_clusters):
                score += belonging_matrix[index_matrix,index_centroid]**self.fuzziness * self.compute_distance_between_points(values[index_matrix], centroids[index_centroid])**2
        return score

    def euclidean_distance_between_points(self, a, b):
        return np.linalg.norm(a-b)

    def max_value_defuzzification(self, belonging_matrix):
        labels = np.zeros(belonging_matrix.shape[0])
        for index_matrix in range(belonging_matrix.shape[0]):
            labels[index_matrix] = np.argmax(belonging_matrix[index_matrix])
        return labels

