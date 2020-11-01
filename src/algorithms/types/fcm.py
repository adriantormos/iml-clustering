from src.algorithms.unsupervised_algorithm import UnsupervisedAlgorithm
from src.auxiliary.file_methods import print_pretty_json
import numpy as np
import time


class FCMAlgorithm(UnsupervisedAlgorithm):

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

    def run(self, values: np.ndarray) -> np.ndarray: # Unsupervised learning
        if self.verbose:
            print('Starting FCM. Config:')
            print_pretty_json(self.config)
            start_time = time.time()

        # Initialize membership matrix U
        membership_matrix = self.compute_initial_matrix(values.shape[0])

        index = 0
        goal = None
        has_converged = False
        while not has_converged and index < self.max_iter:
            time1 = time.time()
            centroids = self.compute_centroids(values, membership_matrix)
            time2 = time.time()
            self.update_matrix(centroids, values, membership_matrix)
            time3 = time.time()
            score = self.compute_objective(values, centroids, membership_matrix)
            time4 = time.time()
            if goal and (abs(goal - score) < self.finish_threshold):
                print(goal, score)
                has_converged = True
            goal = score

            if self.verbose:
                print('Centroids:', time2-time1, 'Matrix:', time3-time2, 'Objective:', time4-time3)
                if index % 50 == 0:
                    print('Iteration {} of {}'.format(index + 1, self.max_iter))
            index += 1

        if self.verbose:
            print('Finished FCM',
                  '{} iterations performed in'.format(index),
                  '{0:.3f} seconds.'.format(time.time() - start_time),
                  'Algorithm converged.' if has_converged else 'Algorithm did not converge.')

        return self.compute_defuzzification(membership_matrix)

    def save(self):
        raise NotImplementedError('Method not implemented')

    # Auxiliary methods

    def random_compute_initial_matrix(self, number_values):
        matrix = np.zeros((number_values, self.n_clusters))
        for index in range(number_values):
            # random initialization for each value, assuring that all arrays sum 1
            matrix[index] = np.random.dirichlet(np.ones(self.n_clusters), size=1)[0]
        return matrix

    def compute_centroids(self, values, membership_matrix):
        # U^m
        exponent_membership_matrix = membership_matrix ** self.fuzziness

        # Every sum((U_ik)^m * x_k)
        weighted_sums = np.array([
            np.sum(np.array([exponent_membership_matrix[k, i] * values[k] for k in range(len(values))]))
            for i in range(len(membership_matrix[0]))
        ])
        # Every sum((U_ik)^m)
        membership_sums = np.sum(exponent_membership_matrix[i] for i in range(len(values)))

        # Every sum((U_ik)^m * x_k) / sum((U_ik)^m)
        centroids = np.array([np.divide(weighted_sums[i], membership_sums[i]) for i in range(len(membership_sums))])

        return centroids

    def update_matrix(self, centroids, values, belonging_matrix):

        M = 2/(self.fuzziness - 1)

        distances_matrix = np.array([self.compute_distance_between_points(i, k) for k in values for i in centroids])
        distances_matrix_numerators = np.array([np.repeat([x], self.n_clusters) for x in distances_matrix])
        distances_matrix_numerators = np.reshape(distances_matrix_numerators,
                                                 (len(values) * self.n_clusters, self.n_clusters))
        distances_matrix_denominators = np.reshape(distances_matrix, (len(values), self.n_clusters))
        # print(np.shape(distances_matrix_denominators))
        # print(distances_matrix_denominators)
        # print(np.shape(distances_matrix_denominators))
        distances_matrix_denominators = np.array([list(x) * self.n_clusters for x in distances_matrix_denominators])
        # print(distances_matrix_denominators)
        distances_matrix_denominators = np.reshape(distances_matrix_denominators,
                                                   (len(values) * self.n_clusters, self.n_clusters))
        # print(np.shape(distances_matrix_numerators))
        # print(distances_matrix_numerators)
        # print(np.shape(distances_matrix_denominators))
        # print(distances_matrix_denominators)
        belonging_matrix = np.divide(distances_matrix_numerators, distances_matrix_denominators)
        belonging_matrix = belonging_matrix ** M
        # print(np.shape(belonging_matrix))
        # print(belonging_matrix)
        belonging_matrix = np.sum(belonging_matrix, axis=1)
        # print(np.shape(belonging_matrix))
        # print(belonging_matrix)

        belonging_matrix = belonging_matrix ** (-1)
        belonging_matrix = np.reshape(belonging_matrix, (len(values), self.n_clusters))

        # print(belonging_matrix)
        # distances = np.reshape(distances, (len(values), self.n_clusters))
        # belonging_matrix = belonging_matrix

        # print(np.shape(distances_matrix_numerators))
        # print(np.shape(distances_matrix_denominators))
        # for index_matrix in range(values.shape[0]):
        #     for index_centroid in range(self.n_clusters):
        #         distances_matrix[index_matrix, index_centroid] = self.compute_distance_between_points(values[index_matrix], centroids[index_centroid])

        # for index_matrix in range(values.shape[0]):
        #     for index_centroid in range(self.n_clusters):
        #         sum = 0
        #         for index_centroid_2 in range(self.n_clusters):
        #             sum += (distances_matrix[index_matrix, index_centroid] / distances_matrix[index_matrix, index_centroid_2])**(2/(self.fuzziness - 1))
        #         belonging_matrix[index_matrix,index_centroid] = sum**(-1)

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

