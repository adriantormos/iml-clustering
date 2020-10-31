from sklearn.metrics import davies_bouldin_score
import time
from src.validation.validation_method import ValidationMethod
from src.factory.types.algorithm_factory import AlgorithmFactory


class DaviesBouldinScore(ValidationMethod):

    def __init__(self, config, output_path, verbose):
        self.clusters = config['clusters']
        self.n_initializations = config['n_initializations']
        self.algorithm = config['algorithm']
        self.verbose = verbose

    def evaluate(self, values):
        scores_per_k = []
        labels_per_k = []

        if self.verbose:
            print('Starting Davies-Bouldin score validation.',
                  '{} initializations per k in {}'.format(self.n_initializations, self.clusters))
            start_time = time.time()

        # Iterate over all k in self.clusters
        for k in self.clusters:

            if self.verbose:
                print('Validating {} clusters'.format(k))

            algorithm_config = {
                'name': self.algorithm['name'],
                'n_clusters': k,
                'max_iter': self.algorithm['max_iter']
            }
            algorithm = AlgorithmFactory.select_algorithm(algorithm_config, '', self.verbose)

            # Initial placeholder values for a certain k
            score_this_k = float('inf')
            labels_this_k = None
            # Perform the algorithm n_initializations times
            for i in range(self.n_initializations):

                if self.verbose:
                    print('Initialization {} of {}'.format(i+1, self.n_initializations))

                # Perform the clustering algorithm
                labels = algorithm.train(values)
                score = davies_bouldin_score(values, labels)

                if self.verbose:
                    print('Score: {}'.format(score))

                # If the score is better, store it with labels
                if score < score_this_k:
                    score_this_k = score
                    labels_this_k = labels

            # Append to the result the best score and labels
            scores_per_k.append((k, score_this_k))
            labels_per_k.append(labels_this_k)

        if self.verbose:
            print('Finished Davies-Bouldin score validation.',
                  '{} scores computed.'.format(len(self.clusters) * self.n_initializations),
                  '{0:.3f} seconds'.format(time.time() - start_time))

        return scores_per_k, labels_per_k
