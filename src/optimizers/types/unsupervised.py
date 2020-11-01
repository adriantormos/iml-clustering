import time
from prettytable import PrettyTable
from src.auxiliary.file_methods import print_pretty_json, save_csv
from src.auxiliary.visualize_methods import compare_multiple_lines
from src.optimizers.optimizer import Optimizer
from src.factory.algorithm import AlgorithmFactory
from src.auxiliary.evaluation_methods import run_evaluation_metric
from src.algorithms.unsupervised_algorithm import UnsupervisedAlgorithm


class UnsupervisedOptimizer(Optimizer):

    # Main methods

    def __init__(self, optimizer_config, algorithm_config, output_path, verbose):
        self.parameters = optimizer_config['parameters']
        self.use_best_parameters = optimizer_config['use_best_parameters']
        self.n_runs = optimizer_config['n_runs']
        self.metrics = optimizer_config['metrics']
        self.verbose = verbose
        self.output_path = output_path
        self.algorithm_config = algorithm_config
        self.optimizer_config = optimizer_config

    def run(self, values, labels):
        if self.verbose:
            print('Starting unsupervised optimizer.', 'Algorithm: ' + self.algorithm_config['name'] + '.', 'Optimizer config:')
            print_pretty_json(self.optimizer_config)
            start_time = time.time()

        if len(self.parameters) > 0:
            best_global_score = None
            best_global_labels = None

            for parameter in self.parameters:
                parameter_name = parameter['name']
                parameter_values = parameter['values']
                algorithm_config = self.algorithm_config.copy()

                scores = []
                best_score = None
                best_value = None
                best_labels = None
                for value in parameter_values:
                    algorithm_config[parameter_name] = value
                    score, output_labels = self.run_algorithm(algorithm_config, values, labels=None)
                    scores.append(score)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_labels = output_labels
                        best_value = value
                self.do_optimization_visualization(parameter_name, parameter_values, scores)

                if self.verbose:
                    print('Best value for param', parameter_name, ':', str(best_value))

                if best_global_score is None or best_score < best_global_score:
                    best_global_score = best_score
                    best_global_labels = best_labels

            if self.use_best_parameters:
                _, output_labels = best_global_labels # TODO merge all the best parameters
            else:
                _, output_labels = self.run_algorithm(self.algorithm_config, values, labels)
        else:
            _, output_labels = self.run_algorithm(self.algorithm_config, values, labels)

        return output_labels

    # Auxiliary methods

    def run_algorithm(self, algorithm_config, values, labels):
        best_score = None
        best_labels = None
        for _ in range(self.n_runs):
            algorithm: UnsupervisedAlgorithm = AlgorithmFactory.select_algorithm(algorithm_config, self.output_path, True) # TODO resolve this disorder
            if labels is None:
                output_labels = algorithm.run(values)
            else:
                output_labels = algorithm.train(values, labels)
            score = self.evaluate_algorithm(values, output_labels)
            if best_score is None or score < best_score:
                best_score = score
                best_labels = output_labels
        return best_score, best_labels

    def evaluate_algorithm(self, values, output_labels):
        sum = 0
        for metric in self.metrics:
            sum += run_evaluation_metric(metric, values, None, output_labels)
        return sum / len(self.metrics)

    def do_optimization_visualization(self, parameter_name, parameter_values, scores):
        self.do_table_results(parameter_name, parameter_values, scores)
        self.do_plot_results(parameter_name, parameter_values, scores)
        self.do_cluster_plots(parameter_name, parameter_values, scores)

    def do_table_results(self, parameter_name, parameter_values, scores):
        rows = [['parameter_value_' + parameter_name, 'score']]
        x = PrettyTable()
        x.field_names = ['parameter_value_' + parameter_name, 'score']
        for index, parameter_value in enumerate(parameter_values):
            rows.append([parameter_value, scores[index]])
            x.add_row([parameter_value, scores[index]])
        if self.output_path is not None:
            save_csv(self.output_path + '/table_scores_parameter_' + parameter_name, rows)
        print(x)

    def do_plot_results(self, parameter_name, parameter_values, scores):
        x_array = parameter_values
        lines = [(scores, None)]
        compare_multiple_lines(x_array, lines, 'Scores for different values for the parameter ' + parameter_name, self.output_path + '/plot_scores_parameter_' + parameter_name, legend=False, xlabel='Different ' + parameter_name + ' values', ylabel='Score', ylim=None)

    def do_cluster_plots(self, parameter_name, parameter_values, scores):
        pass
