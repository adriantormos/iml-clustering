import argparse
import sys
from src.auxiliary.file_methods import load_json, save_json
from src.factory.dataset import DatasetFactory
from src.factory.algorithm import AlgorithmFactory
from src.factory.optimizer import OptimizerFactory
from src.visualize import show_charts
import random
import numpy as np


def parse_arguments():
    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', help="Path to config file", required=True)
    parser.add_argument('--output_path', default=None, help="Path to output directory", required=False)
    parser.add_argument('--visualize', default=True, type=parse_bool, help="Move standard output to a log file",
                        required=False)
    parser.add_argument('--verbose', default=False, type=parse_bool, help="Show more info", required=False)
    args = parser.parse_args()

    # TODO check the first path is a file and a json
    # TODO check the second path is a directory
    # TODO check config file correctness

    return args


def main(config_path: str, output_path: str, visualize: bool, verbose: bool):
    # Load configuration
    config = load_json(config_path)
    data_config = config['data']
    if 'algorithm' in config:
        algorithm_config = config['algorithm']
    if 'optimizer' in config:
        optimizer_config = config['optimizer']
    charts_config = config['charts']

    # Set up
    random.seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])

    # Load and prepare data
    dataset = DatasetFactory.select_dataset(data_config, verbose)
    values, labels = dataset.get_preprocessed_data()
    values, labels = dataset.prepare(values, labels)

    # Run algorithm if required
    if 'algorithm' in config:
        if 'optimizer' in config:
            optimizer = OptimizerFactory.select_optimizer(optimizer_config, algorithm_config, output_path, verbose)
            output_labels = optimizer.run(values, labels)
        else:
            algorithm = AlgorithmFactory.select_algorithm(algorithm_config, output_path, verbose)
            output_labels = algorithm.train(values, labels)
    else:
        output_labels = None

    # Visualize results
    show_charts(charts_config, output_path, values, labels, output_labels, visualize, dataset.get_preprocessed_dataframe(), verbose)

    # Save config json and algorithm model
    if output_path is not None:
        if output_labels is not None:
            np.save(output_path + '/predicted_labels', output_labels)
        save_json(output_path + '/config', config)
        #algorithm.save() -> Not implemented yet


if __name__ == '__main__':
    args = parse_arguments()
    # redirect program output
    if not args.visualize and args.output_path:
        f = open(args.output_path + '/log.txt', 'w')
        sys.stdout = f
    main(args.config_path, args.output_path, args.visualize, args.verbose)
    if not args.visualize and args.output_path:
        f.close()
