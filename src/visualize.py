import numpy as np


def show_charts(config, output_path, labels, visualize, verbose):
    for chart in config:
        eval(chart['name'])(chart, output_path, labels, visualize, verbose)

def class_frequencies(config, output_path, labels, visualize, verbose):
    unique, frequency = np.unique(labels, return_counts=True)
    print('\n'.join(['{}: {}'.format(unique[i], frequency[i]) for i in range(len(unique))]))
    # Save in output path