import numpy as np

# correlation among variables
# https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
# https://www.kaggle.com/minc33/visualizing-high-dimensional-clusters

def show_charts(config, output_path, labels, visualize, verbose):
    for chart in config:
        eval(chart['name'])(chart, output_path, labels, visualize, verbose)

def class_frequencies(config, output_path, labels, visualize, verbose):
    unique, frequency = np.unique(labels, return_counts=True)
    print('\n'.join(['{}: {}'.format(unique[i], frequency[i]) for i in range(len(unique))]))
    # Save in output path