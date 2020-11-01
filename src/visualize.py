import numpy as np
from prettytable import PrettyTable
from src.auxiliary.evaluation_methods import run_evaluation_metric
from src.auxiliary.file_methods import save_csv
import sklearn.metrics as metrics
import pandas as pd

# correlation among variables
# https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
# https://www.kaggle.com/minc33/visualizing-high-dimensional-clusters


def show_charts(config, output_path, values, labels, output_labels, visualize, verbose):
    for chart in config:
        eval(chart['name'])(chart, output_path, values, labels, output_labels, visualize, verbose)


def class_frequencies(config, output_path, values, labels, output_labels, visualize, verbose):
    unique1, counts1 = np.unique(labels, return_counts=True)
    #counts1, unique1 = zip(*sorted(zip(counts1, unique1), reverse=True)) -> show sorted
    unique2, counts2 = np.unique(output_labels, return_counts=True)
    print(unique2, counts2)

    rows = [['class', 'original_samples_distribution', 'predicted_samples_distribution']]
    x = PrettyTable()
    x.field_names = ['class', 'original_samples_distribution', 'predicted_samples_distribution']
    for index, _class in enumerate(unique1):
        number_samples1 = counts1[index]
        number_samples2 = counts2[np.where(unique2 == _class)[0]][0]
        rows.append([_class, number_samples1, number_samples2])
        x.add_row([_class, number_samples1, number_samples2])
    if output_path is not None:
        save_csv(output_path + '/samples_distribution', rows)
    print(x)


def show_metrics_table(config, output_path, values, labels, output_labels, visualize, verbose):
    rows = [[''] + config['metrics']]
    x = PrettyTable()
    x.field_names = [''] + config['metrics']
    scores = []
    for metric in config['metrics']:
        scores.append(run_evaluation_metric(metric, values, labels, output_labels))
    rows.append(['result'] + scores)
    x.add_row(['result'] + scores)
    if output_path is not None:
        save_csv(output_path + '/table_scores', rows)
    print(x)


def show_classification_report(config, output_path, values, labels, output_labels, visualize, verbose):
    print(metrics.classification_report(labels, output_labels))
    classification_report = metrics.classification_report(labels, output_labels, output_dict=True)
    classification_report = pd.DataFrame(classification_report).transpose()
    classification_report.to_csv(output_path + '/classification_report.csv', index=False)


def show_confusion_matrix(config, output_path, values, labels, output_labels, visualize, verbose):
    print('Confusion matrix')
    print(metrics.confusion_matrix(labels, output_labels))
    confusion_matrix = metrics.confusion_matrix(labels, output_labels)
    confusion_matrix = pd.DataFrame(confusion_matrix).transpose()
    confusion_matrix.to_csv(output_path + '/confusion_matrix.csv', index=False)