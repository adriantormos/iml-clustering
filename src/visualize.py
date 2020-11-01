import numpy as np
from prettytable import PrettyTable
from src.auxiliary.evaluation_methods import run_evaluation_metric
from src.auxiliary.file_methods import save_csv
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
#from plotly.offline import plot
#import plotly.graph_objs as go
#from sklearn.decomposition import PCA


def show_charts(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    for chart in config:
        eval(chart['name'])(chart, output_path, values, labels, output_labels, visualize, dataframe, verbose)


def class_frequencies(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    unique1, counts1 = np.unique(labels, return_counts=True)
    #counts1, unique1 = zip(*sorted(zip(counts1, unique1), reverse=True)) -> show sorted
    unique2, counts2 = np.unique(output_labels, return_counts=True)

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

def class_frequencies_separate(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    unique, counts = np.unique(output_labels, return_counts=True)
    rows = [['class', 'predicted_samples_distribution']]
    x = PrettyTable()
    x.field_names = ['class', 'predicted_samples_distribution']
    for index, _class in enumerate(unique):
        number_samples = counts[index]
        rows.append([_class, number_samples])
        x.add_row([_class, number_samples])
    if output_path is not None:
        save_csv(output_path + '/samples_distribution', rows)
    print(x)


def show_metrics_table(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
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


def show_classification_report(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    print(metrics.classification_report(labels, output_labels))
    classification_report = metrics.classification_report(labels, output_labels, output_dict=True)
    classification_report = pd.DataFrame(classification_report).transpose()
    classification_report.to_csv(output_path + '/classification_report.csv', index=False)


def show_confusion_matrix(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    print('Confusion matrix')
    print(metrics.confusion_matrix(labels, output_labels))
    confusion_matrix = metrics.confusion_matrix(labels, output_labels)
    confusion_matrix = pd.DataFrame(confusion_matrix).transpose()
    confusion_matrix.to_csv(output_path + '/confusion_matrix.csv', index=False)


def show_feature_histograms(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    dataframe.hist(bins=config['bins'], color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
    if output_path is not None:
        plt.savefig(output_path + '/feature_histograms', bbox_inches='tight')
    plt.show()


def show_correlation_among_variables(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    f, ax = plt.subplots(figsize=(config['figsize'][0], config['figsize'][1]))
    corr = dataframe.corr()
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',linewidths=.05)
    f.subplots_adjust(top=0.93)
    t = f.suptitle(config['title'], fontsize=14)
    if output_path is not None:
        plt.savefig(output_path + '/correlation_heatmap', bbox_inches='tight')
    plt.show()


def show_parallel_coordinates(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    f, ax = plt.subplots(figsize=(config['figsize'][0], config['figsize'][1]))
    plt.title(config['title'] + ' original labels')
    pc = parallel_coordinates(dataframe, dataframe.columns[-1], color=('#FFE888', '#FF9999'))
    if output_path is not None:
        plt.savefig(output_path + '/parallel_coordinates_labels', bbox_inches='tight')
    plt.show()
    aux = dataframe.copy()
    aux[dataframe.columns[-1]] = output_labels
    f, ax = plt.subplots(figsize=(config['figsize'][0], config['figsize'][1]))
    plt.title(config['title'] + ' predicted labels')
    pc = parallel_coordinates(aux, dataframe.columns[-1])
    if output_path is not None:
        plt.savefig(output_path + '/parallel_coordinates_predicted_labels', bbox_inches='tight')
    plt.show()


def show_pair_wise_scatter_plot(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
    pp = sns.pairplot(dataframe, hue=dataframe.columns[-1], height=1.8, aspect=1.8, plot_kws=dict(edgecolor="black", linewidth=0.5))
    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    t = fig.suptitle(config['title']  + ' for the original labels', fontsize=14)
    if output_path is not None:
        plt.savefig(output_path + '/pair_wise_scatter_plot_labels', bbox_inches='tight')
    plt.show()

    aux = dataframe.copy()
    aux[dataframe.columns[-1]] = output_labels
    pp = sns.pairplot(aux, hue=aux.columns[-1], height=1.8, aspect=1.8,
                      plot_kws=dict(edgecolor="black", linewidth=0.5))
    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    t = fig.suptitle(config['title'] + ' for the predicted labels', fontsize=14)
    if output_path is not None:
        plt.savefig(output_path + '/pair_wise_scatter_plot_predicted_labels', bbox_inches='tight')
    plt.show()


#def show_clusters_pca_2d(config, output_path, values, labels, output_labels, visualize, dataframe, verbose):
#    # trace1 is for 'Cluster 0'
#    plotX = dataframe.copy()
#    pca_2d = PCA(n_components=2)
#    PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Class"], axis=1)))
#    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
#    plotX = pd.concat([plotX, PCs_2d], axis=1, join='inner')
#    cluster0 = plotX[plotX["Class"] == 0]
#    cluster1 = plotX[plotX["Class"] == 1]
#
#    trace1 = go.Scatter(
#        x=cluster0["PC1_2d"],
#        y=cluster0["PC2_2d"],
#        mode="markers",
#        name="Cluster 0",
#        marker=dict(color='rgba(65, 0, 0, 0.8)'),
#        text=None)
#    trace2 = go.Scatter(
#        x=cluster1["PC1_2d"],
#        y=cluster1["PC2_2d"],
#        mode="markers",
#        name="Cluster 1",
#        marker=dict(color='rgba(65, 65, 0, 0.8)'),
#        text=None)
#
#    title = "Visualizing clusters in two dimensions using PCA of the kropt dataset"
#
#    fig = go.Figure(
#        layout=go.Layout(
#            title=title,
#            xaxis=dict(title='PC1', ticklen=5, zeroline=False),
#            yaxis=dict(title='PC2', ticklen=5, zeroline=False)
#        )
#    )
#    #fig.add_layout_image(layout)
#    fig.add_trace(trace1)
#    fig.add_trace(trace2)
#    if output_path is not None:
#        fig.write_image(output_path + '/plot_clusters_2d_pca_labels.png')
#    plot(fig)
