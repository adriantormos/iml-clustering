from sklearn.metrics import davies_bouldin_score, adjusted_rand_score, f1_score


def run_evaluation_metric(name, values, labels, output_labels):
    return eval(name)(values, labels, output_labels)


def davies_bouldin(values, labels, output_labels):
    return davies_bouldin_score(values, output_labels)


def adjusted_rand(values, labels, output_labels):
    return adjusted_rand_score(labels, output_labels)


def f1(values, labels, output_labels):
    return f1_score(labels, output_labels)