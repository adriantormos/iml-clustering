from auxiliary import get_processed_kropt
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from src.algorithms import kmeans, bisecting_kmeans


if __name__ == '__main__':
    kropt = get_processed_kropt().drop(columns=['game'])
    kropt_b = get_processed_kropt(balance=True).drop(columns=['game'])

    # for d in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']:
    # for d in ['euclidean']:
    #     print(d)
    #     for a in ['auto', 'ball_tree', 'kd_tree', 'brute']:
    #         print(a)
    #         labels = DBSCAN(eps=0.2, metric=d, algorithm=a).fit_predict(kropt)
    #         labels_b = DBSCAN(eps=0.2, metric=d, algorithm=a).fit_predict(kropt_b)
    #         print(pd.Series(labels).value_counts())
    #         print(pd.Series(labels_b).value_counts())

    x = kmeans(kropt, n_clusters=16, max_iter=300, verbose=False)
    print(x[:20])
    unique, frequency = np.unique(x, return_counts=True)
    print('\n'.join(['{}: {}'.format(unique[i], frequency[i]) for i in range(len(unique))]))

    x = bisecting_kmeans(kropt, n_clusters=16, max_iter=100, verbose=False)
    print(x[:20])
    unique, frequency = np.unique(x, return_counts=True)
    print('\n'.join(['{}: {}'.format(unique[i], frequency[i]) for i in range(len(unique))]))
    pass