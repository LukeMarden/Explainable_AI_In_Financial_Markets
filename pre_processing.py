from os import listdir
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN
from collect_data import *


class pre_processing:

    def __init__(self, tickers):
        self.continuous_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.tickers = tickers
        self.tables = {}
        for ticker in self.tickers:
            self.tables[ticker] = pd.read_csv('data/ftse100/' + ticker + '.csv')

    def missing_values(self):
        for ticker in self.tickers:
            if self.tables[ticker].isnull().any(axis=1).sum():
                print(self.tables[ticker][self.tables[ticker].isna().any(axis=1)])

    def outlier_detection(self, show_box_plots=False, show_LocalOutlierFactor=True, showDBSCAN=False, show_knn=False, show_IsolationForest=False):
        for ticker in self.tickers:
            self.tables[ticker].dropna(inplace=True)
            if show_box_plots is True:
                #box plots
                boxplot = self.tables[ticker].boxplot(column=self.continuous_features[0:3])
                plt.title(ticker)
                plt.show()
                boxplot = self.tables[ticker].boxplot(column=self.continuous_features[3:6])
                plt.title(ticker)
                plt.show()
            if show_LocalOutlierFactor is True:
                #code as per:
                # https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html
                np.random.seed(42)
                clf = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
                y_pred = clf.fit_predict(self.tables[ticker][self.continuous_features])

                outlier_count = (y_pred==-1).sum()
                print("Number of predicted outliers:", outlier_count)

                X_scores = clf.negative_outlier_factor_

                plt.figure(figsize=(12, 8))
                plt.title("Local Outlier Factor (LOF) of " + ticker)
                Xol = self.tables[ticker][self.continuous_features].to_numpy()
                plt.scatter(Xol[:, 0], Xol[:, 1], color='k', s=3, label='Data points')
                radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
                plt.scatter(Xol[:, 0], Xol[:, 1], s=500 * radius, edgecolors='r',
                            facecolors='none', label='Outlier scores')
                plt.axis('tight')
                # plt.xlim((-5, 5))
                # plt.ylim((-5, 5))
                plt.xlabel("Number Of Outliers: %d" % (outlier_count))
                legend = plt.legend(loc='upper left')
                legend.legendHandles[0]._sizes = [10]
                legend.legendHandles[1]._sizes = [20]
                plt.show()
            if showDBSCAN is True:
                print()
            if show_knn is True:
                print()
            if show_IsolationForest is True:
                clf = IsolationForest(random_state=1, contamination=0.01)
                for feature in self.continuous_features:
                    preds = clf.fit_predict(self.tables[ticker][feature].to_numpy().reshape(-1, 1))
                    totalOutliers = 0
                    for pred in preds:
                        print(ticker, ' = ', pred)
                        if pred == -1:
                            totalOutliers = totalOutliers + 1
                    print("outliers in " + feature + " of " + ticker + " is " + str(totalOutliers))
                    # print("Total number of outliers identified is: ", totalOutliers)



    def perform_preprocessing(self):
        for ticker in self.tickers:
            self.tables[ticker].dropna(inplace=True)


if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    pre_processing = pre_processing([tickers[0]])
    pre_processing.outlier_detection()
