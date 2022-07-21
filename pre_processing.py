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

    def outlier_detection(self, show_box_plots=False, show_LocalOutlierFactor=False, showDBSCAN=False, show_knn=False, show_IsolationForest=True):
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
                print()
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
                        if pred == -1:
                            totalOutliers = totalOutliers + 1
                    print("outliers in " + feature + " of " + ticker + " is " + str(totalOutliers))
                    # print("Total number of outliers identified is: ", totalOutliers)



    def perform_preprocessing(self):
        for ticker in self.tickers:
            self.tables[ticker].dropna(inplace=True)


if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    pre_processing = pre_processing(tickers)
    pre_processing.outlier_detection()
