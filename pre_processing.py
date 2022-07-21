from os import listdir
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from collect_data import *


class pre_processing:

    def __init__(self, tickers):
        self.tickers = tickers
        self.tables = {}
        for ticker in self.tickers:
            self.tables[ticker] = pd.read_csv('data/ftse100/' + ticker + '.csv')

    def missing_values(self):
        for ticker in self.tickers:
            if self.tables[ticker].isnull().any(axis=1).sum():
                print(self.tables[ticker][self.tables[ticker].isna().any(axis=1)])

    def outlier_detection(self, show_box_plots=True):
        for ticker in self.tickers:
            self.tables[ticker].dropna(inplace=True)
            if show_box_plots is True:
                #box plots
                boxplot = self.tables[ticker].boxplot(column=['Open', 'High', 'Low'])
                plt.title(ticker)
                plt.show()
                boxplot = self.tables[ticker].boxplot(column=['Close', 'Adj Close', 'Volume'])
                plt.title(ticker)
                plt.show()


    def perform_preprocessing(self):
        for ticker in self.tickers:
            self.tables[ticker].dropna(inplace=True)


if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    pre_processing = pre_processing(tickers)
    pre_processing.outlier_detection()
