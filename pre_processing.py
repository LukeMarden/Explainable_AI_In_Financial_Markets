from os import listdir
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


    def outlier_detection(self):
        for ticker in self.tickers:
            self.tables[ticker].dropna(inplace=True)





    def perform_preprocessing(self):
        print()


if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    pre_processing = pre_processing(tickers)
    pre_processing.missing_values()
