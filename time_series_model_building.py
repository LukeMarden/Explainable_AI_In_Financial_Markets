from keras.utils import data_utils
from matplotlib import pyplot

from feature_construction import *
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error

class time_series_model_building:

    def __init__(self, tables):
        self.tables = tables
        self.tickers = list(self.tables.keys())
        self.train_X = {}
        self.train_Y = {}
        self.test_X = {}
        self.test_Y = {}

    def split_data(self, test_size=30):
        for ticker in self.tickers:

            train = self.tables[ticker][0:-test_size]
            test = self.tables[ticker][-test_size:]
            train.drop(train[train['outlier'] == 1].index, inplace=True)

            self.train_Y[ticker] = train['label']
            self.test_Y[ticker] = test['label']

            self.train_X[ticker] = train.drop(columns=['label', 'outlier'])
            self.test_X[ticker] = test.drop(columns=['label', 'outlier'])

    def plot_data(self):
        for ticker in self.tickers:
            for column in self.tables[ticker].columns:
                if column == 'outlier':
                    continue
                self.tables[ticker].plot(y=column, legend=None, use_index=True)
                title = ticker + ' Date vs ' + column
                plt.title(title)
                plt.ylabel(column)
                plt.show()

    def check_stationarity(self, univariate=False):
        if univariate is True:
            for ticker in self.tickers:
                x = self.tables[ticker]['Adj Close']
                y = self.tables[ticker]['label']

        else:
            for ticker in self.tickers:
                for column in self.tables[ticker].columns:
                    print(column)
                    adf_results = adfuller(self.tables[ticker][column])
                    print('ADF = ' + str(adf_results[0]))
                    print('p-value = ' + str(adf_results[1]))
                    print('lags = ' + str(adf_results[2]))
                    print('critical points = ' + str(adf_results[4]))

    def plot_stationarity_transform(self):
        x_stationary_features = ['Adj Close', 'SMA_14', 'EMA_14', 'BBL_14_2.0', 'BBM_14_2.0', 'BBU_14_2.0']
        for ticker in self.tickers:
            x_to_transform = self.train_X[ticker][x_stationary_features]
            x_transformed = x_to_transform.diff().dropna()
            y_transformed = self.train_Y[ticker].diff().dropna()

            for feature in x_stationary_features:
                x_transformed.plot(y=feature, legend=None, use_index=True)
                plt.xlabel('Date')
                plt.ylabel(feature)
                plt.title(ticker + ' Date vs transformed ' + feature)
                plt.show()
            y_transformed.plot(legend=None, use_index=True)
            plt.xlabel('Date')
            plt.ylabel('Label')
            plt.title(ticker + ' Date vs transformed Label')
            plt.show()

    def perform_stationarity_transform(self):
        x_stationary_features = ['Adj Close', 'SMA_14', 'EMA_14', 'BBL_14_2.0', 'BBM_14_2.0', 'BBU_14_2.0']
        for ticker in self.tickers:
            x_to_transform = self.train_X[ticker][x_stationary_features]
            self.train_X[ticker][x_stationary_features] = x_to_transform.diff()
            self.train_X[ticker].dropna(inplace=True)
            self.train_Y[ticker] = self.train_Y[ticker].diff()
            self.train_Y[ticker].dropna(inplace=True)

    def check_transformed_stationarity(self, univariate=False):
        if univariate is True:
            for ticker in self.tickers:
                x = self.tables[ticker]['Adj Close']
                y = self.tables[ticker]['label']

        else:
            for ticker in self.tickers:
                for column in self.train_X[ticker].columns:
                    print(column)
                    adf_results = adfuller(self.train_X[ticker][column])
                    print('ADF = ' + str(adf_results[0]))
                    print('p-value = ' + str(adf_results[1]))
                    print('lags = ' + str(adf_results[2]))
                    print('critical points = ' + str(adf_results[4]))

    def check_causality(self):
        for ticker in self.tickers:
            #code as per: https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Granger%20Causality%20Test.ipynb
            variables = self.train_X[ticker].columns
            maxlag = 3
            verbose = False
            df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
            for c in df.columns:
                for r in df.index:
                    test_result = grangercausalitytests(self.train_X[ticker][[r, c]], maxlag=maxlag, verbose=verbose)
                    p_values = [round(test_result[i + 1][0]['ssr_chi2test'][1], 4) for i in range(maxlag)]
                    if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                    min_p_value = np.min(p_values)
                    df.loc[r, c] = min_p_value
            df.columns = [var + '_x' for var in variables]
            df.index = [var + '_y' for var in variables]

            sns.heatmap(df)
            plt.savefig(('plots/' + ticker + '_causality_matrix.png'), bbox_inches='tight')

    def find_arima_p(self):
        for ticker in self.tickers:
            y = self.train_Y[ticker]
            plot_pacf(y)
            plt.title(ticker + '_pacf')
            plt.show()

    def find_arima_q(self):
        for ticker in self.tickers:
            y = self.train_Y[ticker]
            plot_acf(y, lags=300)
            plt.title(ticker + '_acf')
            plt.show()

    def test_arima(self):
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            print()

    def test_var(self):
        for ticker in self.tickers:
            print()

    def test_ltsm(self):
        epoch_number = 50
        batch_size = 64
        for ticker in self.tickers:
            # (num_of_rows, timestamps_per_row, num_of_features)
            train_X = self.train_X[ticker].reshape((self.train_X[ticker].shape[0], 1, self.train_X[ticker].shape[1]))
            test_X = self.test_X[ticker].reshape((self.test_X[ticker].shape[0], 1, self.test_X[ticker].shape[1]))

            model = keras.Sequential()
            model.add(layers.LSTM(epoch_number, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(1))

            history = model.fit(train_X, self.train_Y[ticker], epochs=epoch_number, batch_size=batch_size,
                                validation_data=(test_X, self.test_Y[ticker]), verbose=2, shuffle=False)

            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            pyplot.show()

            predictions = model.predict(test_X)

            print(predictions[0])



if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    # pre_processing = pre_processing(tickers)
    # pre_processing.perform_preprocessing()
    #
    # pre_tables = pre_processing.tables
    #
    # construct_features = feature_construction(pre_tables)
    # construct_features.final_features()
    # construct_features.class_construction()
    # construct_features.scale_variables()
    #
    # # feature_tables = construct_features.tables
    # np.save('feature_tables.npy', construct_features.tables)

    feature_tables = np.load('feature_tables.npy', allow_pickle='TRUE').item()

    model_building = time_series_model_building(feature_tables)
    model_building.tickers = tickers
    # model_building.plot_data()
    # model_building.check_stationarity()
    model_building.split_data()
    # model_building.find_arima_p()
    # model_building.find_arima_q()
    # model_building.plot_stationarity_transform()
    model_building.perform_stationarity_transform()
    # model_building.check_transformed_stationarity()
    # model_building.check_causality()





