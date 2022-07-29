from keras.utils import data_utils
from matplotlib import pyplot

from feature_construction import *
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, InputLayer
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model


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
        self.val_X = {}
        self.val_Y = {}
        self.test_X = {}
        self.test_Y = {}

    def split_data(self, future_days, past_days, test_size=30):
        for ticker in self.tickers:
            self.tables[ticker].drop(columns=['label', 'outlier'], inplace=True)
            num_of_features = len(self.tables[ticker].columns)
            train_X = []
            train_Y = []
            for i in range(past_days, len(self.tables[ticker]) - future_days + 1):
                train_X.append(self.tables[ticker].iloc[i - past_days:i, 0:num_of_features])
                train_Y.append(self.tables[ticker].iloc[i + future_days: - 1:i + future_days, 0])

            train_X, train_Y = np.array(train_X), np.array(train_Y)

            print('total x = ' + str(train_X.shape))
            print('total y = ' + str(train_Y.shape))

            self.train_X[ticker], self.train_Y[ticker] = train_X[:1735], train_Y[:1735]
            self.val_X[ticker], self.val_Y[ticker] = train_X[1735:1983], train_Y[1735:1983]
            self.test_X[ticker], self.test_Y[ticker] = train_X[1983:], train_Y[1983:]
            print('train x = ' + str(self.train_X[ticker].shape))
            print('train y = ' + str(self.train_Y[ticker].shape))
            print('val x = ' + str(self.val_X[ticker].shape))
            print('val y = ' + str(self.val_Y[ticker].shape))
            print('test x = ' + str(self.test_X[ticker].shape))
            print('test y = ' + str(self.test_Y[ticker].shape))


            # test_X = train_X[360:, 0:13, 0:17]
            # print(test_X.shape)

            # train = self.tables[ticker][0:-test_size]
            # test = self.tables[ticker][-test_size:]
            # train.drop(train[train['outlier'] == 1].index, inplace=True)
            #
            # self.train_Y[ticker] = train['label']
            # self.test_Y[ticker] = test['label']
            #
            # self.train_X[ticker] = train.drop(columns=['label', 'outlier'])
            # self.test_X[ticker] = test.drop(columns=['label', 'outlier'])

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

    def test_lstm(self):
        epoch_number = 50
        batch_size = 64
        neurons = 50
        for ticker in self.tickers:
            # (num_of_rows, timestamps_per_row/how many previous days to consider, num_of_features)
            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(self.train_X[ticker].shape[1], self.train_X[ticker].shape[2]), return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(self.train_Y[ticker].shape[1]))

            print(model.summary())

            cp = ModelCheckpoint('ltsm_models/', save_best_only=True)
            model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

            model.fit(self.train_X[ticker], self.train_Y[ticker], validation_data=(self.val_X[ticker], self.val_Y[ticker]), epochs=epoch_number, callbacks=[cp])

            train_predictions = model.predict(self.train_X[ticker]).flatten()
            training_results = pd.DataFrame(data={'predictions': train_predictions, 'actual': self.train_Y[ticker]})
            print(training_results)



if __name__ == '__main__':
    tickers = ['AZN.L'] #['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
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
    model_building.split_data(1, 14)
    # model_building.find_arima_p()
    # model_building.find_arima_q()
    # model_building.plot_stationarity_transform()
    # model_building.perform_stationarity_transform()
    # model_building.check_transformed_stationarity()
    # model_building.check_causality()





