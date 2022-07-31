import sys

from keras.utils import data_utils
from matplotlib import pyplot

from feature_construction import *
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, InputLayer
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
from keras_preprocessing.sequence import TimeseriesGenerator


from tqdm import tqdm

from pmdarima import auto_arima

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


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

    def split_data2(self, future_days, past_days):
        for ticker in self.tickers:
            # self.tables[ticker].reset_index(inplace=True)
            self.tables[ticker].drop(columns=['label', 'outlier'], inplace=True)
            table = self.tables[ticker].astype(float)
            num_of_features = table.shape[1]
            table_X = []
            table_Y = []
            for i in range(past_days, len(table) - future_days + 1):
                table_X.append(table.iloc[i - past_days:i, 0:num_of_features])
                table_Y.append(table.iloc[i + future_days - 1:i + future_days, 0])

            table_X, table_Y = np.array(table_X), np.array(table_Y)

            print('total x = ' + str(table_X.shape))
            print('total y = ' + str(table_Y.shape))
            # print(train_X[0])
            # print(train_Y[0])



            # self.train_X[ticker], self.train_Y[ticker] = train_X[:1983], train_Y[:1983]
            # # self.val_X[ticker], self.val_Y[ticker] = train_X[1735:1983], train_Y[1735:1983]
            # self.test_X[ticker], self.test_Y[ticker] = train_X[1983:], train_Y[1983:]
            # print('train x = ' + str(self.train_X[ticker].shape))
            # print('train y = ' + str(self.train_Y[ticker].shape))
            # # print('val x = ' + str(self.val_X[ticker].shape))
            # # print('val y = ' + str(self.val_Y[ticker].shape))
            # print('test x = ' + str(self.test_X[ticker].shape))
            # print('test y = ' + str(self.test_Y[ticker].shape))


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

    def split_data3(self, past_days):
        for ticker in self.tickers:
            sequences = []
            data_size = len(self.tables[ticker])

            for i in tqdm(range(data_size - past_days)):
                sequence = self.tables[ticker][i:i+past_days]
                label_pos = i + past_days
                label = self.tables[ticker].iloc[label_pos]['label']
                sequences.append((sequence, label))

            print(sequences[0][0])

    def split_data4(self, past_days):
        for ticker in self.tickers:
            target = self.tables[ticker]['label'].to_numpy()
            table = self.tables[ticker].drop(columns=['label', 'outlier']).to_numpy()
            train_X, test_X = train_test_split(table, test_size=0.2, shuffle=False)
            train_Y, test_Y = train_test_split(target, test_size=0.2, shuffle=False)
            sequence_train = TimeseriesGenerator(train_X, train_Y, length=past_days, batch_size=1)
            sequence_test = TimeseriesGenerator(test_X, test_Y, length=past_days, batch_size=1)

            batch_0 = sequence_train[0]
            x, y = batch_0
            print(x)

    def split_data(self, past_days):
        for ticker in self.tickers:
            print(self.tables[ticker].shape)
            df = self.tables[ticker].drop(columns=['outlier', 'label']) #using adj_close instead of label
            table = df.to_numpy()
            x = []
            y = []
            print(range(len(table)-past_days))
            # for i in range(len(table)-past_days):
                # row = [[a] for a in table[i:i+past_days]]
                # x.append(row)
                # label = table[i+past_days]

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
        non_stationary_features = ['Adj Close', 'SMA_14', 'EMA_14', 'BBL_14_2.0', 'BBM_14_2.0', 'BBU_14_2.0']
        for ticker in self.tickers:
            x_to_transform = self.tables[ticker][non_stationary_features]
            self.tables[ticker][non_stationary_features] = x_to_transform.diff()
            self.tables[ticker].dropna(inplace=True)
            # self.train_Y[ticker] = self.train_Y[ticker].diff()
            # self.train_Y[ticker].dropna(inplace=True)
            self.tables[ticker].to_csv('ticker_stationary.csv')

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
            print(ticker)
            table = self.tables[ticker]['Adj Close']
            fit = auto_arima(table, trace=True, suppress_warnings=True)

    def apply_best_arima(self):
        print()

    def test_var(self):
        for ticker in self.tickers:
            self.tables[ticker].drop(columns=['outlier'], inplace=True)
            table = self.tables[ticker].astype(float)

            scaler = StandardScaler()
            scaled_table = scaler.fit_transform(table)

            train, test = train_test_split(scaled_table, test_size=0.2, shuffle=False)

            model = VAR(train)
            x = model.select_order(maxlags=1)
            x.summary()


    def test_lstm(self):
        epoch_number = 50
        batch_size = 64
        past_days = 3
        future_days = 1
        for ticker in self.tickers:
            self.tables[ticker].drop(columns=['outlier'], inplace=True)
            table = self.tables[ticker].astype(float)
            scaler = StandardScaler()
            scaled_table = scaler.fit_transform(scaler)
            num_of_features = scaled_table.shape[1]
            table_X = []
            table_Y = []
            for i in range(past_days, len(scaled_table) - future_days + 1):
                table_X.append(scaled_table.iloc[i - past_days:i, 0:num_of_features])
                table_Y.append(scaled_table.iloc[i + future_days - 1:i + future_days, 0])

            table_X, table_Y = np.array(table_X), np.array(table_Y)

            train_X, test_X = train_test_split(table_X, test_size=0.2, shuffle=False)
            train_Y, test_Y = train_test_split(table_Y, test_size=0.2, shuffle=False)
            # (num_of_rows, timestamps_per_row/how many previous days to consider, num_of_features)
            model = Sequential()
            model.add(LSTM(units=64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
            model.add(LSTM(units=32, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(train_Y.shape[1]))

            print(model.summary())

            cp = ModelCheckpoint('ltsm_models/', save_best_only=True)
            model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

            # self.train_X[ticker] = self.train_X[ticker].reshape((1, self.train_X[ticker].shape[1], self.train_X[ticker].shape[2]))

            history = model.fit(train_X, train_Y, validation_split=0.1, epochs=epoch_number, callbacks=[cp], shuffle=False)

            # train_predictions = model.predict(self.train_X[ticker]).flatten()
            # training_results = pd.DataFrame(data={'predictions': train_predictions, 'actual': self.train_Y[ticker]})
            # print(training_results)

    def test_regression(self, days_ahead=1):
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            # filename = ticker + '_output.txt'
            # sys.stdout = open((ticker + '_output.txt'), "w")
            self.tables[ticker].drop(columns=['outlier'], inplace=True)
            table = self.tables[ticker].astype(float)
            table['label'] = table['Adj Close'].shift(-1)
            table.dropna(inplace=True)

            x = table.drop(columns=['label'])
            y = table['label']

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            scaled_x = x_scaler.fit_transform(x)
            scaled_y = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

            train_X, test_X = train_test_split(scaled_x, test_size=0.2, shuffle=False)
            train_Y, test_Y = train_test_split(scaled_y, test_size=0.2, shuffle=False)

            classifer_paramameters = [
                (LinearRegression, {}),
                (LogisticRegression, {}),
                (DecisionTreeRegressor, {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]}),
                (RandomForestRegressor, {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                          'max_features': ['auto', 'sqrt'],
                                          'n_estimators': [200, 400, 600, 800, 1000]})
            ]
            classifer_combos = [ctor(**para) for ctor, paras in classifer_paramameters for para in ParameterGrid(paras)]
            params = dict(
                transformer=["passthrough"],
                clf=classifer_combos)
            process = [("transformer", "passthrough"), ("clf", LinearRegression)]
            # model = DecisionTreeRegressor()
            # param_search = {'max_depth': [3, 5]}
            cv = TimeSeriesSplit(n_splits=5)

            mse = make_scorer(mean_absolute_error, greater_is_better=False)

            gsearch = GridSearchCV(Pipeline(process), params, cv=cv, verbose=10, scoring=mse)
            gsearch.fit(train_X, train_Y)
            print('ticker = ' + ticker)
            print('score = ' + str(gsearch.score(test_X, test_Y)))
            print('params = ' + str(gsearch.best_params_))

            # sys.stdout.close()

    def test_classification(self):
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            # filename = ticker + '_output.txt'
            # sys.stdout = open((ticker + '_output.txt'), "w")
            self.tables[ticker].drop(columns=['outlier'], inplace=True)
            table = self.tables[ticker].astype(float)
            table['label'] = tables['Adj Close'] - table['Adj Close'].shift(-1)
            table['label'][table['label'] >= 0] = 1
            table['label'][table['label'] < 0] = 0
            table.dropna(inplace=True)

            x = table.drop(columns=['label'])
            y = table['label']

            x_scaler = StandardScaler()
            scaled_x = x_scaler.fit_transform(x)

            train_X, test_X = train_test_split(scaled_x, test_size=0.2, shuffle=False)
            train_Y, test_Y = train_test_split(y, test_size=0.2, shuffle=False)

            classifer_paramameters = [
                (GaussianNB, {}),
                (SVC, {'C': [1,10, 100], "kernel":['linear', 'poly', 'rbf', 'sigmoid']}),
                (DecisionTreeClassifier, {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]}),
                (RandomForestClassifier, {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                         'max_features': ['auto', 'sqrt'],
                                         'n_estimators': [200, 400, 600, 800, 1000]})
            ]
            classifer_combos = [ctor(**para) for ctor, paras in classifer_paramameters for para in ParameterGrid(paras)]
            params = dict(
                transformer=["passthrough"],
                clf=classifer_combos)
            process = [("transformer", "passthrough"), ("clf", LinearRegression)]
            # model = DecisionTreeRegressor()
            # param_search = {'max_depth': [3, 5]}
            cv = TimeSeriesSplit(n_splits=5)

            mse = make_scorer(mean_absolute_error, greater_is_better=False)

            gsearch = GridSearchCV(Pipeline(process), params, cv=cv, verbose=10, scoring=mse)
            gsearch.fit(train_X, train_Y)
            print('ticker = ' + ticker)
            print('score = ' + str(gsearch.score(test_X, test_Y)))
            print('params = ' + str(gsearch.best_params_))




if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    # pre_processing = pre_processing(tickers)
    # pre_processing.perform_preprocessing()
    #
    # pre_tables = pre_processing.tables
    #
    # construct_features = feature_construction(pre_tables)
    # construct_features.final_features()
    # # construct_features.class_construction()
    # # construct_features.scale_variables()
    #
    # # feature_tables = construct_features.tables
    # np.save('feature_tables.npy', construct_features.tables)

    feature_tables = np.load('feature_tables.npy', allow_pickle='TRUE').item()

    model_building = time_series_model_building(feature_tables)
    model_building.tickers = tickers
    # model_building.plot_data()
    # model_building.check_stationarity()
    # model_building.find_arima_p()
    # model_building.find_arima_q()
    # model_building.plot_stationarity_transform()
    model_building.perform_stationarity_transform()
    # model_building.check_transformed_stationarity()
    # model_building.check_causality()
    # model_building.split_data2(1, 14)


    # model_building.test_lstm()
    # model_building.test_regression()
    # model_building.test_arima()
    model_building.test_var()






