import sys

from keras.utils import data_utils
from matplotlib import pyplot

from explainerdashboard import RegressionExplainer, ClassifierExplainer, ExplainerDashboard

from feature_construction import *
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA


import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, InputLayer
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
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
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

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
            y = self.tables[ticker]['Adj Close']
            plot_pacf(y)
            plt.title(ticker + '_pacf')
            plt.show()

    def find_arima_q(self):
        for ticker in self.tickers:
            y = self.tables[ticker]['Adj Close']
            y = y.diff()
            y.dropna(inplace=True)
            plot_acf(y, lags=30)
            plt.title(ticker + '_acf')
            plt.show()

    def evaluate_arima_model(self, X, arima_order):
        # prepare training dataset
        train_size = int(len(X) * 0.66)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        # make predictions
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        error = mean_squared_error(test, predictions)
        return error

    def test_arima_manual(self):
        p = range(1, 10)
        q = range(150, 300, 50)
        best_configs = {}
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            best_score, best_cfg = float("inf"), None
            print(ticker)
            table = self.tables[ticker]['Adj Close']
            train, test = train_test_split(table, test_size=0.6, shuffle=False)
            for i in range(len(p)):
                for j in range(len(q)):
                    order = (p[i], 0, q[j])
                    print(str(order))
                    mse = self.evaluate_arima_model(train, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
            best_configs[ticker] = str((best_cfg, best_score))
            print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
        print(best_configs)

    def test_auto_arima(self):
        for ticker in self.tickers:
            print(ticker)
            table = self.tables[ticker]['Adj Close']
            train, test = train_test_split(table, test_size=0.2, shuffle=False)
            model = auto_arima(train,
                               start_p=1, start_q=1, max_p=10, max_q=14,
                               start_P=1, start_Q=1, max_P=10, max_Q=14,
                               suppress_warnings=True, error_action='ignore', trace=True)

            print(model.summary())

    def apply_best_arima(self):
        print()

    def test_var(self):
        for ticker in self.tickers:
            table = self.tables[ticker].astype(float)
            table.drop(columns='outlier', inplace=True)
            pca = PCA(n_components=5)
            pca_table = pca.fit_transform(table)

            scaler = StandardScaler()
            scaled_table = scaler.fit_transform(pca_table)

            train, test = train_test_split(scaled_table, test_size=0.2, shuffle=False)

            model = VAR(train)
            x = model.select_order(50)
            print(x.summary())

    def test_lstm(self, past_days=3, future_days=1):
        metric_strings = {}
        epoch_number = 50
        for ticker in self.tickers:

            table = self.tables[ticker].astype(float)
            table.drop(columns=['outlier'], inplace=True)
            scaler = StandardScaler()
            scaled_table = scaler.fit_transform(table)
            num_of_features = scaled_table.shape[1]
            table_X = []
            table_Y = []
            for i in range(past_days, len(scaled_table) - future_days + 1):
                table_X.append(scaled_table[i - past_days:i, 0:num_of_features])
                table_Y.append(scaled_table[i + future_days - 1:i + future_days, 0])


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

            model.fit(train_X, train_Y, validation_split=0.1, epochs=epoch_number, callbacks=[cp], shuffle=False)
            model.save(('models/lstm/' + ticker + '_model_' + str(past_days)))

            model = load_model(('models/lstm/' + ticker + '_model_' + str(past_days)))

            pred_Y = model.predict(test_X)

            dummy = pd.DataFrame(np.zeros((test_X.shape[0], len(table.columns))), columns=table.columns)
            dummy['Adj Close'] = pred_Y
            dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=table.columns)
            pred_Y = dummy['Adj Close'].values

            dummy = pd.DataFrame(np.zeros((test_X.shape[0], len(table.columns))), columns=table.columns)
            dummy['Adj Close'] = test_Y
            dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=table.columns)
            test_Y = dummy['Adj Close'].values

            metric_string = ticker + ' & ' + str(mean_squared_error(test_Y, pred_Y, squared=True)) \
                            + ' & ' +  str(mean_squared_error(test_Y, pred_Y, squared=False)) + ' & ' \
                            + str(mean_absolute_error(test_Y, pred_Y)) + ' & ' \
                            + str(mean_absolute_percentage_error(test_Y, pred_Y))
            metric_strings[ticker] = metric_string
            # print(ticker, ' & ', str(mean_squared_error(test_Y, pred_Y, squared=True)),
            #       ' & ', str(mean_squared_error(test_Y, pred_Y, squared=False)), ' & ',
            #       str(mean_absolute_error(test_Y, pred_Y)), ' & ',
            #       str(mean_absolute_percentage_error(test_Y, pred_Y)))
        print(metric_strings)

    def load_lstm(self, past_days=3):
        for ticker in self.tickers:
            model = load_model(('models/lstm/' + ticker + '_model_' + str(past_days)))

    def test_regression(self, days_ahead=1):
        found_params = {}
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            # filename = ticker + '_output.txt'
            # sys.stdout = open((ticker + '_output.txt'), "w")
            # self.tables[ticker].drop(columns=['outlier'], inplace=True)
            table = self.tables[ticker].astype(float)
            table.drop(columns=['outlier'], inplace=True)
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
                (RandomForestRegressor, {'max_depth': [10, 20, 40, 60, 80, 100, None],
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

            gsearch = GridSearchCV(Pipeline(process), params, cv=cv, scoring=mse, verbose=2)
            gsearch.fit(train_X, train_Y)
            print('ticker = ' + ticker)
            print('score = ' + str(gsearch.score(test_X, test_Y)))
            print('params = ' + str(gsearch.best_params_))
            found_params[ticker] = str(gsearch.best_params_)
        print(found_params)
        np.save('regression_params.npy', found_params)
            # sys.stdout.close()

    def test_best_regression(self, days_ahead=1):
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            table = self.tables[ticker].astype(float)
            table.drop(columns=['outlier'], inplace=True)
            table['label'] = table['Adj Close'].shift(-days_ahead)
            table.dropna(inplace=True)

            x = table.drop(columns=['label'])
            y = table['label']

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            scaled_x = x_scaler.fit_transform(x)
            scaled_y = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

            train_X, test_X = train_test_split(scaled_x, test_size=0.2, shuffle=False)
            train_Y, test_Y = train_test_split(scaled_y, test_size=0.2, shuffle=False)
            model = LinearRegression()
            if ticker == 'AZN.L':
                model = RandomForestRegressor(n_estimators=600, max_features='sqrt', max_depth=10)
            elif ticker == 'SHEL.L':
                model = LinearRegression()
            elif ticker == 'HSBA.L':
                model = LinearRegression()
            elif ticker == 'ULVR.L':
                model = LinearRegression()
            elif ticker == 'DGE.L':
                model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=800)
            elif ticker == 'RIO.L':
                model = LinearRegression()
            elif ticker == 'REL.L':
                model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=1000)
            elif ticker == 'NG.L':
                model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=800)
            elif ticker == 'LSEG.L':
                model = LinearRegression()
            elif ticker == 'VOD.L':
                model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=200)
            model.fit(train_X, train_Y)
            pred_Y = model.predict(test_X)
            pred_Y = y_scaler.inverse_transform(pred_Y.reshape(-1, 1))
            test_Y = y_scaler.inverse_transform(test_Y.reshape(-1, 1))
            # print('ticker = ' + ticker)
            # print("MSE : ", mean_squared_error(test_Y, pred_Y, squared=True))
            # print("RMSE : ", mean_squared_error(test_Y, pred_Y, squared=False))
            # print("MAE : ", mean_absolute_error(test_Y, pred_Y))
            # print("MAPE : ", mean_absolute_percentage_error(test_Y, pred_Y))

            print(ticker, ' & ', str(mean_squared_error(test_Y, pred_Y, squared=True)),
                  ' & ', str(mean_squared_error(test_Y, pred_Y, squared=False)), ' & ',
                  str(mean_absolute_error(test_Y, pred_Y)), ' & ',
                  str(mean_absolute_percentage_error(test_Y, pred_Y)))

    def test_classification(self, days_ahead=1):
        found_params = {}
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            # filename = ticker + '_output.txt'
            # sys.stdout = open((ticker + '_output.txt'), "w")
            # self.tables[ticker].drop(columns=['outlier'], inplace=True)
            table = self.tables[ticker].astype(float)
            table.drop(columns=['outlier'], inplace=True)
            table['label'] = table['Adj Close'].shift(-days_ahead) - table['Adj Close']
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
                (RandomForestClassifier, {'max_depth': [10, 20, 40, 60, 80, 100, None],
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

            gsearch = GridSearchCV(Pipeline(process), params, cv=cv, verbose=2)
            gsearch.fit(train_X, train_Y)
            print('ticker = ' + ticker)
            print('score = ' + str(gsearch.score(test_X, test_Y)))
            print('params = ' + str(gsearch.best_params_))
            found_params[ticker] = str(gsearch.best_params_)
        print(found_params)
        np.save('classification_params.npy', found_params)

    def explain_regression(self, ticker, days_ahead=1):
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            table = self.tables[ticker].astype(float)
            table.drop(columns=['outlier'], inplace=True)
            table['label'] = table['Adj Close'].shift(-days_ahead)
            table.dropna(inplace=True)

            x = table.drop(columns=['label'])
            y = table['label']

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            scaled_x = x_scaler.fit_transform(x)
            scaled_y = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

            train_X, test_X = train_test_split(scaled_x, test_size=0.2, shuffle=False)
            train_Y, test_Y = train_test_split(scaled_y, test_size=0.2, shuffle=False)
            model = LinearRegression()
            if ticker == 'AZN.L':
                model = RandomForestRegressor(n_estimators=600, max_features='sqrt', max_depth=10)
            elif ticker == 'SHEL.L':
                model = LinearRegression()
            elif ticker == 'HSBA.L':
                model = LinearRegression()
            elif ticker == 'ULVR.L':
                model = LinearRegression()
            elif ticker == 'DGE.L':
                model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=800)
            elif ticker == 'RIO.L':
                model = LinearRegression()
            elif ticker == 'REL.L':
                model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=1000)
            elif ticker == 'NG.L':
                model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=800)
            elif ticker == 'LSEG.L':
                model = LinearRegression()
            elif ticker == 'VOD.L':
                model = RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=200)
            model.fit(train_X, train_Y)
            explainer = RegressionExplainer(model, test_X, test_Y,
                                            cats=x.columns)

            ExplainerDashboard(explainer).run()





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
    # model_building.perform_stationarity_transform()
    model_building.find_arima_q()
    # model_building.plot_stationarity_transform()
    # model_building.perform_stationarity_transform()
    # model_building.check_transformed_stationarity()
    # model_building.check_causality()
    # print(model_building.tables['AZN.L'])


    # model_building.test_lstm()
    # model_building.test_regression()
    # model_building.test_classification()
    # model_building.test_auto_arima()
    # model_building.test_arima_manual()
    # model_building.test_var()

    # model_building.test_best_regression()






