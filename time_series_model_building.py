import sys

# from keras.utils import data_utils
from matplotlib import pyplot

from explainerdashboard import RegressionExplainer, ClassifierExplainer, ExplainerDashboard
from shap import KernelExplainer
import shap

from shap import DeepExplainer

from feature_construction import *
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from tqdm import tqdm

from pmdarima import auto_arima

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

import numpy as np
import seaborn as sns
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error, recall_score, accuracy_score, f1_score, precision_score

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

    def check_transformed_stationarity(self, number_of_transforms=1):
        for ticker in self.tickers:
            print(ticker)
            table = self.tables[ticker]
            for i in range(number_of_transforms):
                print('transform = ' + str(i))
                table = table.diff()
                table.dropna(inplace=True)
                for column in table.columns:
                    print(column)
                    adf_results = adfuller(table[column])
                    print('ADF = ' + str(adf_results[0]))
                    print('p-value = ' + str(adf_results[1]))
                    print('lags = ' + str(adf_results[2]))
                    print('critical points = ' + str(adf_results[4]))
                print('\n')
            print('\n\n\n')

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
        p = range(0, 4)
        q = range(1, 16)
        best_configs = {}
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            best_score, best_cfg = float("inf"), None
            # print(ticker)
            table = self.tables[ticker]['Adj Close']
            train, test = train_test_split(table, test_size=0.2, shuffle=False)
            for i in range(len(q)):
                for j in range(len(p)):
                    order = (p[j], 0, q[i])
                    # print(str(order))
                    model = ARIMA(train, order=order)
                    fitted_model = model.fit()
                    if fitted_model.aic < best_score:
                        best_score, best_cfg = fitted_model.aic, order
                    # print((str(round(fitted_model.aic, 2)) + ' & '), end=" ")
                    # print('ARIMA%s AIC=%.3f' % (order, fitted_model.aic))
                # print()
            best_configs[ticker] = str((best_cfg, best_score))
            print(ticker + ' & ' + str(best_cfg[0]) + ' & ' + str(best_cfg[2]) + ' & ' + str(round(best_score, 2)))
        print(best_configs)

    def test_auto_arima(self):
        for ticker in self.tickers:
            print(ticker)
            table = self.tables[ticker]['Adj Close']
            table = table.diff()
            table.dropna(inplace=True)
            train, test = train_test_split(table, test_size=0.2, shuffle=False)
            model = auto_arima(train,
                               start_p=1, start_q=1, max_p=3, max_q=15,
                               start_P=1, start_Q=1, max_P=3, max_Q=15,
                               suppress_warnings=True, error_action='ignore', trace=True)

            print(model.summary())

    def apply_best_arima(self):
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            table = self.tables[ticker]['Adj Close']
            train, test = train_test_split(table, test_size=0.2, shuffle=False)
            if ticker == 'AZN.L':
                order = (3, 0, 12)
            elif ticker == 'SHEL.L':
                order = (3, 0, 5)
            elif ticker == 'HSBA.L':
                order = (1, 0, 1)
            elif ticker == 'ULVR.L':
                order = (1, 0, 1)
            elif ticker == 'DGE.L':
                order = (3, 0, 15)
            elif ticker == 'RIO.L':
                order = (3, 0, 13)
            elif ticker == 'REL.L':
                order = (1, 0, 15)
            elif ticker == 'NG.L':
                order = (3, 0, 14)
            elif ticker == 'LSEG.L':
                order = (1, 0, 15)
            elif ticker == 'VOD.L':
                order = (1, 0, 1)

            model = ARIMA(train, order=order)
            model = model.fit()

            pred = model.forecast(test.shape[0])
            pred_df = pd.Series(pred.values, index=test.index)

            print(ticker, ' & ', str(round(mean_squared_error(test, pred_df, squared=True), 3)),
                  ' & ', str(round(mean_squared_error(test, pred_df, squared=False), 2)), ' & ',
                  str(round(mean_absolute_error(test, pred_df), 3)), ' & ',
                  str(round(mean_absolute_percentage_error(test, pred_df), 3)))

    def difference_in_AIC(self):
        warnings.filterwarnings("ignore")
        for ticker in self.tickers:
            table = self.tables[ticker]['Adj Close']
            train, test = train_test_split(table, test_size=0.2, shuffle=False)
            if ticker == 'AZN.L':
                order1 = (1, 0, 5)
                order2 = (3, 0, 12)
            elif ticker == 'SHEL.L':
                order1 = (1, 0, 2)
                order2 = (3, 0, 5)
            elif ticker == 'HSBA.L':
                order1 = (1, 0, 1)
                order2 = (1, 0, 1)
            elif ticker == 'ULVR.L':
                order1 = (1, 0, 1)
                order2 = (1, 0, 1)
            elif ticker == 'DGE.L':
                order1 = (1, 0, 1)
                order2 = (3, 0, 15)
            elif ticker == 'RIO.L':
                order1 = (1, 0, 13)
                order2 = (3, 0, 13)
            elif ticker == 'REL.L':
                order1 = (1, 0, 6)
                order2 = (1, 0, 15)
            elif ticker == 'NG.L':
                order1 = (1, 0, 3)
                order2 = (3, 0, 14)
            elif ticker == 'LSEG.L':
                order1 = (1, 0, 6)
                order2 = (1, 0, 15)
            elif ticker == 'VOD.L':
                order1 = (1, 0, 6)
                order2 = (1, 0, 1)

            model1 = ARIMA(train, order=order1)
            fitted_model1 = model1.fit()

            model2 = ARIMA(train, order=order2)
            fitted_model2 = model2.fit()

            print('& ' + str(round((fitted_model2.aic - fitted_model1.aic), 3)))

    def test_var(self):
        for ticker in self.tickers:
            table = self.tables[ticker].astype(float)
            table.drop(columns='outlier', inplace=True)

            trans_table = table.diff()
            trans_table.dropna(inplace=True)

            pca = PCA(n_components=5)
            pca_table = pca.fit_transform(table)

            scaler = StandardScaler()
            scaled_table = scaler.fit_transform(pca_table)

            train, test = train_test_split(scaled_table, test_size=0.2, shuffle=False)

            model = VAR(train)
            x = model.select_order(50)
            print(ticker)
            print(x.summary())

    def test_best_var(self):
        for ticker in self.tickers:
            table = self.tables[ticker].astype(float)
            table.drop(columns='outlier', inplace=True)

            trans_table = table.diff()
            trans_table.dropna(inplace=True)

            pca = PCA(n_components=5)
            pca_table = pca.fit_transform(table)

            scaler = StandardScaler()
            scaled_table = scaler.fit_transform(pca_table)

            train, test = train_test_split(scaled_table, test_size=0.2, shuffle=False)

            model = VAR(train)

            lag = 0
            if ticker == 'AZN.L':
                lag = 29
            elif ticker == 'SHEL.L':
                lag = 30
            elif ticker == 'HSBA.L':
                lag = 16
            elif ticker == 'ULVR.L':
                lag = 30
            elif ticker == 'DGE.L':
                lag = 29
            elif ticker == 'RIO.L':
                lag = 19
            elif ticker == 'REL.L':
                lag = 32
            elif ticker == 'NG.L':
                lag = 29
            elif ticker == 'LSEG.L':
                lag = 38
            elif ticker == 'VOD.L':
                lag = 29

            results = model.fit(lag)




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

            metric_string = ticker + ' & ' + str(round(mean_squared_error(test_Y, pred_Y, squared=True), 3)) \
                            + ' & ' + str(round(mean_squared_error(test_Y, pred_Y, squared=False), 3)) + ' & ' \
                            + str(round(mean_absolute_error(test_Y, pred_Y), 3)) + ' & ' \
                            + str(round(mean_absolute_percentage_error(test_Y, pred_Y), 3))
            metric_strings[ticker] = metric_string
            # print(ticker, ' & ', str(mean_squared_error(test_Y, pred_Y, squared=True)),
            #       ' & ', str(mean_squared_error(test_Y, pred_Y, squared=False)), ' & ',
            #       str(mean_absolute_error(test_Y, pred_Y)), ' & ',
            #       str(mean_absolute_percentage_error(test_Y, pred_Y)))
        for value in metric_strings.values():
            print(value)

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

            print(ticker, ' & ', str(round(mean_squared_error(test_Y, pred_Y, squared=True), 3)),
                  ' & ', str(round(mean_squared_error(test_Y, pred_Y, squared=False), 3)), ' & ',
                  str(round(mean_absolute_error(test_Y, pred_Y), 3)), ' & ',
                  str(round(mean_absolute_percentage_error(test_Y, pred_Y), 3)))

    def find_class_counts(self):
        for ticker in self.tickers:
            table = self.tables[ticker].astype(float)
            table.drop(columns=['outlier'], inplace=True)
            table['label'] = table['Adj Close'].shift(-1) - table['Adj Close']
            table['label'][table['label'] >= 0] = 1
            table['label'][table['label'] < 0] = 0
            table.dropna(inplace=True)

            y = table['label']

            print(ticker + ' & ' + str(y.value_counts()[0]) + ' & ' + str(y.value_counts()[1]))

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

    def test_best_classification(self, days_ahead=1):
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

            if ticker == 'AZN.L':
                model = SVC(C=100, kernel='linear')
            elif ticker == 'SHEL.L':
                model = RandomForestClassifier(max_features='auto', n_estimators=600)
            elif ticker == 'HSBA.L':
                model = RandomForestClassifier(max_depth=100, max_features='auto', n_estimators=200)
            elif ticker == 'ULVR.L':
                model = SVC(C=1, kernel='linear')
            elif ticker == 'DGE.L':
                model = RandomForestClassifier(max_depth=60, n_estimators=400)
            elif ticker == 'RIO.L':
                model = SVC(C=10, kernel='linear')
            elif ticker == 'REL.L':
                model = SVC(C=100, kernel='linear')
            elif ticker == 'NG.L':
                model = RandomForestClassifier(max_depth=10, max_features='auto', n_estimators=400)
            elif ticker == 'LSEG.L':
                model = RandomForestClassifier(max_depth=10, n_estimators=600)
            elif ticker == 'VOD.L':
                model = RandomForestClassifier(max_depth=10, max_features='auto', n_estimators=200)

            model.fit(train_X, train_Y)
            pred_Y = model.predict(test_X)

            print(ticker + ' & ' +
                  str(round(precision_score(test_Y, pred_Y), 3)) + ' & ' +
                  str(round(recall_score(test_Y, pred_Y), 3)) + ' & ' +
                  str(round(accuracy_score(test_Y, pred_Y),3)) + ' & ' +
                  str(round(f1_score(test_Y, pred_Y),3)))

    def feature_scatter_plots(self, ticker, days_ahead=1):
        table = self.tables[ticker].astype(float)
        table.drop(columns=['outlier'], inplace=True)
        table['label'] = table['Adj Close'].shift(-days_ahead)
        table.dropna(inplace=True)

        x_columns = table.drop(columns=['label'])

        for column in x_columns:
            data = table[[column, 'label']]
            data = data.sort_values(column)

            x_axis = data[column].to_numpy()
            y_axis = data['label'].to_numpy()

            a, b = np.polyfit(x_axis, y_axis, 1)

            plt.scatter(x_axis, y_axis)
            plt.plot(x_axis, a * x_axis + b, color='red')

            plt.xlabel(column)
            plt.ylabel('Future Adjusted Close')
            plt.title((column + ' vs Future Adjusted Close'))

            plt.savefig(('plots/scatters/' + ticker + '/' + column + '.png'), bbox_inches='tight')
            plt.clf()
            # plt.show()

    #Credit: Dr Tahmina Zebin, UEA
    def make_shap_waterfall_plot(self, shap_values, features, num_display=20):

        '''
        A function for building a SHAP waterfall plot.

        SHAP waterfall plot is used to visualize the most important features in a descending order.

        Parameters:
        shap_values (list): SHAP values obtained from a model
        features (pandas DataFrame): a list of features used in a model
        num_display(int): number of features to display

        Returns:
        matplotlib.pyplot plot: SHAP waterfall plot

        '''

        column_list = features.columns
        feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
        column_list = column_list[np.argsort(feature_ratio)[::-1]]
        feature_ratio_order = np.sort(feature_ratio)[::-1]
        cum_sum = np.cumsum(feature_ratio_order)
        column_list = column_list[:num_display]
        feature_ratio_order = feature_ratio_order[:num_display]
        cum_sum = cum_sum[:num_display]

        num_height = 0
        if (num_display >= 20) & (len(column_list) >= 20):
            num_height = (len(column_list) - 20) * 0.4

        fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
        ax1.plot(cum_sum[::-1], column_list[::-1], c='blue', marker='o')
        ax2 = ax1.twiny()
        ax2.barh(column_list[::-1], feature_ratio_order[::-1], alpha=0.6)

        ax1.grid(True)
        ax2.grid(False)
        ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1) + 1, 10))
        ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1) + 1, 10))
        ax1.set_xlabel('Cumulative Ratio')
        ax2.set_xlabel('Composition Ratio')
        ax1.tick_params(axis="y", labelsize=13)
        plt.ylim(-1, len(column_list))

        plt.show()

    def explain_regression(self, days_ahead=1):
        ticker = 'ULVR.L'
        warnings.filterwarnings("ignore")

        table = self.tables[ticker].astype(float)
        table.drop(columns=['outlier'], inplace=True)
        table['label'] = table['Adj Close'].shift(-days_ahead)
        table.dropna(inplace=True)

        x = table.drop(columns=['label'])
        y = table['label']

        x.rename(columns={'BBL_14_2.0': 'BBL_14_2',
                          'BBM_14_2.0': 'BBM_14_2',
                          'BBU_14_2.0': 'BBU_14_2',
                          'BBB_14_2.0': 'BBB_14_2',
                          'BBP_14_2.0': 'BBP_14_2'}, inplace=True)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        scaled_x = x_scaler.fit_transform(x)
        scaled_y = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

        scaled_x = pd.DataFrame(scaled_x, index=x.index, columns=x.columns)
        scaled_y = pd.Series(scaled_y.flatten(), index=y.index, name='label')

        train_X, test_X = train_test_split(scaled_x, test_size=0.2, shuffle=False)
        train_Y, test_Y = train_test_split(scaled_y, test_size=0.2, shuffle=False)

        model = LinearRegression()
        model.fit(train_X, train_Y)
        explainer = RegressionExplainer(model, test_X, test_Y)

        ExplainerDashboard(explainer).run()

    def explain_classification(self, days_ahead=1):
        ticker = 'ULVR.L'
        warnings.filterwarnings("ignore")

        table = self.tables[ticker].astype(float)
        table.drop(columns=['outlier'], inplace=True)
        table['label'] = table['Adj Close'].shift(-days_ahead) - table['Adj Close']
        table['label'][table['label'] >= 0] = 1
        table['label'][table['label'] < 0] = 0
        table.dropna(inplace=True)

        x = table.drop(columns=['label'])
        y = table['label']

        x.rename(columns={'BBL_14_2.0': 'BBL_14_2',
                          'BBM_14_2.0': 'BBM_14_2',
                          'BBU_14_2.0': 'BBU_14_2',
                          'BBB_14_2.0': 'BBB_14_2',
                          'BBP_14_2.0': 'BBP_14_2'}, inplace=True)

        x_scaler = StandardScaler()
        scaled_x = x_scaler.fit_transform(x)

        scaled_x = pd.DataFrame(scaled_x, index=x.index, columns=x.columns)

        train_X, test_X = train_test_split(scaled_x, test_size=0.2, shuffle=False)
        train_Y, test_Y = train_test_split(y, test_size=0.2, shuffle=False)

        model = SVC(C=100, kernel='linear', probability=True)
        model.fit(train_X, train_Y)

        train_num_of_samples=20
        test_num_of_samples=10

        sampled_train_X = shap.sample(test_X, train_num_of_samples)
        sampled_test_X = shap.sample(test_X, test_num_of_samples)
        explainer = KernelExplainer(model.predict_proba, sampled_train_X)
        shap_values = explainer.shap_values(sampled_test_X)

        # self.make_shap_waterfall_plot(shap_values, test_X)

        shap.summary_plot(shap_values, sampled_test_X)
        plt.show()

        shap.force_plot(explainer.expected_value[0], shap_values[0], test_X.iloc[0, :])
        plt.show()

        shap.force_plot(shap_values)
        plt.show()

        shap.plots.waterfall(shap_values[0])
        plt.show()



    def test_regression_day_forecasting(self, values=[1, 3, 7, 14, 30, 180]):
        first_run = True
        for value in values:
            ticker = 'ULVR.L'
            warnings.filterwarnings("ignore")

            table = self.tables[ticker].astype(float)
            table.drop(columns=['outlier'], inplace=True)
            table['label'] = table['Adj Close'].shift(-value)
            table['prediction_date'] = table.index.values
            table['prediction_date'] = table['prediction_date'].shift(-value)
            table.dropna(inplace=True)

            x = table.drop(columns=['label', 'prediction_date'])
            y = pd.Series(data=table['label'].to_numpy(), index=pd.DatetimeIndex(table['prediction_date']))

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            scaled_x = x_scaler.fit_transform(x)
            scaled_y = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

            scaled_x = pd.DataFrame(scaled_x, index=x.index, columns=x.columns)
            scaled_y = pd.Series(scaled_y.flatten(), index=y.index, name='label')

            train_X, test_X = train_test_split(scaled_x, test_size=0.2, shuffle=False)
            train_Y, test_Y = train_test_split(scaled_y, test_size=0.2, shuffle=False)

            model = LinearRegression()
            model.fit(train_X, train_Y)



            if first_run:
                unscaled_train_Y = y_scaler.inverse_transform(train_Y.to_numpy().reshape(-1, 1))
                unscaled_test_Y = y_scaler.inverse_transform(test_Y.to_numpy().reshape(-1, 1))

                train_Y = pd.Series(data=unscaled_train_Y.flatten(), index=train_Y.index)
                test_Y = pd.Series(data=unscaled_test_Y.flatten(), index=test_Y.index)

                plt.plot(train_Y, label='Training', linewidth=1)
                plt.plot(test_Y, label='Actual', linewidth=1)

                first_run = False

            pred_Y = model.predict(test_X)
            pred_Y = y_scaler.inverse_transform(pred_Y.reshape(-1, 1))
            pred_Y = pd.Series(data=pred_Y.flatten(), index=test_Y.index)

            print(str(value) + ' & ' + str(round(mean_squared_error(test_Y, pred_Y, squared=True), 3)),
                  ' & ', str(round(mean_squared_error(test_Y, pred_Y, squared=False), 3)), ' & ',
                  str(round(mean_absolute_error(test_Y, pred_Y), 3)), ' & ',
                  str(round(mean_absolute_percentage_error(test_Y.to_numpy(), pred_Y.to_numpy()), 3)))

            plt.plot(pred_Y, label=(str(value) + ' Day(s) Ahead'), linewidth=0.5)
        plt.legend(loc="upper left")
        plt.show()



    def test_linear_plotting(self, ticker, days_ahead=1):
        warnings.filterwarnings("ignore")

        table = self.tables[ticker].astype(float)
        table.drop(columns=['outlier'], inplace=True)
        table['label'] = table['Adj Close'].shift(-days_ahead)
        table.dropna(inplace=True)

        x = table.drop(columns=['label'])
        y = table['label']

        x.rename(columns={'BBL_14_2.0': 'BBL_14_2',
                          'BBM_14_2.0': 'BBM_14_2',
                          'BBU_14_2.0': 'BBU_14_2',
                          'BBB_14_2.0': 'BBB_14_2',
                          'BBP_14_2.0': 'BBP_14_2'}, inplace=True)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        scaled_x = x_scaler.fit_transform(x)
        scaled_y = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

        scaled_x = pd.DataFrame(scaled_x, index=x.index, columns=x.columns)
        scaled_y = pd.Series(scaled_y.flatten(), index=y.index, name='label')

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
        test_Y = y_scaler.inverse_transform(test_Y.to_numpy().reshape(-1, 1))
        train_Y = y_scaler.inverse_transform(train_Y.to_numpy().reshape(-1, 1))

        train_Y = pd.Series(data=train_Y.flatten(), index=train_X.index)
        test_Y = pd.Series(data=test_Y.flatten(), index=test_X.index)
        pred_Y = pd.Series(data=pred_Y.flatten(), index=test_X.index)

        plt.plot(train_Y, label='Training', linewidth=1)
        plt.plot(pred_Y, label='Predicted', linewidth=2)
        plt.plot(test_Y, label='Actual', linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('Adj Close')
        plt.title((ticker + '\'s Simple Regression Model'))
        plt.legend(loc="upper left")
        plt.show()

    def test_arima_plotting(self, ticker):
        warnings.filterwarnings("ignore")


        table = self.tables[ticker]['Adj Close']
        train, test = train_test_split(table, test_size=0.2, shuffle=False)

        if ticker == 'AZN.L':
            order = (3, 0, 12)
        elif ticker == 'SHEL.L':
            order = (3, 0, 5)
        elif ticker == 'HSBA.L':
            order = (1, 0, 1)
        elif ticker == 'ULVR.L':
            order = (1, 0, 1)
        elif ticker == 'DGE.L':
            order = (3, 0, 15)
        elif ticker == 'RIO.L':
            order = (3, 0, 13)
        elif ticker == 'REL.L':
            order = (1, 0, 15)
        elif ticker == 'NG.L':
            order = (3, 0, 14)
        elif ticker == 'LSEG.L':
            order = (1, 0, 15)
        elif ticker == 'VOD.L':
            order = (1, 0, 1)

        model = ARIMA(train, order=order)
        model = model.fit()

        pred = model.forecast(test.shape[0])
        pred_df = pd.Series(pred.values, index=test.index)

        plt.plot(train, label='Training', linewidth=1)
        plt.plot(pred_df, label='Predicted', linewidth=2)
        plt.plot(test, label='Actual', linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('Adj Close')
        plt.title((ticker + '\'s ARIMA Model'))
        plt.legend(loc="upper left")
        plt.show()

    def test_lstm_plotting(self, ticker, future_days=1, past_days=7, epoch_number=1):
        warnings.filterwarnings("ignore")

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

        index = table.index
        train_index, test_index = train_test_split(index, test_size=0.2, shuffle=False)
        train_X, test_X = train_test_split(table_X, test_size=0.2, shuffle=False)
        train_Y, test_Y = train_test_split(table_Y, test_size=0.2, shuffle=False)

        # (num_of_rows, timestamps_per_row/how many previous days to consider, num_of_features)
        model = Sequential()
        model.add(LSTM(units=64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(train_Y.shape[1]))

        cp = ModelCheckpoint('ltsm_models/', save_best_only=True)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

        model.fit(train_X, train_Y, validation_split=0.1, epochs=epoch_number, callbacks=[cp], shuffle=False)

        pred_Y = model.predict(test_X)

        dummy = pd.DataFrame(np.zeros((test_X.shape[0], len(table.columns))), columns=table.columns)
        dummy['Adj Close'] = pred_Y
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=table.columns)
        pred_Y = pd.Series(dummy['Adj Close'].values, index=test_index[:-future_days])

        dummy = pd.DataFrame(np.zeros((test_X.shape[0], len(table.columns))), columns=table.columns)
        dummy['Adj Close'] = test_Y
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=table.columns)
        test_Y = pd.Series(dummy['Adj Close'].values, index=test_index[:-future_days])

        dummy = pd.DataFrame(np.zeros((train_X.shape[0], len(table.columns))), columns=table.columns)
        dummy['Adj Close'] = train_Y
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=table.columns)
        train_Y = pd.Series(dummy['Adj Close'].values, index=train_index[past_days-future_days:])


        print(train_Y)
        # train_Y = pd.Series(train_Y.flatten(), index=train_index[:-past_days+future_days])

        plt.plot(train_Y, label='Training', linewidth=1)
        plt.plot(pred_Y, label='Predicted', linewidth=2)
        plt.plot(test_Y, label='Actual', linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('Adj Close')
        plt.title((ticker + '\'s LSTM Model'))
        plt.legend(loc="upper left")
        plt.show()

    def test_classification_plotting(self, ticker, days_ahead=1):
        warnings.filterwarnings("ignore")

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

        model = SVC()
        if ticker == 'AZN.L':
            model = SVC(C=100, kernel='linear')
        elif ticker == 'SHEL.L':
            model = RandomForestClassifier(max_features='auto', n_estimators=600)
        elif ticker == 'HSBA.L':
            model = RandomForestClassifier(max_depth=100, max_features='auto', n_estimators=200)
        elif ticker == 'ULVR.L':
            model = SVC(C=1, kernel='linear')
        elif ticker == 'DGE.L':
            model = RandomForestClassifier(max_depth=60, n_estimators=400)
        elif ticker == 'RIO.L':
            model = SVC(C=10, kernel='linear')
        elif ticker == 'REL.L':
            model = SVC(C=100, kernel='linear')
        elif ticker == 'NG.L':
            model = RandomForestClassifier(max_depth=10, max_features='auto', n_estimators=400)
        elif ticker == 'LSEG.L':
            model = RandomForestClassifier(max_depth=10, n_estimators=600)
        elif ticker == 'VOD.L':
            model = RandomForestClassifier(max_depth=10, max_features='auto', n_estimators=200)

        model.fit(train_X, train_Y)
        pred_Y = model.predict(test_X)

        cf_matrix = confusion_matrix(test_Y, pred_Y)


        #credit https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/

        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

        ax.set_title('Confusion Matrix of ' + ticker)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ')

        ax.xaxis.set_ticklabels(['False', 'True'])
        ax.yaxis.set_ticklabels(['False', 'True'])

        plt.show()

if __name__ == '__main__':
    # shap.initjs()
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    # tickers = ['AZN.L']
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

    # values = feature_tables['AZN.L'].drop(columns=['outlier'])
    # feature_tables['AZN.L'] = feature_tables['AZN.L'][feature_tables['AZN.L'][values.columns] != 0]

    model_building = time_series_model_building(feature_tables)
    model_building.tickers = tickers

    # model_building.plot_data()
    # model_building.check_stationarity()
    # model_building.find_class_counts()
    # model_building.feature_scatter_plots('ULVR.L')

    # model_building.perform_stationarity_transform()
    # model_building.find_arima_q()
    # model_building.find_arima_p()
    # model_building.plot_stationarity_transform()
    # model_building.perform_stationarity_transform()
    # model_building.check_transformed_stationarity()
    # model_building.check_causality()


    # model_building.test_regression()
    # model_building.test_classification()
    # # model_building.test_auto_arima()
    # model_building.test_arima_manual()
    # model_building.difference_in_AIC()
    model_building.test_var()

    # model_building.test_regression_day_forecasting()

    # model_building.test_lstm()
    # model_building.test_best_regression()
    # model_building.test_best_classification()
    # model_building.apply_best_arima()

    # model_building.test_linear_plotting('ULVR.L')
    # model_building.test_linear_plotting('NG.L')
    # model_building.test_arima_plotting('ULVR.L')
    # model_building.test_arima_plotting('RIO.L')
    # model_building.test_lstm_plotting('VOD.L')
    # model_building.test_lstm_plotting('LSEG.L')
    # model_building.test_classification_plotting('REL.L')
    # model_building.test_classification_plotting('DGE.L')

    # model_building.explain_regression()
    # model_building.explain_classification()








