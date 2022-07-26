import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pre_processing import *


def simple_moving_average(data, num_of_days):
    moving_average = data['Close'].rolling(num_of_days).mean()
    return pd.Series(moving_average, name='simple_moving_average')

def exponential_moving_average(data, num_of_days):
    moving_average = data['Close'].ewm(span=num_of_days, adjust=False).mean()
    return pd.Series(moving_average, name='exponential_moving_average')

def bollinger_bands(data, num_of_days, band_size=2):
    sma = simple_moving_average(data, num_of_days).tolist()
    std = data['Close'].rolling(num_of_days).std()
    upper_band = pd.Series((sma + std * band_size), name='upper_bollinger_band')
    lower_band = pd.Series((sma - std * band_size), name='lower_bollinger_band')
    return upper_band, lower_band

def relative_strength_index(data, period=14, simple=True):
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = delta.clip(upper=0) * -1
    if simple is True: #simple moving average
        up_moving_average = simple_moving_average(up, period)
        dowm_moving_average = simple_moving_average(down, period)
    else: #exponential moving average
        up_moving_average = exponential_moving_average(up, period)
        dowm_moving_average = exponential_moving_average(down, period)
    return 100 - (100/(1 + up_moving_average/dowm_moving_average)) #rsi


class feature_construction:
    def __init__(self, tables):
        self.tables = tables
        self.tickers = list(self.tables.keys())

    def process_indicators(self):

        for ticker in self.tickers:
            df = self.tables[ticker]
            df.set_index(pd.DatetimeIndex(df["Date"]), inplace=True)

            mystrat = ta.Strategy(
                name='first',
                ta=[
                    {"kind": "sma", "length": 7},
                    {"kind": "sma", "length": 14},
                    {"kind": "sma", "length": 30},
                    {"kind": "ema", "length": 7},
                    {"kind": "ema", "length": 14},
                    {"kind": "ema", "length": 30},
                    {"kind": "rsi", "length": 14},
                    {"kind": "macd"},
                    {"kind": "adx", "length": 14},
                    {"kind": "stoch"},
                    {"kind": "bbands", "length": 14, "std": 2},
                ]
            )
            df.ta.strategy(mystrat)
            df.dropna(inplace=True)
            # print(ticker + ' & ' + str(df.shape[0]) + ' & ' + str(df.shape[1]))
            # df.to_csv(ticker + '_features.csv')
            self.tables[ticker] = df

    def show_correlation(self):
        for ticker in self.tickers:
            df = self.tables[ticker]
            df.drop(columns='outlier', inplace=True)
            df.drop('Date', axis=1, inplace=True)
            plt.figure(figsize=(12, 10))
            sns.heatmap(df.corr(), annot=False, cmap=plt.cm.Reds)
            plt.show()

    def final_features(self):
        for ticker in self.tickers:
            df = self.tables[ticker]
            df.set_index(pd.DatetimeIndex(df["Date"]), inplace=True)

            mystrat = ta.Strategy(
                name='first',
                ta=[
                    {"kind": "sma", "length": 14},
                    {"kind": "ema", "length": 14},
                    {"kind": "rsi", "length": 14},
                    {"kind": "macd"},
                    {"kind": "adx", "length": 14},
                    {"kind": "stoch"},
                    {"kind": "bbands", "length": 14, "std": 2},
                ]
            )
            df.ta.strategy(mystrat)
            df.dropna(inplace=True)
            df.drop(['MACDs_12_26_9', 'Open', 'High', 'Low', 'Close'], inplace=True)
            self.tables[ticker] = df

    def class_construction(self, time=1, classification=False):
        for ticker in self.tickers:
            self.tables[ticker]['label'] = self.tables[ticker]['Adj Close'].shift(-time)
            self.tables[ticker].dropna(inplace=True)
            if classification is True:
                self.tables[ticker]['class'] = self.tables[ticker]['label'] - self.tables[ticker]['Adj Close']
                self.tables[ticker]['class'][self.tables[ticker]['class'] >= 0] = 1
                self.tables[ticker]['class'][self.tables[ticker]['class'] < 0] = 0
                self.tables[ticker]['label'] = self.tables[ticker]['class']
                self.tables[ticker].drop(columns=['class'], inplace=True)

    def scale_variables(self, classification=False):
        for ticker in self.tickers:
            x_scaler = MinMaxScaler()

            y = self.tables[ticker]['label']
            x = self.tables[ticker].drop(columns=['label', 'outlier'])

            x_columns = x.columns.values.tolist()
            x_scaler.fit_transform(x)
            self.tables[ticker][x_columns] = x

            if classification is False:
                y_scaler = MinMaxScaler()

                y_columns = y.columns.values.tolist()
                y_scaler.fit_transform(y)
                self.tables[ticker][y_columns] = y






if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    pre_processing = pre_processing(tickers)
    pre_processing.perform_preprocessing()

    tables = pre_processing.tables

    construct_features = feature_construction(tables)
    construct_features.process_indicators()
    construct_features.show_correlation()

    construct_features.tables = tables
    construct_features.final_features()
    construct_features.class_construction()
    construct_features.scale_variables()
