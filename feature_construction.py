import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns


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

if __name__ == '__main__':
    tables = {}
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    for ticker in tickers:
        tables[ticker] = pd.read_csv('data/ftse100/' + ticker + '.csv')

    construct_features = feature_construction(tables)
    construct_features.process_indicators()
    construct_features.show_correlation()

    construct_features.tables = tables
    construct_features.final_features()
