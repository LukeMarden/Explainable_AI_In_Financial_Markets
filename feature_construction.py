import pandas as pd

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








if __name__ == '__main__':
    token = 'MSFT'
    data = pd.read_csv('data/s&p500/' + token + '.csv')
    print(moving_average(data, 20))
