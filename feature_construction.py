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




if __name__ == '__main__':
    token = 'MSFT'
    data = pd.read_csv('data/s&p500/' + token + '.csv')
    print(simple_moving_average(data, 20))
