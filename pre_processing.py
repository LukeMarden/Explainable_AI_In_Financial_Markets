from os import listdir
from collect_data import *
class pre_processing:

    def __init__(self, tickers):
        self.tickers = tickers
        self.tables = {}

    def outliers(self):
        print()

    def missing_data(self):
        print()

    def perform_preprocessing(self):
        for ticker in self.tickers:
            df = pd.read_csv('data/ftse100/' + ticker + '.csv')
            # preprocessing process
            self.tables[ticker] = df
            print(df)


if __name__ == '__main__':
    collect_data = collect_data()
    tickers = collect_data.ftse100_tickers
    pre_processing = pre_processing(tickers)
