import pandas as pd
import matplotlib.pyplot as plt
class raw_data_analysis:

    def __init__(self, tickers):
        self.continuous_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.tickers = tickers
        self.tables = {}
        for ticker in self.tickers:
            self.tables[ticker] = pd.read_csv('data/ftse100/' + ticker + '.csv', header=0)


    def show_missing_data(self):
        for ticker in self.tickers:
            print('=======' + ticker + '=======')
            print('size = ' + str(self.tables[ticker].shape))
            print('Total Missing Values = ' + str((self.tables[ticker].isnull().sum().sum() / (self.tables[ticker].shape[0]*self.tables[ticker].shape[1]))*100))
            print('Missing Data Percent By Column= \n' + str(100*(self.tables[ticker].isnull().sum())))

    def show_ranges(self):
        for ticker in self.tickers:
            print('=======' + ticker + '=======')
            pd.set_option('display.max_columns', 6)
            print(str(self.tables[ticker].describe()))
            # print(str(self.tables[ticker].describe()['Volume'].loc['max']))

    def isolate_zero_volume(self):
        data = pd.DataFrame()
        for ticker in self.tickers:
            print('======' + ticker + '======')
            print(self.tables[ticker].query('Volume == 0'))
            data = pd.concat([data, self.tables[ticker].query('Volume == 0')])
        print(data)
        x = pd.unique(data['Date'])
        y = []
        for value in x:
            y.append(data[data['Date'] == value].shape[0])
        plt.bar(x, y, color='green', width=0.2)
        plt.xlabel('Date')
        plt.ylabel('Number Of Occurrences')
        plt.title('0 Volume Trading Days')

        plt.show()

    def plot_raw_data(self, ticker):
        #function used to plot date against close, volume etc
        for variable in self.continuous_features:
            # self.tables[ticker].plot('Date', variable)
            self.tables[ticker].plot('Date', variable, legend=None)
            plt.title(ticker)
            plt.ylabel(variable)
            plt.show()



if __name__ == '__main__':

    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    raw_data_analysis = raw_data_analysis(tickers)
    raw_data_analysis.show_missing_data()
    raw_data_analysis.show_ranges()
    print(raw_data_analysis.tables[tickers[0]].info())
    raw_data_analysis.isolate_zero_volume()
    raw_data_analysis.plot_raw_data(tickers[0])