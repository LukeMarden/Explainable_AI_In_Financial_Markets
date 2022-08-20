import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta

class aim3:

    def __init__(self, tickers):
        self.tickers = tickers
        self.tables = {}
        for ticker in self.tickers:
            self.tables[ticker] = pd.read_csv('data/ftse100/' + ticker + '.csv', header=0)
            self.tables[ticker] = self.tables[ticker].set_index('Date')
            self.tables[ticker].index = pd.to_datetime(self.tables[ticker].index)

    def analyse_event(self, event_start, event_name, show=False):
        event_date = datetime.strptime(event_start, '%Y-%m-%d')
        before = event_date - timedelta(days=100)
        before = before.strftime('%Y-%m-%d')
        after = event_date + timedelta(days=100)
        after = after.strftime('%Y-%m-%d')

        gradient_changes = []
        for ticker in self.tickers:
            plt.clf()
            table = self.tables[ticker]['Adj Close']
            before_df = table[before: event_start]
            after_df = table[event_start: after]


            plt.plot(before_df, label='Before')
            plt.plot(after_df, label='After')

            before_before_reset = before_df.index
            before_df = before_df.reset_index(drop=True)
            a_before, b_before = np.polyfit(before_df.index, before_df.to_numpy(), 1)

            before_after_reset = after_df.index
            after_df = after_df.reset_index(drop=True)
            a_after, b_after = np.polyfit(after_df.index, after_df.to_numpy(), 1)

            #a = gradient
            plt.plot(before_before_reset, a_before * before_df.index + b_before, label='Before LOBF')
            plt.plot(before_after_reset, a_after * after_df.index + b_after, label='After LOBF')
            plt.xlabel('Date')
            plt.ylabel('Adjusted Close')
            plt.title('Effect Of ' + event_name + ' On ' + ticker)
            plt.legend(loc="upper left")
            plt.savefig(("plots/aim3/" + event_name + "_" + ticker + ".png"), bbox_inches='tight', dpi=100)
            if show:
                plt.show()

            gradient_change = round((a_after-a_before), 3)
            gradient_changes.append(gradient_change)
            print(ticker + ' & ' +
                  str(round(a_before, 3)) + ' & ' +
                  str(round(a_after, 3)) + ' & ' +
                  str(gradient_change) + ' \\\\ \\hline')

        plt.clf()
        x = self.tickers
        y = gradient_changes
        plt.bar(x, y)
        plt.title('Change In Gradient For ' + event_name)
        plt.xlabel('Tickers')
        plt.ylabel('Change In Gradient')
        plt.savefig(("plots/aim3/" + event_name + "_gradient_change.png"), bbox_inches='tight', dpi=100)

        if show:
            plt.show()


if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    # tickers = ['AZN.L']
    aim3 = aim3(tickers)

    aim3.tickers = tickers

    #brexit
    aim3.analyse_event('2016-06-01', 'Brexit')
    #covid-19
    aim3.analyse_event('2019-12-01', 'COVID-19')

