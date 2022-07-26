from pre_processing import *
from feature_construction import *
class classification_model_building:

    def __init__(self, tickers, tables, forecast_length=1):
        self.tickers = tickers
        self.tables = tables
        self.forecast_length=forecast_length

    def find_class_distributions(self):
        for ticker in self.tickers:
            print()








if __name__ == '__main__':
    # tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    tickers = ['AZN.L']
    pre_processing = pre_processing(tickers)
    pre_processing.perform_preprocessing()

    classification_model_building = classification_model_building(tickers, pre_processing.tables)

