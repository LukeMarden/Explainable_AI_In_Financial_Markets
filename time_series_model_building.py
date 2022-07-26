from feature_construction import *
class time_series_model_building:

    def __init__(self, tables):
        self.tables = tables
        self.tickers = list(self.tables.keys())

    def check_stationarity(self):
        print()

    def test_arima(self):
        print()

    def test_ltsm(self):
        print()


if __name__ == '__main__':
    tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'DGE.L', 'RIO.L', 'REL.L', 'NG.L', 'LSEG.L', 'VOD.L']
    pre_processing = pre_processing(tickers)
    pre_processing.perform_preprocessing()

    pre_tables = pre_processing.tables

    construct_features = feature_construction(pre_tables)
    construct_features.final_features()
    construct_features.class_construction()
    construct_features.scale_variables()

    feature_tables = construct_features.tables

    model_building = time_series_model_building(feature_tables)



