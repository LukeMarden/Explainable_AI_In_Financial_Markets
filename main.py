from collect_data import *
from feature_construction import *

if __name__ == '__main__':
    #collect data
    collect_data = collect_data()
    # collect_data.download_ftse100()
    ftse100_tickers = collect_data.ftse100_tickers

    #pre processing
    


    # contruct features such as indicators
    feature_construction = feature_construction(ftse100_tickers)
    feature_construction.process_indicators()
    tables = feature_construction.tables








