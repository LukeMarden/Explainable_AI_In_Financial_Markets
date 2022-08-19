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
    feature_construction.final_features()
    feature_tables = feature_construction.tables
    
    # # plot lstm variations
    # x = [1,3,7,14,30,100]
    # lag3 = [0.029, 0.039, 0.043, 0.055, 0.077, 0.097]
    # lag7 = [0.037, 0.04, 0.057, 0.077, 0.09, 0.108]
    # lag14 = [0.047, 0.059, 0.065, 0.082, 0.1, 0.134]
    # lag30 = [0.046, 0.072, 0.067, 0.099, 0.119, 0.117]
    # lag50 = [0.048, 0.063, 0.095, 0.09, 0.124, 0.119]
    #
    # plt.plot(x, lag3, label=('lag = 3'), linewidth=0.5)
    # plt.plot(x, lag7, label=('lag = 7'), linewidth=0.5)
    # plt.plot(x, lag14, label=('lag = 14'), linewidth=0.5)
    # plt.plot(x, lag30, label=('lag = 30'), linewidth=0.5)
    # plt.plot(x, lag50, label=('lag = 50'), linewidth=0.5)
    # plt.xticks(x)
    # plt.legend(loc="upper left")
    # plt.show()







