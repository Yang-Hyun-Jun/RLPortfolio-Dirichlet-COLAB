import numpy as np


Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/content"
DATA_DIR = "/content/RLPortfolio-Dirichlet-COLAB/Data"

STOCK_LIST = None
NOW_PORT = None
NOW_PRICE = None
NOW_PV = None
NOW_BALANCE = None
NOW_STOCKS = None


a = "/content/RLPortfolio-Dirichlet-COLAB/Data/HA"
b = "/content/RLPortfolio-Dirichlet-COLAB/Data/WBA"
c = "/content/RLPortfolio-Dirichlet-COLAB/Data/INCY"
d = "/content/RLPortfolio-Dirichlet-COLAB/Data/AAPL"
e = "/content/RLPortfolio-Dirichlet-COLAB/Data/COST"
f = "/content/RLPortfolio-Dirichlet-COLAB/Data/BIDU"
g = "/content/RLPortfolio-Dirichlet-COLAB/Data/TCOM"


dataset1 = [a, b, c]
dataset2 = [a, b, c, f]
dataset3 = [a, b, c, f, g]
dataset4 = [a, b, c, d]
dataset5 = [a, b, c, d, e]
dataset6 = [a, b, c, f, g, d, e]
dataset7 = [f, g, d, e]


"""""""""""""""""""""""""""""""""""""""
            테스트를 위한 메소드
"""""""""""""""""""""""""""""""""""""""


def decide_trading_unit(confidence, price):
    trading_amount = NOW_PV * confidence
    trading_unit = int(np.array(trading_amount) / price)
    return trading_unit


def validate_action(action, delta):
    m_action = action.copy()
    for i in range(action.shape[0]):
        if delta < action[i] <= 1:
            if NOW_BALANCE < NOW_PRICE[i] * (1 + 0.0025):
                m_action[i] = 0.0

        elif -1 <= action[i] < -delta:
            if NOW_STOCKS[i] == 0:
                m_action[i] = 0.0
    return m_action


def check_fee(action):
    fee = 0
    delta = 0.005
    cost = 0.0025
    max_trading_price = 400
    close_p = NOW_PRICE
    confidence = abs(action)
    m_action = validate_action(action, delta)

    for i in range(m_action.shape[0]):
        price = close_p[i]
        if -1 <= m_action[i] < -delta:
            trading_unit = decide_trading_unit(confidence[i], price)
            trading_unit = min(trading_unit, NOW_STOCKS)
            invest_amount = price * trading_unit
            fee += invest_amount * cost

    for i in range(m_action.shape[0]):
        price = close_p[i]
        if delta < m_action[i] <= 1:
            trading_unit = decide_trading_unit(confidence[i], price)
            cal_balance = (NOW_BALANCE - price * trading_unit * (1+cost))

            if cal_balance < 0:
                trading_unit = min(
                    int(NOW_BALANCE / (price * (1+cost))),
                    int(max_trading_price / (price * (1+cost))))

            invest_amount = price * trading_unit
            fee += invest_amount * cost
    return fee
