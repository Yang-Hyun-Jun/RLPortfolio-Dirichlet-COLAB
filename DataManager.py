import pandas as pd
import numpy as np
import utils

from collections import defaultdict

Features_Raw = ["Open", "High", "Low", "Close", "Volume", "Price"]


def get_data(path,
             train_date_start=None, train_date_end=None,
             test_date_start=None, test_date_end=None):
    data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
    data = data.replace(0, np.nan)
    data = data.fillna(method="bfill")
    data.insert(data.shape[1], "Price", data["Close"].values)
    data = get_scaling(data)

    train_date_start = data["Date"].iloc[0] if train_date_start is None else train_date_start
    train_date_end = data["Date"].iloc[-1] if train_date_end is None else train_date_end
    test_date_start = data["Date"].iloc[0] if test_date_start is None else test_date_start
    test_date_end = data["Date"].iloc[-1] if test_date_end is None else test_date_end

    train_data = data[(data["Date"] >= train_date_start) & (data["Date"] <= train_date_end)]
    test_data = data[(data["Date"] >= test_date_start) & (data["Date"] <= test_date_end)]

    train_data = train_data.set_index("Date", drop=True)
    test_data = test_data.set_index("Date", drop=True)

    train_data = train_data.astype("float32")
    test_data = test_data.astype("float32")
    return train_data.loc[:,Features_Raw], test_data.loc[:,Features_Raw]


def get_data_tensor(path_list,
                    train_date_start=None, train_date_end=None,
                    test_date_start=None, test_date_end=None):

    for path in path_list:
        train_data, test_data = get_data(path,
                                         train_date_start, train_date_end,
                                         test_date_start, test_date_end)

        if path == path_list[0]:
            common_date_train = set(train_data.index.unique())
            common_date_test = set(test_data.index.unique())
        else:
            common_date_train = common_date_train & set(train_data.index.unique())
            common_date_test = common_date_test & set(test_data.index.unique())

    common_date_train = list(common_date_train)
    common_date_test = list(common_date_test)

    common_date_train.sort()
    common_date_test.sort()

    for path in path_list:
        train_data_, test_data_ = get_data(path,
                                         train_date_start, train_date_end,
                                         test_date_start, test_date_end)

        train_data_ = train_data_[train_data_.index.isin(common_date_train)].to_numpy()
        test_data_ = test_data_[test_data_.index.isin(common_date_test)].to_numpy()

        train_data_ = train_data_[:, :, np.newaxis]
        test_data_ = test_data_[:, :, np.newaxis]

        if path == path_list[0]:
            train_data = train_data_
            test_data = test_data_
        else:
            train_data = np.concatenate([train_data, train_data_], axis=-1)
            test_data = np.concatenate([test_data, test_data_], axis=-1)

    print("-------------------------------------")
    print(f"학습 데이터 시작 거래일:{common_date_train[0]}")
    print(f"학습 데이터 마지막 거래일:{common_date_train[-1]}")
    print(f"테스트 데이터 시작 거래일:{common_date_test[0]}")
    print(f"테스트 데이터 마지막 거래일:{common_date_test[-1]}")
    print("-------------------------------------")
    return train_data, test_data


def get_scaling(data):
    feature_names = list(data.columns)
    not_scaling = ["MACD", "MACDsignal", "MACDoscillator", "Price", "Date"]
    for name in not_scaling:
        if name in data.columns:
            feature_names.remove(name)
    for name in feature_names:
        feature_data = data.loc[:, name]
        feature_mean = feature_data.mean()
        feature_std = feature_data.std()
        data.loc[:, name] = (data.loc[:, name] - feature_mean) / feature_std
    return data


def get_covariance():
    path_list = utils.dataset6
    return_data = defaultdict(float)
    names = [path.split("/")[-1] for path in path_list]

    for i, path in enumerate(path_list):
        data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
        data = data.replace(0, np.nan)
        data = data.fillna(method="bfill")

        data = data[data["Date"] <= "2021-12-31"]
        data = data.set_index("Date", drop=True)

        data = data.astype("float32")
        prices = data["Close"].values
        returns = [0] * len(prices)

        for k in range(len(prices)):
            if k == 0:
                returns[k] = 0.
            else:
                returns[k] = (prices[k] - prices[k - 1]) / prices[k - 1]

        return_data[names[i]] = returns

    len_list = [len(data) for data in return_data.values()]
    min_len = min(len_list)

    for name in names:
        return_data[name] = return_data[name][:min_len]

    return_data = pd.DataFrame(return_data)
    cov = return_data.cov()
    return cov


def get_mean():
    path_list = ["/Users/mac/PycharmProjects/RLPortfolio(Dirichlet for GPU)/Data/HA",
                 "/Users/mac/PycharmProjects/RLPortfolio(Dirichlet for GPU)/Data/WBA",
                 "/Users/mac/PycharmProjects/RLPortfolio(Dirichlet for GPU)/Data/INCY",
                 "/Users/mac/PycharmProjects/RLPortfolio(Dirichlet for GPU)/Data/BIDU",
                 "/Users/mac/PycharmProjects/RLPortfolio(Dirichlet for GPU)/Data/TCOM",
                 "/Users/mac/PycharmProjects/RLPortfolio(Dirichlet for GPU)/Data/AAPL",
                 "/Users/mac/PycharmProjects/RLPortfolio(Dirichlet for GPU)/Data/COST"]

    price_data = defaultdict(float)
    names = [path.split("/")[-1] for path in path_list]

    for i, path in enumerate(path_list):
        data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
        data = data.replace(0, np.nan)
        data = data.fillna(method="bfill")

        data = data[data["Date"] <= "2021-12-31"]
        data = data.set_index("Date", drop=True)

        data = data.astype("float32")
        prices = data["Close"].values
        price_data[names[i]] = prices

    len_list = [len(data) for data in price_data.values()]
    min_len = min(len_list)

    for name in names:
        price_data[name] = price_data[name][:min_len]

    price_data = pd.DataFrame(price_data)
    mean_data = price_data.mean()
    return mean_data


def VaR(stock_list, weight):
    cov = pd.read_csv(utils.DATA_DIR + "/COV", index_col=0)
    cov = cov.loc[stock_list, stock_list]
    weight = np.array(weight)
    variance = weight.T.dot(cov).dot(weight)
    deviation = np.sqrt(variance)
    VaR = 1.65 * 15000000 * deviation
    return VaR


def expected(stock_list, weight):
    # mean_data = pd.read_csv(utils.DATA_DIR + "/MEAN", index_col=0)
    mean_data = pd.read_csv("/Users/mac/PycharmProjects/RLPortfolio(Dirichlet for GPU)/Data/MEAN", index_col=0)
    mean_data = mean_data.loc[stock_list]
    mean_data = np.array(mean_data).reshape(-1)
    weight = np.array(weight).reshape(-1)
    expected = np.dot(mean_data, weight)
    return expected


if __name__ == "__main__":
    mean_data = get_mean()
    print(mean_data.loc[["HA", "BIDU"]])