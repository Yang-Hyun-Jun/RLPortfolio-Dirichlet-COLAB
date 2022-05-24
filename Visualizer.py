import numpy as np
import utils
import matplotlib.pyplot as plt
import DataManager

from mplfinance.original_flavor import candlestick_ohlc

def get_chart_image(stock_code, date_start=None, date_end=None, save_path=None):
    data_path = utils.Base_DIR + "/" + stock_code
    save_path = utils.SAVE_DIR + "/Metrics" + "/Candlestick_train"\
        if save_path == None else save_path

    chart_data = DataManager.get_data(data_path, date_start, date_end)
    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()

    ax.set_ylabel("Price")
    ax.set_xlabel("Day")
    plt.title(f"CandleStick {stock_code}/ {date_start} - {date_end}")
    index = np.arange(len(chart_data))
    ohlc = np.hstack((index.reshape(-1, 1), np.array(chart_data)))  # 차트 데이터에 인덱스 붙이기

    candlestick_ohlc(ax, ohlc, colorup="r", colordown="b")
    plt.xticks(np.linspace(0, len(chart_data), 6))
    plt.grid(True, alpha=0.5)
    fig.savefig(save_path)
    # plt.show()

def get_close_price_curve(stock_code, date_start=None, date_end=None, save_path=None):
    data_path = utils.Base_DIR + "/" + stock_code
    save_path = utils.SAVE_DIR + "/Metrics" + "/Close Price Curve_train"\
        if save_path == None else save_path

    chart_data = DataManager.get_data(data_path, date_start, date_end)
    close_price = chart_data["Close"]
    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Close Price")
    ax.set_xlabel("Day")
    plt.title(f"Close Price {stock_code}/ {date_start} - {date_end}")
    plt.plot(close_price)

    xticks = [int(i) for i in np.linspace(0, len(close_price), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    fig.savefig(save_path)
    # plt.show()

def get_portfolio_value_curve(portfolio_values, save_path=None):
    save_path = utils.SAVE_DIR + "/Metrics" + "/Portfolio Value Curve_train"\
        if save_path == None else save_path

    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Portfolio_values")
    ax.set_xlabel("Time step")
    plt.title("Portfolio_values")
    plt.plot(portfolio_values)

    xticks = [int(i) for i in np.linspace(0, len(portfolio_values), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    fig.savefig(save_path)
    # plt.show()

def get_profitloss_curve(profitlosses, Bench1=None, Bench2=None, save_path=None):
    save_path = utils.SAVE_DIR + "/Metrics" + "/Profitloss Curve_train"\
        if save_path == None else save_path

    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Profitloss")
    ax.set_xlabel("Time step")
    plt.title("Profitloss")
    plt.plot(profitlosses, color='dodgerblue', label="model")

    if Bench1 is not None:
        plt.plot(Bench1, color='violet', label="B&H")
    if Bench2 is not None:
        plt.plot(Bench2, color="red", label="Random")

    xticks = [int(i) for i in np.linspace(0, len(profitlosses), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    plt.legend()
    fig.savefig(save_path)
    # plt.show()

def get_daily_return_curve(daily_returns, save_path=None):
    save_path = utils.SAVE_DIR + "/Metrics" + "/Daily Return Curve_train"\
        if save_path == None else save_path

    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Daily return")
    ax.set_xlabel("Time step")
    plt.title("Daily return")
    plt.plot(daily_returns, label="Daily Return")

    xticks = [int(i) for i in np.linspace(0, len(daily_returns), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    fig.savefig(save_path)
    # plt.show()

def get_action_and_candle(chart_data, action_data, save_path=None):
    chart_data = chart_data
    action_data = action_data

    save_path = utils.SAVE_DIR + "/Metrics" + "/Action and Candle_test"\
        if save_path == None else save_path

    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()

    ax.set_ylabel("Price")
    ax.set_xlabel("Day")

    index = np.arange(len(chart_data))
    ohlc = np.hstack((index.reshape(-1, 1), np.array(chart_data)))
    candlestick_ohlc(ax, ohlc, colorup="r", colordown="b", alpha=0.4)

    for i in range(len(chart_data)):
        y = chart_data.iloc[i]["Close"]
        if action_data.iloc[i] > 0: #Buy
            plt.scatter(i, y, color="m", s=20, alpha=1.0)
        elif action_data.iloc[i] < 0: #Sell
            plt.scatter(i, y, color="c", s=20, alpha=1.0)
        elif action_data.iloc[i] == 0: #Hold
            plt.scatter(i, y, color="yellow", s=20, alpha=1.0)

    xticks = [int(i) for i in np.linspace(0, len(chart_data), 6)]
    plt.xticks(xticks)
    plt.grid(True, alpha=0.5)
    plt.savefig(save_path)
    # plt.show()