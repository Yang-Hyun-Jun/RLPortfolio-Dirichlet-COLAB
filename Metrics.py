import numpy as np
import pandas as pd
import utils

class Metrics:
    def __init__(self):
        self.portfolio_values = [] #portfolio_value는 첫 step 진행 이후부터 저장
        self.profitlosses = []
        self.profitlosses_BH = []
        self.daily_returns = []
        self.cum_fees = []
        self.balances = []
        self.total_return = None
        self.volatility = None

    def reset(self):
        self.portfolio_values = []
        self.profitlosses = []
        self.daily_returns = []
        self.total_return = None
        self.volatility = None

    def get_fees(self, save_path=None):
        save_path = utils.SAVE_DIR + "/Metrics" + "/cum_fees"\
            if save_path is None else save_path
        df = pd.DataFrame({"cum_fees": self.cum_fees})
        df.to_csv(save_path)

    def get_balances(self, save_path=None):
        save_path = utils.SAVE_DIR + "/Metrics" + "/balances"\
            if save_path is None else save_path
        df = pd.DataFrame({"Balances": self.balances})
        df.to_csv(save_path)

    def get_profitlosses_BH(self, save_path=None):
        save_path = utils.SAVE_DIR + "/Metrics" + "/Profitloss_BH"\
            if save_path is None else save_path
        df = pd.DataFrame({"Profitloss BH": self.profitlosses_BH})
        df.to_csv(save_path)

    def get_portfolio_values(self, save_path=None):
        save_path = utils.SAVE_DIR + "/Metrics" + "/Portfolio Value_train"\
            if save_path is None else save_path
        df = pd.DataFrame({"Portfolio Value": self.portfolio_values})
        df.to_csv(save_path)

    def get_profitlosses(self, save_path=None):
        save_path = utils.SAVE_DIR + "/Metrics" + "/Profitloss_train"\
            if save_path is None else save_path
        df = pd.DataFrame({"Profitloss": self.profitlosses})
        df.to_csv(save_path)

    def get_daily_returns(self, save_path=None):
        save_path = utils.SAVE_DIR + "/Metrics" + "/Daily Return_train"\
            if save_path is None else save_path

        for i in range(len(self.portfolio_values)-1):
            t1_step_pv = self.portfolio_values[i]
            t2_step_pv = self.portfolio_values[i+1]
            daily_return = (t2_step_pv - t1_step_pv)/t1_step_pv
            self.daily_returns.append(daily_return)
        df = pd.DataFrame({"Daily Return": self.daily_returns})
        df.to_csv(save_path)

    def get_total_return(self):
        self.total_return = \
            (self.portfolio_values[-1]-self.portfolio_values[0])/self.portfolio_values[0]
        return self.total_return

    def get_volatility(self):
        daily_returns = self.daily_returns
        return np.std(daily_returns)

