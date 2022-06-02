import Visualizer
import utils
import torch
import numpy as np

from Metrics import Metrics
from Environment import environment
from Agent import agent
from Network import Actor
from Network import Critic
from Network import Score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DIRITester:
    def __init__(self,
                 test_data, balance,
                 min_trading_price, max_trading_price,
                 delta, K, repre="mean", cost=0.0025, holding=False):

        self.test_data = test_data

        self.state1_dim = 5
        self.state2_dim = 2
        self.K = K

        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0

        self.score_net = Score().to(device)
        self.actor = Actor(self.score_net).to(device)
        self.critic = Critic(self.score_net, header_dim=self.K).to(device)
        self.critic_target = Critic(self.score_net, header_dim=self.K).to(device)

        self.repre = repre
        self.delta = delta
        self.cost = cost
        self.holding = holding
        self.balance = balance
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        self.env = environment(chart_data=test_data)
        self.agent = agent(environment=self.env, cost=self.cost,
                           actor=self.actor, K=self.K, delta=self.delta,
                           critic=self.critic, critic_target=self.critic_target,
                           lr=0.0, tau=0.0, discount_factor=0.0,
                           min_trading_price=self.min_trading_price,
                           max_trading_price=self.max_trading_price)

        actor_path = utils.SAVE_DIR + "/Models" + "/DirichletPortfolio_actor.pth"
        score_path = utils.SAVE_DIR + "/Models" + "/DirichletPortfolio_score.pth"
        self.agent.actor.load_state_dict(torch.load(actor_path))
        self.agent.actor.score_net.load_state_dict(torch.load(score_path))

    def check_frequency(self, action):
        for a in action:
            if a > 0:
                self.num_buy += 1
            elif a < 0:
                self.num_sell += 1
            elif a == 0:
                self.num_hold += 1

    def run(self):
        metrics = Metrics()
        self.agent.set_balance(self.balance)
        self.agent.reset()
        self.agent.environment.reset()
        state1 = self.agent.environment.observe()
        portfolio = self.agent.portfolio
        steps_done = 0

        while True:
            action, confidence, log_prob = \
                self.agent.get_action(torch.tensor(state1, device=device).float().view(1, self.K, -1),
                                      torch.tensor(portfolio, device=device).float().view(1, self.K + 1, -1), self.repre)

            #3일 단위로 거래
            if self.holding:
                if steps_done % 3:
                    action = np.zeros(shape=self.K)

            m_action, next_state1, next_portfolio, reward, done = self.agent.step(action, confidence)
            self.check_frequency(m_action)

            steps_done += 1
            state1 = next_state1
            portfolio = next_portfolio

            metrics.portfolio_values.append(self.agent.portfolio_value)
            metrics.profitlosses.append(self.agent.profitloss)
            metrics.balances.append(self.agent.balance)
            metrics.cum_fees.append(self.agent.cum_fee)
            if steps_done % 50 == 0:
                print(f"balance:{self.agent.balance}")
                print(f"stocks:{self.agent.num_stocks}")
                print(f"actions:{action}")
                print(f"portfolio:{self.agent.portfolio}")
                print(f"cum_fee:{self.agent.cum_fee}")
            if done:
                print(f"model:{self.agent.profitloss}")
                print(f"num buy:{self.num_buy}")
                print(f"num sell:{self.num_sell}")
                print(f"num hold:{self.num_hold}")
                break


        # Benchmark: B&H
        self.agent.set_balance(self.balance)
        self.agent.reset()
        self.agent.environment.reset()
        self.agent.delta = 0.0
        self.agent.environment.observe()
        while True:
            action = np.ones(self.K) / self.K
            confidence = abs(action)
            _, next_state1, next_portfolio, reward, done = self.agent.step(action, confidence)

            metrics.profitlosses_BH.append(self.agent.profitloss)
            if done:
                print(f"B&H{self.agent.profitloss}")
                break

        # metric and visualizing
        Vsave_path2 = utils.SAVE_DIR + "/" + "/Metrics" + "/Portfolio Value Curve_test"
        Vsave_path4 = utils.SAVE_DIR + "/" + "/Metrics" + "/Profitloss Curve_test"
        Msave_path1 = utils.SAVE_DIR + "/" + "/Metrics" + "/Portfolio Value_test"
        Msave_path2 = utils.SAVE_DIR + "/" + "/Metrics" + "/Profitloss_test"
        Msave_path3 = utils.SAVE_DIR + "/" + "/Metrics" + "/Profitloss B&H"
        Msave_path4 = utils.SAVE_DIR + "/" + "/Metrics" + "/Balances"
        Msave_path5 = utils.SAVE_DIR + "/" + "/Metrics" + "/fees_test"

        metrics.get_portfolio_values(save_path=Msave_path1)
        metrics.get_profitlosses(save_path=Msave_path2)
        metrics.get_profitlosses_BH(save_path=Msave_path3)
        metrics.get_balances(save_path=Msave_path4)
        metrics.get_fees(save_path=Msave_path5)

        Visualizer.get_portfolio_value_curve(metrics.portfolio_values, save_path=Vsave_path2)
        Visualizer.get_profitloss_curve(metrics.profitlosses, metrics.profitlosses_BH, save_path=Vsave_path4)