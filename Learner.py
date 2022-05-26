import torch
import Visualizer
import numpy as np

from Environment import environment
from Agent import agent
from ReplayMemory import ReplayMemory
from Network import Actor
from Network import Critic
from Network import Score
from Metrics import Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DIRILearner:
    def __init__(self,
                 lr=1e-4,
                 tau = 0.005, delta=0.05,
                 discount_factor=0.9,
                 batch_size=30, memory_size=100,
                 chart_data=None, K=None, cost=0.0025,
                 min_trading_price=None, max_trading_price=None):

        assert min_trading_price >= 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price

        self.environment = environment(chart_data)
        self.memory = ReplayMemory(max_size=memory_size)
        self.chart_data = chart_data
        self.batch_size = batch_size

        self.score_net_actor = Score().to(device)
        self.score_net_critic = Score().to(device)
        self.actor = Actor(score_net=self.score_net_actor).to(device)
        self.critic = Critic(score_net=self.score_net_critic, header_dim=K).to(device)
        self.critic_target = Critic(score_net=self.score_net_critic, header_dim=K).to(device)

        self.lr = lr
        self.tau = tau
        self.K = K
        self.delta = delta
        self.cost = cost
        self.discount_factor = discount_factor
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        self.agent = agent(environment=self.environment,
                           critic=self.critic, lr=self.lr,
                           critic_target=self.critic_target,
                           actor=self.actor, K=self.K, cost=self.cost,
                           tau=self.tau, delta=self.delta,
                           discount_factor=self.discount_factor,
                           min_trading_price=min_trading_price,
                           max_trading_price=max_trading_price)

    def reset(self):
        self.environment.reset()
        self.agent.reset()

    @staticmethod
    def prepare_training_inputs(sampled_exps):
        states1 = []
        states2 = []
        actions = []
        rewards = []
        next_states1 = []
        next_states2 = []
        log_probs = []
        dones = []

        for sampled_exp in sampled_exps:
            states1.append(sampled_exp[0])
            states2.append(sampled_exp[1])
            actions.append(sampled_exp[2])
            rewards.append(sampled_exp[3])
            next_states1.append(sampled_exp[4])
            next_states2.append(sampled_exp[5])
            log_probs.append(sampled_exp[6])
            dones.append(sampled_exp[7])

        states1 = torch.cat(states1, dim=0).float()
        states2 = torch.cat(states2, dim=0).float()
        actions = torch.cat(actions, dim=0).float()
        rewards = torch.cat(rewards, dim=0).float()
        next_states1 = torch.cat(next_states1, dim=0).float()
        next_states2 = torch.cat(next_states2, dim=0).float()
        log_probs = torch.cat(log_probs, dim=0).float()
        dones = torch.cat(dones, dim=0).float()
        return states1, states2, actions, rewards, next_states1, next_states2, log_probs, dones


    def run(self, num_episode=None, balance=None):
        self.agent.set_balance(balance)
        metrics = Metrics()
        steps_done = 0

        for episode in range(num_episode):
            self.reset()
            cum_r = 0
            state1 = self.environment.observe()
            portfolio = self.agent.portfolio
            while True:
                action, confidence, log_prob = \
                    self.agent.get_action(torch.tensor(state1, device=device).float().view(1,self.K,-1),
                                          torch.tensor(portfolio, device=device).float().view(1,self.K+1,-1))

                m_action, next_state1, next_portfolio, reward, done = self.agent.step(action, confidence)
                steps_done += 1

                experience = (torch.tensor(state1, device=device).float().view(1,self.K,-1),
                              torch.tensor(portfolio, device=device).float().view(1,self.K+1,-1),
                              torch.tensor(m_action, device=device).float().view(1,-1),
                              torch.tensor(reward, device=device).float().view(1,-1),
                              torch.tensor(next_state1, device=device).float().view(1,self.K,-1),
                              torch.tensor(next_portfolio, device=device).float().view(1,self.K+1,-1),
                              torch.tensor(log_prob, device=device).float().view(1,-1),
                              torch.tensor(done, device=device).float().view(1,-1))

                self.memory.push(experience)
                cum_r += reward
                state1 = next_state1
                portfolio = next_portfolio

                if done:
                    break

                if steps_done % 300 == 0:
                    np.set_printoptions(precision=4, suppress=True)
                    value = self.agent.critic(torch.tensor(state1, device=device).float().view(1,self.K,-1),
                                              torch.tensor(portfolio, device=device).float().view(1,self.K+1,-1)).cpu().detach().numpy()[0]

                    alpha = self.agent.actor(torch.tensor(state1, device=device).float().view(1,self.K,-1),
                                             torch.tensor(portfolio, device=device).float().view(1,self.K+1,-1)).cpu().detach()[0]
                    a = action
                    al = torch.cat([torch.tensor([2.0]), alpha], dim=-1).numpy()
                    p = self.agent.portfolio
                    pv = self.agent.portfolio_value
                    sv = self.agent.portfolio_value_static
                    cum_fee = self.agent.cum_fee
                    stocks = self.agent.num_stocks
                    balance = self.agent.balance
                    change = self.agent.change
                    pi_vector = self.agent.pi_operator(change)
                    profitloss = self.agent.profitloss
                    loss = self.agent.loss
                    tex = self.agent.TRADING_TEX
                    charge = self.agent.TRADING_CHARGE
                    print(f"episode:{episode} ========================================================================")
                    print(f"price:{self.environment.get_price()}")
                    print(f"value:{value}")
                    print(f"action:{a}")
                    print(f"maction:{m_action}")
                    print(f"gap:{a-m_action}")
                    print(f"stocks:{stocks}")
                    print(f"cum_fee:{cum_fee}")
                    print(f"alpha:{al}")
                    print(f"portfolio:{p}")
                    print(f"pi_vector:{pi_vector}")
                    print(f"portfolio value:{pv}")
                    print(f"static value:{sv}")
                    print(f"balance:{balance}")
                    print(f"cum reward:{cum_r}")
                    print(f"profitloss:{profitloss}")
                    print(f"tex:{tex}")
                    print(f"charge:{charge}")
                    print(f"loss:{loss}")

                # 학습
                if len(self.memory) >= self.batch_size:
                    sampled_exps = self.memory.sample(self.batch_size)
                    sampled_exps = self.prepare_training_inputs(sampled_exps)
                    self.agent.update(*sampled_exps)
                    self.agent.soft_target_update(self.agent.critic.parameters(), self.agent.critic_target.parameters())

                #metrics 마지막 episode에 대해서만
                if episode == range(num_episode)[-1]:
                    metrics.portfolio_values.append(self.agent.portfolio_value)
                    metrics.profitlosses.append(self.agent.profitloss)
                    metrics.cum_fees.append(self.agent.cum_fee)

            #시각화 마지막 episode에 대해서만
            if episode == range(num_episode)[-1]:
                #metric 계산과 저장
                metrics.get_profitlosses()
                metrics.get_portfolio_values()
                metrics.get_fees()

                #계산한 metric 시각화와 저장
                Visualizer.get_portfolio_value_curve(metrics.portfolio_values)
                Visualizer.get_profitloss_curve(metrics.profitlosses)

    def save_model(self, critic_path, actor_path, score_net_path):
        torch.save(self.agent.critic.state_dict(), critic_path)
        torch.save(self.agent.actor.state_dict(), actor_path)
        torch.save(self.agent.actor.score_net.state_dict(), score_net_path)