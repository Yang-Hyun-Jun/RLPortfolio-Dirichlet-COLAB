import torch
import torch.nn as nn
import numpy as np
import utils

from numpy import dot
from numpy.linalg import norm
from Distribution import Dirichlet
from itertools import product
from DataManager import VaR
from DataManager import expected
from DataManager import variance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Score(nn.Module):
    def __init__(self, state1_dim=5, state2_dim=2, output_dim=1):
        super().__init__()

        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(state1_dim+state2_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Identity()

        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")

    def forward(self, s1, s2):
        x = torch.concat([s1, s2], dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        x = self.out_act(x)
        return x


class Actor(nn.Module):
    def __init__(self, score_net):
        super().__init__()
        self.score_net = score_net

    def forward(self, s1_tensor, portfolio):
        """
        state = (s1_tensor, portfolio)
        s1_tensor: (batch, assets, features)
        """

        for k in range(s1_tensor.shape[1]):
            state2 = torch.cat([portfolio[:,0], portfolio[:,k+1]], dim=-1)
            globals()[f"score{k+1}"] = self.score_net(s1_tensor[:,k,:], state2)

        for j in range(s1_tensor.shape[1]):
            scores = list() if j == 0 else scores
            scores.append(globals()[f"score{j+1}"])

        alpha = torch.cat(scores, dim=-1)
        alpha = torch.exp(alpha)
        return alpha

    def sampling(self, s1_tensor, portfolio, repre=False):
        batch_num = s1_tensor.shape[0]
        cash_alpha = torch.ones(size=(batch_num, 1), device=device) * 1.1
        alpha = torch.cat([cash_alpha, self(s1_tensor, portfolio)], dim=-1)
        dirichlet = Dirichlet(alpha)

        B = alpha.shape[0]  # Batch num
        N = alpha.shape[1]  # Asset num + 1

        #Representative value
        if repre == "mean":
            sampled_p = dirichlet.mean

        elif repre == "mode":
            grid_seed = list(product(range(1, 10), repeat=N-1))
            grid_seed = torch.tensor(grid_seed, device=device).float().view(-1, N-1)
            cash_bias = torch.ones(size=(grid_seed.shape[0], 1), device=device) * 5.0
            grid_seed = torch.cat([cash_bias, grid_seed], dim=-1)
            grid = torch.softmax(grid_seed, dim=-1)

            y = dirichlet.log_prob(grid)
            y = y.detach()

            pseudo_mode = grid[torch.argmax(y)]
            pseudo_mode = pseudo_mode.view(B, -1)
            sampled_p = pseudo_mode

        elif repre == "var":
            samples = dirichlet.sample(sample_shape=[30]).view(-1, N).cpu()
            vars = [VaR(utils.STOCK_LIST, torch.softmax(sample[1:], dim=-1)) for sample in samples]
            var_mean = VaR(utils.STOCK_LIST, torch.softmax(dirichlet.mean[0, 1:].cpu(), dim=-1))
            vars.append(var_mean)

            max_ind = np.argmax(vars)
            min_ind = np.argmin(vars)
            max_por = samples[max_ind] if max_ind < 30 else dirichlet.mean
            min_por = samples[min_ind] if min_ind < 30 else dirichlet.mean
            sampled_p = max_por.to(device)

        elif repre == "random":
            sample = dirichlet.sample(sample_shape=[1]).view(-1, N).cpu()
            sample = sample.to(device)
            sampled_p = sample

        elif repre == "expected":
            samples = dirichlet.sample(sample_shape=[30]).view(-1, N).cpu()
            returns = [expected(utils.STOCK_LIST, torch.softmax(sample[1:], dim=-1)) for sample in samples]
            return_mean = expected(utils.STOCK_LIST, torch.softmax(dirichlet.mean[0, 1:].cpu(), dim=-1))
            returns.append(return_mean)

            max_ind = np.argmax(returns)
            max_por = samples[max_ind] if max_ind < 30 else dirichlet.mean
            sampled_p = max_por.to(device)

        elif repre == "cost":
            now_port = utils.NOW_PORT
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu().numpy()
            fees = [utils.check_fee((now_port - sample)[1:]) for sample in samples]
            fee_mean = utils.check_fee((now_port - dirichlet.mean.cpu().numpy()[0])[1:])
            fees.append(fee_mean)

            min_ind = np.argmin(fees)
            # min_por = samples[min_ind]
            min_por = samples[min_ind] if min_ind < 10000 else dirichlet.mean.cpu().numpy()
            sampled_p = torch.tensor(min_por).to(device)

        elif repre == "cossim":
            """
            코사인 유사도
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu().numpy()
            mean = dirichlet.mean[0].cpu().numpy()
            mean = np.array([0.4, 0.4, 0.1, 0.1])
            sims = [dot(mean, sample)/(norm(mean) * norm(sample)) for sample in samples]

            max_ind = np.argmax(sims)
            max_por = samples[max_ind]
            sampled_p = torch.tensor(max_por).to(device)

        elif repre == "pearsim":
            """
            피어슨 유사도
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu().numpy()
            mean = dirichlet.mean[0].cpu().numpy()
            sims = [np.dot((mean - np.mean(mean)), (sample-np.mean(sample))) \
                    / (norm(mean - np.mean(mean)) * norm(sample - np.mean(sample))) for sample in samples]

            max_ind = np.argmax(sims)
            max_por = samples[max_ind]
            sampled_p = torch.tensor(max_por).to(device)

        elif repre == "cosmix1":
            """
            cos 유사도 + 기대 수익률 high
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu()
            mean = dirichlet.mean[0].cpu().numpy()
            sims = [dot(mean, sample)/(norm(mean) * norm(sample)) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:10]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]

            returns = [expected(utils.STOCK_LIST, torch.softmax(por[1:], dim=-1)) for por in high_por]
            max_ind = np.argmax(returns)
            max_por = high_por[max_ind]
            sampled_p = max_por.to(device)

        elif repre == "cosmix2":
            """
            cos 유사도 + 기대 수익률 low
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu()
            mean = dirichlet.mean[0].cpu().numpy()
            sims = [dot(mean, sample)/(norm(mean) * norm(sample)) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:10]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]

            returns = [expected(utils.STOCK_LIST, torch.softmax(por[1:], dim=-1)) for por in high_por]
            low_ind = np.argmin(returns)
            low_por = high_por[low_ind]
            sampled_p = low_por.to(device)

        elif repre == "cosmix3":
            """
            cos 유사도 + cost
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu()
            mean = dirichlet.mean[0].cpu().numpy()
            sims = [dot(mean, sample)/(norm(mean) * norm(sample)) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:10]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]

            now_port = utils.NOW_PORT
            fees = [utils.check_fee((now_port - high.numpy())[1:]) for high in high_por]

            min_ind = np.argmin(fees)
            min_por = high_por[min_ind]
            sampled_p = min_por.to(device)

        elif repre == "cosmix4":
            """
            cos 유사도 + VaR high
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu()
            mean = dirichlet.mean[0].cpu().numpy()
            sims = [dot(mean, sample)/(norm(mean) * norm(sample)) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:10]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]

            vars = [VaR(utils.STOCK_LIST, torch.softmax(por[1:], dim=-1)) for por in high_por]
            max_ind = np.argmax(vars)
            max_por = high_por[max_ind]
            sampled_p = max_por.to(device)

        elif repre == "cosmix5":
            """
            cos 유사도 + VaR low
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu()
            mean = dirichlet.mean[0].cpu().numpy()
            sims = [dot(mean, sample)/(norm(mean) * norm(sample)) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:10]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]

            vars = [VaR(utils.STOCK_LIST, torch.softmax(por[1:], dim=-1)) for por in high_por]
            min_ind = np.argmin(vars)
            min_por = high_por[min_ind]
            sampled_p = min_por.to(device)

        elif repre == "cosmix6":
            """
            mode + cos 유사도 + 기대 수익률 low
            """
            samples = dirichlet.sample(sample_shape=[2000]).view(-1, N)
            logs = [dirichlet.log_prob(sample).cpu() for sample in samples]
            samples = samples.cpu()

            high = samples[logs.index(max(logs))]
            sims = [dot(high, sample)/(norm(high) * norm(sample)) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:10]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]

            returns = [expected(utils.STOCK_LIST, torch.softmax(por[1:], dim=-1)) for por in high_por]
            max_ind = np.argmax(returns)
            max_por = high_por[max_ind]
            sampled_p = max_por.to(device)

        elif repre == "cosmix7":
            """
            코사인 유사도 + 최소 분산 
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu().numpy()
            mean = dirichlet.mean[0].cpu().numpy()
            sims = [dot(mean, sample)/(norm(mean) * norm(sample)) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:10]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]
            high_por = torch.tensor(high_por)

            returns = [variance(utils.STOCK_LIST, torch.softmax(por[1:], dim=-1)) for por in high_por]
            min_ind = np.argmin(returns)
            min_por = high_por[min_ind]
            sampled_p = min_por.to(device)

        elif repre == "pearmix1":
            """
            피어슨 유사도 + 기대 수익률
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu().numpy()
            mean = dirichlet.mean[0].cpu().numpy()
            sims = [np.dot((mean - np.mean(mean)), (sample - np.mean(sample))) \
                    / (norm(mean - np.mean(mean)) * norm(sample - np.mean(sample))) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:10]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]
            high_por = torch.tensor(high_por)

            returns = [expected(utils.STOCK_LIST, torch.softmax(por[1:], dim=-1)) for por in high_por]
            max_ind = np.argmax(returns)
            max_por = high_por[max_ind]
            sampled_p = max_por.to(device)

        elif repre == "pearmix2":
            """
            피어슨 유사도 + 최소 분산 
            """
            samples = dirichlet.sample(sample_shape=[10000]).view(-1, N).cpu().numpy()
            mean = dirichlet.mean[0].cpu().numpy()
            # sims = [np.dot((mean - np.mean(mean)), (sample - np.mean(sample))) \
            #         / (norm(mean - np.mean(mean)) * norm(sample - np.mean(sample))) for sample in samples]
            sims = [dot(mean, sample)/(norm(mean) * norm(sample)) for sample in samples]
            sims_ = sims.copy()
            sims_.sort(reverse=True)

            high_sim = sims_[:500]
            high_ind = [sims.index(high) for high in high_sim]
            high_por = samples[high_ind]
            high_por = list(high_por)
            # high_por = torch.tensor(high_por)

            returns = [variance(utils.STOCK_LIST, torch.softmax(torch.tensor(por[1:]), dim=-1)) for por in high_por]
            min_ind = np.argmin(returns)
            min_por = high_por[min_ind]
            sampled_p = torch.tensor(min_por).to(device)

        elif repre is False:
            sampled_p = dirichlet.sample([1])[0]

        log_pi = dirichlet.log_prob(sampled_p)
        return sampled_p, log_pi


class Critic(nn.Module):
    def __init__(self, score_net, header_dim=None):
        super().__init__()
        self.score_net = score_net
        self.header = Header(input_dim=header_dim)

    def forward(self, s1_tensor, portfolio):

        for k in range(s1_tensor.shape[1]):
            state2 = torch.cat([portfolio[:,0], portfolio[:,k+1]], dim=-1)
            globals()[f"score{k+1}"] = self.score_net(s1_tensor[:,k,:], state2)

        for j in range(s1_tensor.shape[1]):
            scores = list() if j == 0 else scores
            scores.append(globals()[f"score{j+1}"])

        scores = torch.cat(scores, dim=-1)
        v = self.header(scores)
        return v


class Header(nn.Module):
    def __init__(self, output_dim=1, input_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128 ,64)
        self.layer3 = nn.Linear(64, output_dim)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Identity()

    def forward(self, scores):
        x = self.layer1(scores)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        x = self.out_act(x)
        return x



if __name__ == "__main__":
    root = "/Users/mac/Downloads/alphas.npy"
    K = 3
    s1_tensor = torch.rand(size=(1, K, 5))
    portfolio = torch.rand(size=(1, K+1, 1))

    score_net = Score()
    actor = Actor(score_net)
    critic = Critic(score_net, K)

    batch_num = s1_tensor.shape[0]
    cash_alpha = torch.ones(size=(batch_num, 1), device=device) * 1.0
    alpha = torch.cat([cash_alpha, actor(s1_tensor, portfolio)], dim=-1).detach().view(1,-1)

    D = Dirichlet(alpha)
    samples = D.sample(sample_shape=[10000]).view(-1, K+1).cpu()
    logs = [D.log_prob(sample) for sample in samples]
    high = samples[logs.index(max(logs))]
