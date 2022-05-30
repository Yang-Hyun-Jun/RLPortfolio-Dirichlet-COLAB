import torch
import torch.nn as nn
from Distribution import Dirichlet

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
        # alpha = torch.tensor(0.5) * torch.tanh(alpha) + 0.5
        alpha = torch.exp(alpha)
        return alpha

    def sampling(self, s1_tensor, portfolio, repre=False):
        batch_num = s1_tensor.shape[0]
        cash_alpha = torch.ones(size=(batch_num, 1), device=device) * 1.0
        alpha = torch.cat([cash_alpha, self(s1_tensor, portfolio)], dim=-1)
        dirichlet = Dirichlet(alpha)

        #Dirichlet 분포 mode 계산
        B = alpha.shape[0] #Batch num
        N = alpha.shape[1] #Asset num
        total = torch.sum(alpha, dim=1).view(B, 1)
        vector_1 = torch.ones(size=alpha.shape, device=device)
        vector_N = torch.ones(size=(B, 1), device=device) * N

        #Representative value
        mode = (alpha - vector_1) / (total - vector_N)
        mean = dirichlet.mean

        if repre == "mean":
            sampled_p = mean
        elif repre == "mode":
            sampled_p = mode
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
