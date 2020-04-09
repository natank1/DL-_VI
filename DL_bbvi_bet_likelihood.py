# https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyperparameters
hidden_size =10
b_learning_rate = 3e-3

# Constants
num_steps = 6000
max_episodes = 20
import numpy as np
import torch.distributions.beta as beta_dist

s_size=200
gamma_prior_alpha =torch.distributions.Gamma(torch.tensor([12.0]),torch.tensor([3.]))
gamma_prior_beta =torch.distributions.Gamma(torch.tensor([1.0]),torch.tensor([1.]))

class bbvi_proc(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(bbvi_proc, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
        self.alpha_1 = nn.Linear(num_inputs, hidden_size)
        self.alpha_2 = nn.Linear(hidden_size, 2)
        self.beta_1 = nn.Linear(num_inputs, hidden_size)
        self.beta_2 = nn.Linear(hidden_size, 2)

    def forward(self, state):

        x= self.alpha_1(state)
        x= self.alpha_2(x)
        x =F.softplus(x)
        gam_a= torch.distributions.Gamma(x.data[0][0],x.data[0][1])
        z_alpha =gam_a.sample((s_size,))
        y = self.beta_1(state)
        y = self.beta_2(y)
        y = F.softplus(y)
        gam_b = torch.distributions.Gamma(y.data[0][0], y.data[0][1])
        z_beta = gam_b.sample((s_size,))



        return z_alpha, z_beta,x.data,y.data,gam_a,gam_b

def bbvi(xx):

    bbvi_model = bbvi_proc(1, 2, hidden_size)
    bbvi_optimizer = optim.Adam(bbvi_model.parameters(), lr=b_learning_rate)



    for episode in range(max_episodes):

        bbvi_optimizer.param_groups[0]['lr']= b_learning_rate*(1/(1.+episode)) #robinsmonro
        np.random.shuffle(xx)
        for steps in range(num_steps):
            state =xx[steps]
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            z_alpha, z_beta, x0,y0 ,ga,gb = bbvi_model.forward(state)
            q_alpha= x0.detach().numpy()[0]
            q_beta= y0.detach().numpy()[0]

            alpha_prior = gamma_prior_alpha.log_prob(z_alpha)
            beta_prior  = gamma_prior_beta.log_prob(z_beta)
            lq_alpha =ga.log_prob(z_alpha)
            lq_beta = gb.log_prob(z_beta)

            # print (z_alpha,z_beta)
            bb =torch.distributions.Beta(z_alpha,z_beta)
            like_beta = bb.log_prob(state)
            loss0= (lq_alpha)*(like_beta+alpha_prior+beta_prior -lq_alpha - lq_beta)
            loss0= loss0.mean()
            loss1 = (lq_beta) * (like_beta + alpha_prior + beta_prior - lq_alpha - lq_beta)
            loss1 = loss1.mean()
            loss =loss0
            bbvi_optimizer.zero_grad()

            loss = Variable(loss,requires_grad=True)
            loss.backward()

            bbvi_optimizer.step()
            bbvi_optimizer.zero_grad()
            loss =loss1
            loss = Variable(loss, requires_grad=True)

            loss.backward()
            bbvi_optimizer.zero_grad()

        print (z_alpha.mean(0),z_beta.mean(0))
    return q_alpha,q_beta

if __name__ == "__main__":

    synth_data= np.random.beta(2.,1.,20000)

    xdata= np.expand_dims(synth_data,axis=1)
    q_alpha,q_beta= bbvi(xdata)
    VI_data = np.random.beta(q_alpha[0] / q_alpha[1], q_beta[0] / q_beta[1], 20000)

    #Graphs
    hist, bins = np.histogram(synth_data, bins=400)
    bin_centers = (bins[1:] + bins[:-1]) * 0.5
    plt.plot(bin_centers, hist, '-b', label="synth")

    hist_vi, bin_vi = np.histogram(VI_data, bins=400)
    bin_centers1 = (bin_vi[1:] + bin_vi[:-1]) * 0.5
    plt.plot(bin_centers1, hist_vi, '-r', label="Train")
    plt.legend(loc="upper left")
    plt.title("Beta Comparison-BBVI")
    plt.show('-')

#