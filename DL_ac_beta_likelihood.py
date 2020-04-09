# https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

# hyperparameters
hidden_size =10
learning_rate = 5e-3

# Constants
GAMMA = 0.99
num_steps = 6000
max_episodes = 2000
num_outputs=2
import random
import numpy as np
from collections import deque
import torch.distributions.beta as beta_dist


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
        self.alpha_1 = nn.Linear(num_inputs, hidden_size)
        self.alpha_2 = nn.Linear(hidden_size, num_outputs)
        self.beta_1 = nn.Linear(num_inputs, hidden_size)
        self.beta_2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, state):
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        x= self.alpha_1(state)
        x= self.alpha_2(x)
        x =F.softplus(x)
        gam_a= torch.distributions.Gamma(x.data[0][0],x.data[0][1])
        z_alpha =gam_a.sample((1,))
        y = self.beta_1(state)
        y = self.beta_2(y)
        y = F.softplus(y)
        gam_b = torch.distributions.Gamma(y.data[0][0], y.data[0][1])
        z_beta = gam_b.sample((1,))

        policy_dist = F.softplus(self.alpha_1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, z_alpha, z_beta,policy_dist,x.data,y.data,gam_a,gam_b
gamma_prior_alpha =torch.distributions.Gamma(torch.tensor([10.0]),torch.tensor([3.]))
gamma_prior_beta =torch.distributions.Gamma(torch.tensor([2.0]),torch.tensor([2.]))

def a2c(xx):

    actor_critic = ActorCritic(1, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []


        for steps in range(num_steps):
            state =xx[steps]
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            value, z_alpha, z_beta, policy_dist,x0,y0 ,ga,gb = actor_critic.forward(state)

            # dist = policy_dist.detach().numpy()
            bb =torch.distributions.Beta(z_alpha,z_beta)
            value = value.detach().numpy()[0, 0]


            entropy =ga.entropy()+gb.entropy()
            # uu=torch.exp(ga.log_prob(z_alpha)+gb.log_prob(z_beta))

            values.append(value)
            # log_probs.append(log_prob)
            bbbb =bb.log_prob(state)
            log_probs.append(bbbb)
            rewa_termp= bbbb -entropy + gamma_prior_alpha.log_prob(z_alpha)+gamma_prior_beta.log_prob(z_beta)
            rewa_termp = rewa_termp
            rewards.append(rewa_termp)


            entropy_term += entropy

            if  steps == num_steps - 1:
                # Qval, _ = actor_critic.forward(new_state)
                Qval, _,_,_,_,_,_,_ = actor_critic.forward(state)

                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    sys.stdout.write(
                        "episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards),
                                                                                                  steps,
                                                                                                  average_lengths[-1]))
                break
        print ("yyy ",x0,y0)
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            # Qval = rewards[t] + GAMMA * Qval
            Qval = rewards[t]

            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
        # ac_loss = actor_loss + critic_loss

        ac_loss = Variable(ac_loss,requires_grad=True)
        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # Plot results
    # smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    # smoothed_rewards = [elem for elem in smoothed_rewards]
    return x0,y0

if __name__ == "__main__":
    synth_data = np.random.beta(2., 1., 20000)

    xdata = np.expand_dims(synth_data, axis=1)

    hist, bins = np.histogram(synth_data, bins=400)
    bin_centers = (bins[1:] + bins[:-1]) * 0.5


    x0,y0=a2c(xdata)
    print (x0,y0)
    a= x0.detach().numpy()[0]
    b = y0.detach().numpy()[0]
    vi_data = np.random.beta(a[0] / a[1], b[0] / b[1], 20000)
    hist1, bins1 = np.histogram(vi_data, bins=400)
    bin_centers1 = (bins1[1:] + bins1[:-1]) * 0.5
    plt.plot(bin_centers, hist, '-b', label="synth")
    plt.plot(bin_centers1, hist1, '-r', label="Train")
    plt.legend(loc="upper left")
    plt.title("Beta Comparison-AC")
    plt.show('-')
    print ("ok")
