# https://github.com/altosaar/variational-autoencoder/blob/master/train_variational_autoencoder_pytorch.py
# https://github.com/altosaar/variational-autoencoder/blob/master/train_variational_autoencoder_pytorch.py
import torch
import torch.utils
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.distributions.beta
import matplotlib.pyplot as plt


gam_al =torch.distributions.Gamma(torch.tensor([80.]),torch.tensor([20.]))
gam_be =torch.distributions.Gamma(torch.tensor([2.]),torch.tensor([2.]))

class VariationalMeanField(nn.Module):
  """Approximate posterior parameterized by an inference network."""

  def __init__(self, input_dim, output_dim):
        super(VariationalMeanField, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mu_fc_0 =nn.Linear(self.input_dim,40)
        self.mu_fc_00 = nn.Linear(40,20)

        self.rel0 =nn.ReLU()
        self.mu_fc_1 = nn.Linear(20,2)
        self.sig_fc_0 = nn.Linear(self.input_dim, 40)
        self.sig_fc_00 = nn.Linear(40, 20)

        self.sig_fc_1 = nn.Linear(20, 2)
        self.el0= nn.Softplus()
        self.nsample =30
  def forward(self, state):

        state =state.float()
        # exit(66)
        x_m = self.mu_fc_0(state)
        x_m = self.mu_fc_00(x_m)


        x_m = self.mu_fc_1(x_m)
        x_m = self.el0(x_m)

        gam_a = torch.distributions.Gamma(x_m[:,0],x_m[:,1])
        zalpha= gam_a.sample((self.nsample,))

        x_s = self.sig_fc_0(state)
        x_s = self.sig_fc_00(x_s)
        x_s = self.sig_fc_1(x_s)
        x_s = self.el0(x_s)

        gam_b = torch.distributions.Gamma(x_s[:,0],x_s[:,1])

        zbeta= gam_b.sample((self.nsample,))

        gamal =gam_a.log_prob(zalpha).mean(0)
        gambe = gam_b.log_prob(zbeta).mean(0)
        alp_prio = gam_al.log_prob(zalpha).mean(0)
        bet_prior = gam_be.log_prob(zbeta).mean(0)
        tmp_score =alp_prio+bet_prior-gamal-gambe
        bb= []
        for j in range(batch_s):
            bb.append( torch.distributions.Beta(zalpha[:,j], zbeta[:,j]).log_prob(state[j]).numpy().mean())

        tmp_score += torch.tensor([bb]).squeeze()
        tmp_score= tmp_score.mean()


        return tmp_score ,x_m,x_s



import torch.distributions.beta as b1
def beta_loss(x,alpha,beta1):
  v =b1.Beta(alpha,beta1)
  return v.log_prob(x)

if __name__ == '__main__':
  device = "cpu"

  learning_rate= 0.01

  var_flg ="mean-field"
  variational = VariationalMeanField(input_dim=1, output_dim=2)

  variational.to(device)

  optimizer = torch.optim.RMSprop(  list(variational.parameters()),
                                  lr=learning_rate,
                                  centered=True)


  x11 =np.random.beta(2.,1.,20000)

  hist ,bins =np.histogram(x11,bins=400)
  bin_centers = (bins[1:] + bins[:-1]) * 0.5
  x22 = Variable(torch.from_numpy(x11))


  best_valid_elbo = -np.inf
  num_no_improvement = 0
  batch_s=8
  bb=[]
  for k in range (30):
    for jj in range (0,20000,batch_s):

      x=x22[jj:(jj+batch_s)].float()
      x=x.unsqueeze(dim=-1)


      variational.zero_grad()
      tmp_score,xm,xs  = variational(x)
      loss = tmp_score
      loss= Variable(loss,requires_grad=True)
      if jj % 1500 == 64:
         print(" uuuuu ", xm.mean(0),xs.mean(0))
         a= xm.mean(0)
         b= xs.mean(0)

      loss.backward()
      optimizer.step()
    a = xm.mean(0)
    b = xs.mean(0)
  a= a.detach().numpy()
  b = b.detach().numpy()

  x33 = np.random.beta(a[0] / a[1], b[0] / b[1], 20000)

  hist1, bins1 = np.histogram(x33, bins=400)
  bin_centers1 = (bins1[1:] + bins1[:-1]) * 0.5
  plt.plot(bin_centers, hist,'-b',label= "synth")
  plt.plot(bin_centers1, hist1,'-r',label="Train")
  plt.legend(loc="upper left")
  plt.title("Beta Comparison")
  plt.show('-')

