import torch
import numpy as np
from torch.distributions.normal import Normal
import torch.optim as optim
from nde.flows import autoregressive as ar
import matplotlib.pyplot as plt

# flow = realnvp.SimpleRealNVP(
#     features=2,
#     hidden_features=20,
#     num_layers=8,
#     num_blocks_per_layer=2,
# )
#
flow = ar.MaskedAutoregressiveFlow(
            features=2,
            hidden_features=20,
            num_layers=8,
            num_blocks_per_layer=2,
        )

device = torch.device("cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
print("==run on=> ", device)

centre = 0.0
level = 10
xlimits = [0, 30]
ylimits = [0, 30]
myNorm = Normal(torch.tensor([centre]), torch.tensor([1.0]))
def logf(x):
    return myNorm.log_prob(x[:,0])  + torch.log(x[:,0]>level)\
             + torch.log(x[:,1]>0) + torch.log(x[:,1]<1)

def logf(x):
      return -x[:, 0] -x[:, 1] + torch.log(x[:,0] + x[:,1] > level) \
             + torch.log(x[:,0]>0) + torch.log(x[:,1]>0)

def survey_sample(n, centre, rho = 1):
    x = np.random.uniform(size=n, low =0, high=level*1.5)
    y = np.random.uniform(size=n, low =0, high=level*1.5)
    return torch.FloatTensor(np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1))

def filterInputs(inputs):
    return inputs[((inputs[:,0]>xlimits[0]) * (inputs[:,0]<xlimits[1]))
                  * ((inputs[:,1]>ylimits[0]) * (inputs[:,1]<ylimits[1]))]

def calLoss(inputs, log_prob):
    return - ((logf(inputs) - log_prob).exp() * log_prob).mean()

def plotHistory(history, level=10):
    history = np.array(history)
    # tmp  = np.where(history > level)
    idx =  0
    history =  history[idx+1:]
    plt.plot(history)
    plt.show()

optimizer_servy = optim.Adam(flow.parameters(), lr=1e-2)
history = []
for epoch in range(100): #tqdm.notebook.tqdm(, desc='Survey', leave=False):
    with torch.no_grad():
        inputs = survey_sample(1000, level + 0.2)
        # inputs = survey_sample(1000, 1, 5)
        inputs  = filterInputs(inputs)
        # print(inputs.shape)
    log_prob = flow.log_prob(inputs)
    optimizer_servy.zero_grad()
    loss = calLoss(inputs, log_prob)
    loss.backward()
    if torch.isnan(loss):
        break
    optimizer_servy.step()
    history.append(loss.item())


print("===>", inputs[:,0].mean(), loss.item())
optimizer_refine = optim.Adam(flow.parameters(), lr=5e-4)
for epoch in range(500): #tqdm.notebook.tqdm(, desc='Refine', leave=False):
    if torch.isnan(loss):
        break
    # print(epoch)
    with torch.no_grad():
        inputs = flow.sample(1000).detach()
        inputs  = filterInputs(inputs)
        if(epoch % 50 == 49):
            print(inputs[:,0].mean(), loss.item())
    log_prob = flow.log_prob(inputs)
    optimizer_refine.zero_grad()
    loss = calLoss(inputs, log_prob)

    loss.backward()
    optimizer_refine.step()
    history.append(loss.item())

# plotHistory(history, 5)



def calIntegral(needprint=False):
    if torch.isnan(loss):
        print("Nan!")
        return
    with torch.no_grad():
        x, loggx = flow.sample_and_log_prob(10000)
        s0, s1 = x[:,0], x[:,1]

        intgral = torch.exp(logf(x) - loggx).mean()

        if(needprint):
            plt.scatter(s0, s1, marker='o', alpha=0.05)
            plt.plot(0, 0, 'rp', markersize=5)
            plt.xlim([-5, 20])
            plt.ylim([-5, 20])
            plt.show()
        return intgral


v = [calIntegral(i==0) for i in range(100)]
real = 0.0004994
print("mean: %.10f, std: %.10f, accuracy: %.3f%%" % (np.mean(v), np.std(v),
                                              (1-np.abs(np.mean(v) - real)/real)*100))

