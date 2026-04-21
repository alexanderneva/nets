import scipy.stats as stats
import numpy as np
N_sample = 10000
cdf = stats.distributions.norm.cdf
a = np.linspace(-1,1,20)
print(cdf(a))
def make_estimate(N,i,x):
    samples = np.random.uniform(i,x,size=N)
    return samples

# next
for i in np.linspace(-1,10,5):
    print('-'*25)
    theta = (10-i)*np.exp(-make_estimate(N_sample,i,10))
    [print(f'Uniform sample: Theta mean {theta.mean()}, Theta Variance {theta.std()**2}')]

    # sample from exponential

    def make_estimate_expon(N):
        samples = np.random.standard_exponential(size=N)
        return (samples > i) & (samples < 10)
    print(f'Exponential Theta mean {make_estimate_expon(N_sample).mean()}, Theta Variance {make_estimate_expon(N_sample).std()**2}')




### ridge regression
def sigmoid(w):
    return 1 / (1 + np.exp(-w))

Nsamples = 1000
p = 20

X = np.random.normal(size=(Nsamples,p))
print(X)
def ridge_update(X,y,w,eta,lambda_):
    m = X.shape[0]
    grad_w = (1/m)*X.T@(sigmoid(X@w)-y)+lambda_*w
    w_new = w - eta*grad_w
    return w_new

def train_ridge_logistic_regression(X, y, eta, lambda_, num_epochs): 
    p = X.shape[1]
    w_init = np.random.uniform(size=(p,1))
    w = w_init
    for epoch in range(num_epochs):
        w = ridge_update(X,y,w,eta,lambda_)
        if epoch % 10000 == 0:
            print(f"Update {epoch} and \n weights {w}")
    return w
        
y = np.random.randint(0,2,size=(Nsamples,1))
eta = 0.01
lambda_ = 0.1
num_epochs = 10000
weights = train_ridge_logistic_regression(X,y,eta,lambda_,num_epochs)
print(f"final weights {weights}")

zeros = np.zeros(shape=weights.shape)
maxx = np.maximum(weights,zeros)
print(maxx, maxx.shape)







tau = 0.025
def soft_threshold(w,tau):
    zeros = np.zeros(shape=w.shape)
    value = np.abs(w)-tau
    return np.sign(w)*np.maximum(value,zeros)

#print(f"Thresholded weights with tau = {tau} ", soft_threshold(weights,tau))



def lasso_update(X,y,w,eta,lambda_):
    tau = eta*lambda_
    m = X.shape[0]
    grad_w = (1/m)*X.T@(sigmoid(X@w)-y)
    descent = w - eta*grad_w
    prox = soft_threshold(descent,tau)
    return prox

print(f"Lasso update {lasso_update(X,y,weights,eta,lambda_)}")


def train_lasso_logistic_regression(X, y, eta, lambda_, num_epochs): 
    p = X.shape[1]
    w_init = np.random.uniform(size=(p,1))
    w = w_init
    for epoch in range(num_epochs):
        w = lasso_update(X,y,w,eta,lambda_)
        if epoch % 1000 == 0:
            print(f"Update {epoch} and \n weights {w}")
    return w

num_epochs=10000
lambda_ = 0.01
train_lasso_logistic_regression(X,y,eta,lambda_,num_epochs)

#### ADMM

def w_update(X,y,w,z,u,eta,rho):
    m = X.shape[0]
    grad_w = (1/m)*X.T@(sigmoid(X@w)-y)+rho*(w-z+u)
    w_new = w - eta*grad_w
    return w_new

def z_update(w,u,lambda_,rho):
    return soft_threshold((u+w),lambda_/rho)

def train_admm_lasso(X,y,eta,lambda_,rho,num_epochs):
    p = X.shape[1]
    w_init = np.random.uniform(size=(p,1))
    w = w_init
    z = w 
    u = w-z
    for epoch in range(num_epochs):
        w = w_update(X,y,w,z,u,eta,rho)
        z = z_update(w,u,lambda_,rho)
        u = u + (w-z)
        if epoch % 100 == 0:
            print(f"ADMM up epoch {epoch}")
    return z
rho=0.4
print(train_admm_lasso(X,y,eta,lambda_,rho,num_epochs))

### pytorch
###
import torch
from torch import nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
X_ = torch.randn(Nsamples,p)
print(f"Shape of X {X.shape}")

def sigmoid(w):
    return 1/(1+torch.exp(-w))

class Model(nn.Module):
    def __init__(self,num_features,Nsamples):
        super().__init__()
        self.layer=nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features,1,bias=False)
        )
    def forward(self,x):
        return self.layer(x)

X_ = torch.randn(Nsamples,p)
print(f"Shape of X {X.shape}")
y = torch.randn(Nsamples,1)
model = Model(num_features=p,Nsamples=Nsamples)
model.eval()
predictions = Model(p,Nsamples)(X_)
pyro.sample("model",dist.Bernoulli(logits=predictions),obs=y)
weights = pyro.sample("weights",dist.Normal(loc=0,scale=1))
print(weights)

class BayesianModel(PyroModule):
    def __init__(self, num_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](num_features,1,bias=False)
        self.linear.weight = PyroSample(
            dist.Normal(loc=0.,scale=1.).expand([1,num_features]).to_event(2)

        )
    def forward(self, x):
        x = x.view(x.size(0),-1)
        return self.linear(x)

bayesian_model = BayesianModel(num_features=p)

def model(x,y):
    predictions = bayesian_model(x)
    samples = pyro.sample("model",dist.Bernoulli(logits=predictions),obs=y)
    return samples

print(model(X_,y))


