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


### random uniform
import torch
a=torch.randint(1,5,size=(10,10))
print(a)

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
num_epochs = 100000
weights = train_ridge_logistic_regression(X,y,eta,lambda_,num_epochs)
print(f"final weights {weights}")






#X = torch.randn(Nsamples,p)
#print(f"Shape of X {X.shape}")
#
#def sigmoid(w):
#    return 1/(1+torch.exp(-w))
#
#print(f"Shape : {sigmoid(X).shape}")
