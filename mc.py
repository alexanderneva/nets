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




#X = torch.randn(Nsamples,p)
#print(f"Shape of X {X.shape}")
#
#def sigmoid(w):
#    return 1/(1+torch.exp(-w))
#
#print(f"Shape : {sigmoid(X).shape}")
