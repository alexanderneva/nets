import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
def generate_data_log(n,p):
    epsilon = 0.01
    size = (n,p)
    y_size = (n,1)
    x = np.random.normal(size=size)
    y = np.random.randint(0,2,size=y_size)
    return x,y

X, y = generate_data_log(20,2)

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def grad_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def y_hat(X,w):
    return sigmoid(X@w)
def grad_y_hat(X,w):
    z = X@w
    return X.T


def BCE(y,y_hat):
    return np.mean(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat))

def grad_BCE(y,X,w):
    prediction = y_hat(X,w)
    return -X.T@(y - prediction)

weights = np.random.normal(size=(X.shape[1],1))
print(y_hat(X,weights))
def training(X,y,num_epochs=100,eta=0.01):
    size=X.shape[1]
    weights = np.random.normal(size=(size,1))
    storage = []
    for epoch in range(num_epochs):
        storage.append(weights)
        print(f"Epoch number {epoch}, weights {weights}")
        weights = weights - eta*grad_BCE(y,X,weights)
    return storage
    
training(X,y)


### standard lin_reg


def generate_data(n,p):
    epsilon = 0.01
    size = (n,p)
    y_size = (n,1)
    x = np.random.normal(size=size)
    y = np.random.normal(0,2,size=y_size)
    return x,y

n = 20
p = 2
X_, y = generate_data(n,p)


def prediction(X,w,b):
    return X@w+b
def MSE(y,y_hat):
    return np.mean((y-y_hat)**2) / 2
def grad_MSE_b(y,X,w,b):
    return X@w + b - y
def grad_MSE_w(y,X,w,b):
    y_hat = prediction(X,w,b)
    return X.T@(y_hat-y)

def training_L(X,y,num_epochs=100,eta=0.01):
    size=X.shape[1]
    weights = np.random.normal(size=(size,1))
    size_bias = y.shape[0]
    bias = np.random.normal(size=(size_bias,1))
    storage = []
    loss_storage = []
    for epoch in range(num_epochs):
        storage.append(weights)
        loss = MSE(y,prediction(X_,weights,bias))
        loss_storage.append(loss)
        print(f"Epoch number {epoch}, weights {weights}")
        weights = weights - eta*grad_MSE_w(y,X_,weights,bias)
        bias = bias - eta*grad_MSE_b(y,X_,weights,bias)
    return storage, weights, bias
    
storage, weights, bias = training_L(X_,y)
storage=np.array(storage)
storage = storage[:,:,-1]
print(bias)

