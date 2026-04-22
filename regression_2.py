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


def prediction(X,w):
    size = X.shape[1]
    return X@w
def MSE(y,y_hat):
    return np.mean((y-y_hat)**2) / 2
def grad_MSE(y,X,w):
    y_hat = prediction(X,w)
    return X.T@(y_hat-y)
print(grad_MSE(y,X,weights))

def training_L(X,y,num_epochs=100,eta=0.01):
    size=X.shape[1]
    weights = np.random.normal(size=(size,1))
    storage = []
    loss_storage = []
    for epoch in range(num_epochs):
        storage.append(weights)
        loss = MSE(y,prediction(X_,weights))
        loss_storage.append(loss)
        print(f"Epoch number {epoch}, weights {weights}")
        weights = weights - eta*grad_MSE(y,X_,weights)
    return storage
    
storage = training_L(X_,y)
storage=np.array(storage)
storage = storage[:,:,-1]

