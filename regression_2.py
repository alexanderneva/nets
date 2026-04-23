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

def grad_z(X):
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
    return np.sum(X@w + b - y)
def grad_MSE_w(y,X,w,b):
    y_hat = prediction(X,w,b)
    return X.T@(y_hat-y)

def training_L(X,y,num_epochs=100,eta=0.01):
    size=X.shape[1]
    weights = np.random.normal(size=(size,1))
    bias = np.random.normal(1)*np.ones_like(y)
    storage = []
    loss_storage = []
    for epoch in range(num_epochs):
        storage.append(weights)
        loss = MSE(y,prediction(X_,weights,bias))
        loss_storage.append(loss)
        if epoch % 100 ==0:
            print(f"Epoch number {epoch}, weights {weights}")
        weights = weights - eta*grad_MSE_w(y,X_,weights,bias)
        bias = bias - eta*grad_MSE_b(y,X_,weights,bias)
    return storage, weights, bias
    
storage, weights, bias = training_L(X_,y)
storage=np.array(storage)
storage = storage[:,:,-1]

## visualization in notebook



### logistic regression with bias


def y_hat_b(X,w,b):
    return sigmoid(X@w+b)


def grad_BCE_w(y,X,w,b):
    y_hat = prediction(X,w,b)
    return -X.T@(y - y_hat)
def grad_BCE_b(y,X,w,b):
    y_hat = prediction(X,w,b)
    return np.sum(y_hat-y)


def training_Lb(X,y,num_epochs=100,eta=0.01):
    size=X.shape[1]
    weights = np.random.normal(size=(size,1))
    bias = np.random.normal(1)*np.ones_like(y)
    storage = []
    loss_storage = []
    for epoch in range(num_epochs):
        storage.append(weights)
        loss = BCE(y,prediction(X_,weights,bias))
        loss_storage.append(loss)
        if epoch % 100 ==0:
            print(f"Epoch number {epoch}, weights {weights}")
        weights = weights - eta*grad_BCE_w(y,X_,weights,bias)
        bias = bias - eta*grad_BCE_b(y,X_,weights,bias)
    return storage, weights, bias

storage,weights,bias = training_Lb(X,y,eta=0.01)

print(f"Bias: {bias[0]}")

### hyperbolic tangent activation

def tanh_(x):
    return 2*sigmoid(2*x)-1

def grad_tanh(x):
    return 4*grad_sigmoid(2*x)

def change_target(y):
    return (1 + y) / 2

def grad_MSE_uw(y,X,w,b):
    y_hat = prediction(X,w,b)
    u = 2*y_hat
    return 2*X.T@grad_sigmoid(u)

def grad_MSE_ub(y,X,w,b):
    y_hat = prediction(X,w,b)
    u = 2*y_hat
    return 2*np.sum(grad_sigmoid(u))

w = np.random.normal(size=(2,1))
b = np.random.normal(1)*np.ones_like(y)
print(f"Gradient wrt w {grad_MSE_uw(y,X,w,b)}")
print(f"Gradient wrt b {grad_MSE_ub(y,X,w,b)}")


def training_Lbt(X,y,num_epochs=100,eta=0.01):
    size=X.shape[1]
    weights = np.random.normal(size=(size,1))
    bias = np.random.normal(1)*np.ones_like(y)
    y = change_target(y)
    storage = []
    loss_storage = []
    for epoch in range(num_epochs):
        storage.append(weights)
        loss = MSE(y,tanh_(prediction(X,weights,bias)))
        loss_storage.append(loss)
        if epoch % 100 ==0:
            print(f"Epoch number {epoch}, weights {weights},bias {bias[0]}")
        weights = weights - eta*grad_MSE_uw(y,X,weights,bias)
        bias = bias - eta*grad_MSE_ub(y,X,weights,bias)
    return storage, weights, bias

storage,weights,bias = training_Lbt(X,y,num_epochs=500)

