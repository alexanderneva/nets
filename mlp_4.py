# notes from 
# https://cs231n.github.io/optimization-2/
# https://cs231n.github.io/neural-networks-1/
import numpy as np

def sigmoid(x):
    return 1 / (1+np.e**(-x))

class Neuron(object):
    def __init__(self,n_in,n_out,X):
        self.weights = np.random.normal(size=(n_in,n_out))
        self.bias = np.random.normal(size=(1,n_out))
        self.waits = np.vstack([self.weights,self.bias])
        self.dW = np.zeros_like(self.waits)

    def forward(self, inputs):
        dummy = np.ones((inputs.shape[0],1))
        self.design = np.column_stack([inputs,dummy])
        self.z = sigmoid(self.design.dot(self.waits))
        return self.z

    def backward(self,dD):
        dZ = dD*self.z*(1-self.z)
        self.dW += self.design.T@dZ
        return self.dW
    def update(self, lr):
        # gradient descent update
        self.waits -= lr*self.dW
    def zero_grad_(self):
        # zero gradients like pytorch
        self.dW=np.zeros_like(self.dW)

#n = 10 # number of data points
#p = 3  # number of features
#c = 2   # number of classes
#X = np.random.normal(size=(n,p))
#y = np.random.randint(0,c,size=(n,1))
#d_ = np.ones((n,c))
#model = Neuron(p,c,X)
#print("data shape",X.shape)
#print("Forward shape",model.forward(X).shape)
#print("Backward shape", model.backward(d_).shape)


