# notes from 
# https://cs231n.github.io/optimization-2/
# https://cs231n.github.io/neural-networks-1/
import numpy as np
from entropy import BCE

def sigmoid(x):
    return 1 / (1+np.e**(-x))
def accuracy(y,y_pred): 
    acc = (y_pred==y).sum()
    acc /= len(y_pred)
    return acc

def return_labels(z):
    ### given output of sigmoid classifier, convert to 0 or 1 label
    pass

class Neuron(object):
    def __init__(self,n_in,n_out):
        self.weights = np.random.normal(size=(n_in,n_out))
        self.bias = np.random.normal(size=(1,n_out))
        self.waits = np.vstack([self.weights,self.bias])
        self.dW = np.zeros_like(self.waits)

    def forward(self, inputs):
        dummy = np.ones((inputs.shape[0],1))
        # numerical stabilizer of sigmoid compute
        inputs -= np.max(inputs)
        self.design = np.column_stack([inputs,dummy])
        self.z = sigmoid(self.design.dot(self.waits))
        return self.z

    def backward(self,dD):
        dZ = dD*self.z*(1-self.z)
        self.dW += self.design.T@dZ
        return self.dW
    def update(self, lr=1e-2):
        # gradient descent update
        self.waits -= lr*self.dW
    def zero_grad_(self):
        # zero gradients like pytorch
        self.dW=np.zeros_like(self.dW)

    def loss(self,y):
        loss = 0
        y_hat = self.z
        loss += BCE(y,y_hat)
        return loss

n = 10 # number of data points
p = 3  # number of features
c = 2   # number of classes
X = np.random.normal(size=(n,p)) # synthetic data
y = np.random.randint(0,c,size=(n,1)) # binaru class
model = Neuron(p,c) #classifier neuron

### gradient descent
num_epochs = 200
a = range(num_epochs)
inits = 0.
loss = 0.
lr = 0.01
loss_dict = {}
loss_dict = loss_dict.fromkeys(a,inits)
for epoch in range(num_epochs):
    acc = 0.
    forward = model.forward(X)
    dL = forward - y
    backward = model.backward(dL)
    loss = model.loss(y)
    loss_dict[epoch]+=loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch} loss {loss}")
        ### implement accuracy
        print(f"Accuracy {acc}")
    model.update(lr=lr)
    model.zero_grad_()


#import matplotlib.pyplot as plt
#
#for k in loss_dict:
#    losses = loss_dict[k]
#    plt.scatter([k] , losses,alpha=0.3)
#plt.savefig('graph.png')
