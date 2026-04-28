# notes from 
# https://cs231n.github.io/optimization-2/
# https://cs231n.github.io/neural-networks-1/
import numpy as np
from entropy import BCE

def sigmoid(x):
    return 1 / (1+np.e**(-x))
def accuracy(y,y_pred): 
    acc = (y==y_pred).mean()
    return acc

def return_labels(z):
    ### given output of sigmoid classifier, convert to 0 or 1 label
    return z.argmax(1)

def relu(x):
    if x > 0:
        return x
    else:
        return 0



class Neuron(object):
    def __init__(self,n_in,n_out):
        self.weights = np.random.normal(size=(n_in,n_out))
        #self.bias = np.random.zeros(size=(1,n_out))
        self.bias = np.zeros((1,n_out))
        self.waits = np.vstack([self.weights,self.bias])
        self.dW = np.zeros_like(self.waits)

    def forward(self, inputs):
        dummy = np.ones((inputs.shape[0],1))
        # numerical stabilizer of sigmoid compute
        self.design = np.column_stack([inputs,dummy])
        #self.design -= np.max(self.design)
        scores = self.design.dot(self.waits)
        scores -= np.max(scores,axis=1)[:,None]
        self.z = sigmoid(scores)
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

    def predict(self, X): 
        dummy = np.ones((X.shape[0],1))
        # numerical stabilizer of sigmoid compute
        design = np.column_stack([X,dummy])
        scores = design.dot(self.waits)
        scores -= np.max(scores,axis=1)[:,None]
        z = sigmoid(scores)
        y_pred = return_labels(z)
        return y_pred
        


n = 5000 # number of data points
p = 10  # number of features
c = 2   # number of classes
X = np.random.normal(size=(n,p)) # synthetic data
y = np.random.randint(0,c,size=(n,1)) # binaru class
# selecting a random batch size of data
batch_size = np.random.randint(0,n)
mask = np.random.choice(batch_size, n, replace=True)
X_batch = X[mask]
y_batch = y[mask]

model = Neuron(p,c) #classifier neuron


### gradient descent
num_epochs = 1000
a = range(num_epochs)
inits = 0.
loss = 0.
lr = 1e-3
#loss_dict = {}
#loss_dict = loss_dict.fromkeys(a,inits)
loss_hist= [] 
for epoch in range(num_epochs):
    acc = 0.
    forward = model.forward(X)
    dL = (forward - y)
    backward = model.backward(dL)
    loss = model.loss(y)
    loss_hist.append(loss)
    #loss_dict[epoch]+=loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch} loss {loss}")
        ### implement accuracy
        acc += accuracy(model.predict(X),y)
        print(f"Accuracy {acc}")
    model.update(lr=lr)
    model.zero_grad_()


import matplotlib.pyplot as plt

plt.plot(loss_hist)
plt.savefig('mlp_loss_graph.png')
#
#for k in loss_dict:
#    losses = loss_dict[k]
#    plt.scatter([k] , losses,alpha=0.3)
#plt.savefig('graph.png')
