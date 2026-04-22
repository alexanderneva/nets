#https://d2l.ai/chapter_linear-regression/synthetic-regression-data.html
import torch
from torch import nn
import random
class SyntheticData(nn.Module):
    def __init__(self, w, b, noise= 0.01, num_train=1000,num_val=1000,batch_size=32):
        super().__init__()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        n = num_train + num_val
        # design matrix
        self.X = torch.randn(n,len(w))
        noise = torch.randn(n,1) * noise
        self.y = torch.matmul(self.X,w.reshape((-1,1))) + b + noise
    def get_dataloader(self, train=True):
        if train:
            indices = list(range(0,self.num_train))
            random.shuffle(indices)
        else: 
            indices = list(range(self.num_train,self.num_train+self.num_val))
        for i in range(0, len(indices),self.batch_size):
            batch_indices = torch.tensor(indices[i: i+ self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
#    def get_tensorloader(self, tensors, train=True, indices=slice(0,None)):
#        tensors = tuple(a[indices] for a in tensors)
#        dataset = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)
#    def get_dataloader_2(self,train=True):
#        i = slice(0, self.num_train) if train else slice(self.num_train, None)
#        return self.get_tensorloader((self.X, self.y), train, i)

data = SyntheticData(w=torch.tensor([2,-3.4]),b=4.2)
print(f"feature shape {data.X.shape} target y {data.y.shape}")
X, y = next(iter(data.get_dataloader()))
print(f"Batch matrix {X}")
print(f"Batch respone \n{y}")


### Linear regression from scratch
### https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html

class LinearRegression(nn.Module):
    def __init__(self, num_inputs, lr, sigma =0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.lr = lr
        self.sigma = sigma
        self.w = torch.normal(0,sigma, (num_inputs, 1),requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    def forward(self, X):
        return torch.matmul(X, self.w) + self.b
    def loss(self, y_hat, y):
        l = (y_hat - y)**2 / 2
        return l.mean()

model = LinearRegression(X.shape[1],0.01)

