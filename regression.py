#https://d2l.ai/chapter_linear-regression/synthetic-regression-data.html
import inspect
from typing import override
import torch
from torch import nn
import random

class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
class Module(nn.Module,HyperParameters):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters
    def loss(self,y_hat,y):
        raise NotImplementedError
    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network there'
        return self.net(X)
    def training_step(self,batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    def validation_step(self,batch):
        l = self.loss(self(*batch[-1]),batch[-1])
    def configure_optimizers(self):
        raise NotImplementedError
    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

class DataModule(HyperParameters):
    def __init__(self, root='./data',num_workers=4):
        self.save_hyperparameters()
    def get_dataloader(self, train):
        raise NotImplementedError
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    def val_dataloader(self):
        return self.get_dataloader(train=False)
    def get_tensorloader(self, tensors, train, indices=slice(0,None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)

class SyntheticData(DataModule):
    def __init__(self, w, b, noise= 0.01, num_train=1000,num_val=1000,batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        # design matrix
        self.X = torch.randn(n,len(w))
        noise = torch.randn(n,1) * noise
        self.y = torch.matmul(self.X,w.reshape((-1,1))) + b + noise
    def get_dataloader(self, train):
        if train:
            indices = list(range(0,self.num_train))
            random.shuffle(indices)
        else: 
            indices = list(range(self.num_train,self.num_train+self.num_val))
        for i in range(0, len(indices),self.batch_size):
            batch_indices = torch.tensor(indices[i: i+ self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
# https://d2l.ai/chapter_linear-regression/oo-design.html#oo-design-training
    #def train_dataloader(self):
    #    return self.get_dataloader(train=True)
    #def val_dataloader(self):
    #    return self.get_dataloader(train=False)
#    def get_tensorloader(self, tensors, train=True, indices=slice(0,None)):
#        tensors = tuple(a[indices] for a in tensors)
#        dataset = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)
#    def get_dataloader_2(self,train=True):
#        i = slice(0, self.num_train) if train else slice(self.num_train, None) return self.get_tensorloader((self.X, self.y), train, i)

data = SyntheticData(w=torch.tensor([2,-3.4]),b=4.2)
print(f"feature shape {data.X.shape} target y {data.y.shape}")
X, y = next(iter(data.train_dataloader()))
print(f"Batch matrix {X}")
print(f"Batch respone \n{y}")
print(len(X))


### Linear regression from scratch
### https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html

class LinearRegression(Module):
    def __init__(self, num_inputs, lr, sigma =0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0,sigma, (num_inputs, 1),requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    def forward(self, X):
        return torch.matmul(X, self.w) + self.b
    def loss(self, y_hat, y):
        l = (y_hat - y)**2 / 2
        return l.mean()
    def configure_optimizers(self):
        return GD([self.w, self.b], self.lr)

class GD(HyperParameters):
    def __init__(self, params, lr):
        self.save_hyperparameters()
    def step(self):
        for param in self.params:
            param -= self.lr*param.grad
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
        

### data trainer class
class Trainer(HyperParameters):
    def __init__(self,max_epochs, gradient_clip_val=0):
        self.save_hyperparameters()
        self.max_epochs = max_epochs
        #self.gradient_clip_val = gradient_clip_val
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = self.train_dataloader
        self.num_val_batches = (len(list(self.val_dataloader))
                                 if self.val_dataloader is not None else 0)
    def prepare_model(self, model):
        model.trainer = self
        self.model = model
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
    def prepare_batch(self, batch):
        return batch
    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val,self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

model = LinearRegression(X.shape[1],0.01)
trainer = Trainer(max_epochs=3)
trainer.fit(model,data)
with torch.no_grad():
    print(f"error for w: {data.w - model.w.reshape(data.w.shape)}")
    print(f"error for bias: {data.b - model.b}")
#optimizer = GD(model.parameters(),model.lr)
#optimizer.step()


### Lazy https://d2l.ai/chapter_linear-regression/linear-regression-concise.html

class LinearRegression(Module):
    def __init__(self,lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0,0.01)
        self.net.bias.data.fill_(0)
    def forward(self, X):
        self.net(X)
    def loss(self,y_hat,y):
        criterion = nn.MSELoss()
        return criterion(y_hat,y)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),self.lr)
    def get_w_b(self):
        return(self.net.weight.data, self.net.bias.data)

model_2 = LinearRegression(0.01)
model_2.eval()
prediction=model(X)
print(prediction)
print(model_2.loss(prediction,y))

data = SyntheticData(w=torch.tensor([2,-3.4]),b=4.2)
trainer.fit(model_2,data)
w,b = model_2.get_w_b()
print(w)
print(b)
#print(f"Error for w: {data.w - w.reshape(data.w.shape)}")
#print(f'Error for bias: {data.b - b}')
