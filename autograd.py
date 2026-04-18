# https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
print(f'Model parameters {model}')
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

prediction = model(data)
print(f" Prediction { prediction }")
loss = (prediction - labels).sum()

# sgd with momentum
optim = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)
optim.step()

x = torch.tensor([-1.,3.],requires_grad=True)
y = torch.tensor([6., 7.],requires_grad=True)
z = torch.tensor([-1., 4.],requires_grad=True)
Q = x**2+y**2-z**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
print(x.grad)

# resnet freeze

for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512,10)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)
optim.step()

