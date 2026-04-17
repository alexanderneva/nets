# https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
import torch
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
