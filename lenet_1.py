#https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
# Set device to 'cuda' if available, otherwise 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # take 1 image, into 6 channels, 5x5 mask
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # Wx + b weights
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        s2 = F.max_pool2d(c1,(2,2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3,2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output

net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())

# random input test
x = torch.randn(1,1,32,32)
out = net(x)
#print(out)
#net.zero_grad()
#out.backward(torch.randn(1,10))

# loss

target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(out,target)
print(loss)
print("A few steps backward")
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# back propogation

net.zero_grad()
print('conv1.bias.grad before')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward()')
print(net.conv1.bias.grad)


# updating the weights

learning_rate = 1e-2
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)

# optimizations

import torch.optim as optim

optimizer = optim.SGD(net.parameters(),lr=learning_rate)
optimizer.zero_grad()
out = net(x)
loss = criterion(out,target)
loss.backward()
optimizer.step()
