# https://d2l.ai/chapter_convolutional-neural-networks/lenet.html
# vanishing gradient
import torch
from torch import nn

# Set device to 'cuda' if available, otherwise 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


#def init_cnn(module):
#    """Xavier initialization"""
#    if type(module)==nn.Linear or type(module) == nn.Conv2d:
#        nn.init.xavier_uniform_(module.weight)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.LazyConv2d(16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(10),
        )

#    def layer_summary(self, X_shape):
#        X = torch.randn(*X_shape)
#        for layer in self.net:
#            X = X.to(device)
#            X = layer(X)
#            print(layer.__class__.__name__,'output shape:\t',X.shape)
    def forward(self,x):
        return self.net(x)




model = LeNet().to(device)
#model.layer_summary((1,1,28,28))
print(model)


# get data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Load MNIST


## normalizing
#raw_data = train_data.data
#print(raw_data.shape)
##mean = raw_data.mean() / 255.
##std = raw_data.std(axis=0) / 255.
## normalizing tranformation
#transform = transforms.Compose([
#                               transforms.ToTensor(),
#                               transforms.Normalize(mean=mean, std=std)
#])
#train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=32, shuffle=True,num_workers=0)
test_loader = DataLoader(test_data, batch_size=32,num_workers=0)

loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.MSELoss()
test_im = torch.randn(1,1,28,28)
test_im = test_im.to(device)
print(model(test_im).argmax(1))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

images, labels = next(iter(train_loader))

images, labels = images.to(device), labels.to(device)
for i in range(100): 
    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs,labels)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f'Step {i}, Loss: {loss.item():.4f}')

def train(dataloader, model, loss_fn, optimizer):
    # training the model
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred,y)

        #Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fun):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fun(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:8f} \n")

epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done")
