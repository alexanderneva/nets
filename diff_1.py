import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

#### defining the noise schedule
timesteps = 1000
beta_start = 0.0001
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, timesteps)
alphas = 1. - betas
alphas_prod = alphas.cumprod(dim=0)
sqrt_alphas_cp = torch.sqrt(alphas_prod)
sqrt_minus_alphas = torch.sqrt(1. - alphas_prod)
#print(torch.cumprod(alphas))

def extract(a, t, x_shape):
    """extracting values and reshaping"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1 ))).to(t.device)

def q_sample(x_0, t, noise=None):
    """noise adder to x0 over time t"""
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t =extract(sqrt_alphas_cp,t,x_0.shape)
    sqrt_minus_alpha_bar_t = extract(sqrt_minus_alphas, t, x_0.shape)
    x_t = sqrt_alpha_bar_t*x_0 + sqrt_minus_alpha_bar_t*noise
    return x_t

### load data to visualize
transform = transforms.Compose([
    # transform image to [0,1]
    transforms.ToTensor(),
    # transform [0,1] to [-1,1]
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = torchvision.datasets.CIFAR10(root="data/",train=True,download=False,transform=transform)
image_no = 37
image, label = dataset[image_no]
x_0 = image.unsqueeze(0)
print(f"Returning the batch dimension {x_0.shape}")

### noising the image
noisy_images = []
timesteps = [0,50,100,500,999]

for step in timesteps:
    if step == 0:
        # image, batch squeezed
        noisy_images.append(x_0.squeeze(0))
    else:
        # tensor to pass to q_sample
        t = torch.tensor([step])
        # noise shaped like x_0
        noise = torch.randn_like(x_0)
        x_t = q_sample(x_0,t,noise)
        noisy_images.append(x_t.squeeze(0))

import matplotlib.pyplot as plt
#### plot image 
fig, axes = plt.subplots(1, 5, figsize=(15,3))
for i, img, in enumerate(noisy_images):
    # permute the (C,H,W) torch to (H,C,C) for matplotlib
    img = img.permute(1,2,0)
    # change back to [0,1] range
    img = (img + 1) / 2
    img = torch.clamp(img,0,1)
    axes[i].imshow(img)
    axes[i].set_title(f"Step {timesteps[i]}")
    axes[i].axis('off')
    fig.suptitle(f'Forward diffusion on image {image_no}')

plt.savefig('forward_diff_cifar10.png',bbox_inches='tight',dpi=300)
print('Image saved')


print('Loading the autoencoder to train')
from autoencoder_1 import Autoencoder
import torch.optim as optim

model = Autoencoder().to(device)
print(model)

trainloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)

### loss and optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 3
print(f"Training on {device}")

for epoch in range(epochs):
    running_loss = 0.
    for i, data in enumerate(trainloader,0):
        #ignore labels
        inputs, _ = data
        inputs = inputs.to(device)
        optimizer.zero_grad()

        #forward
        outputs = model(inputs)
        # loss
        loss = criterion(outputs, inputs)
        loss.backward()
        # update model
        optimizer.step()
        running_loss += loss.item()
        if i%200 == 199:
            print(f'[Epoch {epoch+1}, Batch {i + 1:4d}] MSE Loss: {running_loss / 200:.4f}')
            running_loss =0.0
    print('Finished!')

