import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

#### defining the noise schedule
timesteps = 1000
beta_start = 0.0001
beta_end = 0.02

from embedding_1 import TimeBlock, sinusoidal_embedding

class DiffUNet(nn.Module):
    ## modified UNet_1 code in autoencoder.py
    def __init__(self):
        super().__init__()
        self.time_embedder = sinusoidal_embedding(dim=256)
        self.down_block_1 = TimeBlock(in_channels=3,out_channels=16,time_emb_dim=256)
        self.down_block_2 = TimeBlock(in_channels=16,out_channels=32,time_emb_dim=256)


        self.up_block_1 = TimeBlock(in_channels=48,out_channels=16,time_emb_dim=256)
        self.up_block_2 = TimeBlock(in_channels=16,out_channels=16,time_emb_dim=256)
        ### upsample

        self.up_block_3 = nn.Sequential(
            nn.ConvTranspose2d(16,3,kernel_size=3,padding=1),
            #    nn.Sigmoid()
            # sigmoid removed for raw pixel values
        )
        self.blender = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        ### downsampling
        self.pool = nn.MaxPool2d(2)
        ### upsampling 
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
    def forward(self, x, t,debug=False):
        # time embedder
        time_emb = self.time_embedder(t)
        # skip connection features
        skip_1 = self.down_block_1(x,time_emb)
        pooled = self.pool(skip_1)
        bottleneck = self.down_block_2(pooled,time_emb)
        ### upsample
        #        print(f"bottleneck shape {bottleneck.shape}")
        upsampled = self.upsample(bottleneck)
        #        print(f"upsample shape {upsampled.shape}")
        concat = torch.cat((upsampled,skip_1),dim=1)
        ### blending the skip
        blended = self.up_block_1(concat,time_emb)
        reconstruct = self.up_block_2(blended,time_emb)
        #        print(f"reconstruct shape {reconstruct.shape}")
        final_image = self.up_block_3(reconstruct)
        #        print(f"final_image shape {final_image.shape}")
        if debug:
            return {
                "Input": x,
                "Skip 1(16ch)": skip_1,
                "Bottleneck(32ch)": bottleneck,
                "Upsampled (32ch)": upsampled,
                "Blended (16ch)": blended,
                "Final Output": final_image
            }
        return final_image


### debug visualization
model = DiffUNet().to(device)


### load data to visualize
transform = transforms.Compose([
    # transform image to [0,1]
    transforms.ToTensor(),
    # transform [0,1] to [-1,1]
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
dataset = torchvision.datasets.CIFAR10(root="data/",train=True,download=False,transform=transform)
train_loader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)

#model(images,t)
## loss and optim
### defining the noise schedule
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

#### training with noise schedule
import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 100
print(f"Training on {device}")

for epoch in range(epochs):
    running_loss = 0.
    for i, (images, _) in enumerate(train_loader,0):
        images = images.to(device)
        batch_size = images.shape[0]
        ### adding time at each batch member
        t = torch.randint(0,timesteps,(batch_size,),device=images.device).long()
        ### making noise
        noise = torch.randn_like(images)
        x_t=q_sample(images,t,noise=noise)

        optimizer.zero_grad()
        #forward
        outputs = model(x_t,t)
        # loss
        loss = criterion(outputs, noise)
        loss.backward()
        # update model
        optimizer.step()
        running_loss += loss.item()
        if i%200 == 199:
            print(f'[Epoch {epoch+1}, Batch {i + 1:4d}] MSE Loss: {running_loss / 200:.4f}')
            running_loss =0.0
    print('Finished!')

torch.save(model.state_dict(),"model_weights/diff_3_100.pt")
## reverse process
model = DiffUNet().to(device)
print("Loading model")
model.load_state_dict(torch.load("model_weights/diff_3_100.pt",weights_only=True))
print(model)
#dataset.data
images, labels = next(iter(train_loader))
model.eval()
@torch.no_grad()
def sample(model,image_size=32,batch_size=8,channels=3):
    img=torch.randn((batch_size,channels,image_size,image_size),device=device)

    ### reverse time steps
    for i in reversed(range(timesteps)):
        t = torch.full((batch_size,), i,device=img.device,dtype=torch.long)
        predicted_noise=model(img,t)
        # noise schedule for the time step
        alpha_t = extract(alphas,t,img.shape)
        alpha_bar_t = extract(alphas_prod,t,img.shape)
        beta_t = extract(betas,t,img.shape)

        ### denoised image
        mean = (1. /torch.sqrt(alpha_t)) *(
            img - ((1. - alpha_t) / torch.sqrt(1.-alpha_bar_t)) * predicted_noise
        )
        if i > 0:
            noise = torch.randn_like(img)
            sigma_t = torch.sqrt(beta_t)
            img = mean + sigma_t*noise
        else:
            img = mean
    return img

img = sample(model,32,64,3)
#print(img.shape)
#test_image = img[0:5,]
##test_image = test_image.permute(1,2,0).cpu().numpy()
#print(f"shape {test_image.shape}")
##### plot image 
#fig, axes = plt.subplots(2, 3, figsize=(15,3))
#for i, img in enumerate(test_image):
#    # permute the (C,H,W) torch to (H,C,C) for matplotlib
#    img = img.permute(1,2,0)
#    # change back to [0,1] range
#    img = (img + 1) / 2
#    img = torch.clamp(img,0,1)
#    axes[i].imshow(img)
#    axes[i].set_title(f"Step Generated Image {i}")
#    axes[i].axis('off')
#    fig.suptitle(f'Reverse Diffusion after {epochs} epochs ')
#
#plt.savefig('forward_diff_10.png',bbox_inches='tight',dpi=300)
#print('Image saved')


# Generate the batch

# Grab the first 5 images
test_images = img[0:5] 

fig, axes = plt.subplots(1, 5, figsize=(9, 5))

# Loop through our 5 images and our 5 axes simultaneously
for i, img_tensor in enumerate(test_images):
    # Rearrange channels for Matplotlib (C, H, W) -> (H, W, C)
    img_format = img_tensor.permute(1, 2, 0)
    
    # Un-normalize from [-1, 1] back to [0, 1]
    img_format = (img_format + 1) / 2
    img_format = torch.clamp(img_format, 0, 1)
    
    # Convert safely to NumPy for Matplotlib
    img_final = img_format.cpu().numpy()
    
    # Plot it!
    axes[i].imshow(img_final)
    axes[i].axis('off') # Hides the messy coordinates
    fig.suptitle(f'Reverse Diffusion after {epochs} epochs ')
plt.tight_layout
plt.savefig('images/reverse_diff_100.png',bbox_inches='tight',dpi=150)
plt.show()
