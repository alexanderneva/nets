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
            nn.ConvTranspose2d(16,3,kernel_size=3),
            nn.Sigmoid()
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
        upsampled = self.upsample(bottleneck)
        concat = torch.cat((upsampled,skip_1),dim=1)
        ### blending the skip
        blended = self.up_block_1(concat,time_emb)
        reconstruct = self.up_block_2(blended,time_emb)
        final_image = self.up_block_3(reconstruct)
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
train_loader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)
### test image 
image_no = 37
image, label = dataset[image_no]
print(image.shape)
images, labels = next(iter(train_loader))
#x_0 = image.unsqueeze(0)
#print(f"Returning the batch dimension {x_0.shape}")
#x_0.shape
#t = torch.randint(0,1000,size=(64,))
num_samples = 3
images= images.to(device)

stack = images[:num_samples]
timesteps_to_plot = [0,250,500,750,999]
model.eval()
time_predictions = {}
with torch.no_grad():
    for t_val in timesteps_to_plot:
        t = torch.full((num_samples,),250,dtype=torch.long)
        t=t.to(device)
        journey = model(stack,t)
        time_predictions[f"step {t_val}"] = journey

fig, axs = plt.subplots(
    nrows=num_samples,
    ncols=len(timesteps_to_plot),
    figsize=(12,4),
    dpi=150
)
fig.suptitle("UNet over Time",fontsize=14)

for row_idx in range(num_samples):
    for col_idx, (time_label,tensor_stack) in enumerate(time_predictions.items()):
        ax = axs[row_idx,col_idx]

        if row_idx ==0:
            ax.set_title(time_label,fontsize=10)
        ax.axis('off')
        img_data = tensor_stack[row_idx]
        img_rgb = img_data.permute(1,2,0).cpu().numpy()

        img_rgb = (img_rgb-img_rgb.min()) / (img_rgb.max() - img_rgb.min())
        ax.imshow(img_rgb)
plt.tight_layout()
plt.savefig("diff_trace.png")
plt.show()

#fig, axs = plt.subplots(nrows=num_samples,ncols=len(journey),figsize=(12,4))
#fig.suptitle("Tracing a batch stack through the net",fontsize=13)
#
#
#for row_idx in range(num_samples):
#    for col_idx, (name,tensor) in enumerate(journey.items()):
#        ax = axs[row_idx, col_idx]
#
#        if row_idx == 0:
#            ax.set_title(name,fontsize=12)
#            ax.axis('off')
#        img_data = tensor[row_idx]
#
#        if name in ["Input","Final Output"]:
#            img_rgb = img_data.permute(1,2,0).cpu().numpy()
#            img_rgb = (img_rgb-img_rgb.min()) / (img_rgb.max() - img_rgb.min())
#            ax.imshow(img_rgb)
#        else: 
#            feature_map = img_data[0].cpu().numpy()
#            ax.imshow(feature_map, cmap='viridis')
#plt.tight_layout()
#plt.savefig('diff_pass.png',dpi=150)

#model(images,t)
### loss and optim

#import torch.optim as optim
#criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(),lr=0.001)
#
#epochs = 3
#print(f"Training on {device}")
#
#for epoch in range(epochs):
#    running_loss = 0.
#    for i, data in enumerate(train_loader,0):
#        #ignore labels
#        inputs, _ = data
#        inputs = inputs.to(device)
#        optimizer.zero_grad()
#
#        #forward
#        outputs = model(inputs)
#        # loss
#        loss = criterion(outputs, inputs)
#        loss.backward()
#        # update model
#        optimizer.step()
#        running_loss += loss.item()
#        if i%200 == 199:
#            print(f'[Epoch {epoch+1}, Batch {i + 1:4d}] MSE Loss: {running_loss / 200:.4f}')
#            running_loss =0.0
#    print('Finished!')
