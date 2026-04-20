# sinusoidal embeddings 
# https://arxiv.org/pdf/1706.03762
import torch
import math
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class sinusoidal_embedding(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim //2
        emb = math.log(10000) / (half_dim-1)
        emb = torch.exp(torch.arange(half_dim,device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(),emb.cos()),dim=-1)
        return emb

model = sinusoidal_embedding(256)

time = torch.tensor([50,250,500,999])
embeddings = model(time)
print(embeddings)
print(embeddings.shape)

#### plotting embeddings
#import matplotlib.pyplot as plt
#embeddings_np = embeddings.numpy()
#plt.figure(figsize=(12,6))
#for i,t in enumerate(time):
#    plt.plot(embeddings_np[i], label =f"t = {t.item()}",alpha=0.8)
#plt.title("Sinusoidal embeddings on 256 dimensions")
#plt.xlabel("embedding dimension")
#plt.ylabel("Between -1. 1.")
#plt.legend(loc="upper right")
#plt.grid(True,alpha=0.3)
#plt.show()
#plt.savefig("embeddings_squish.png",dpi=300)
#
#fig, axs = plt.subplots(nrows=4,ncols=1,figsize=(12,10),sharex=True,sharey=True)
#colours = ["red","blue","green","orange"]
#for i, (t, ax) in enumerate(zip(time,axs)):
#    ax.plot(embeddings_np[i],color=colours[i],alpha=0.9)
#    ax.set_title(f"Time step t = {t.item()}",fontsize=11,loc='left')
#    ax.grid(True,alpha=0.3)
#    ax.set_ylabel("Value")
#axs[-1].set_xlabel(f"Embedding dim {embeddings.dim}",fontsize=11)
#plt.tight_layout()
#plt.show()
#plt.savefig('embeddings_stacked.png',dpi=300)

### adding the time block
class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        # project time from 256 dimension to channels out
        self.time_mlp = nn.Linear(time_emb_dim,out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
    def forward(self, x, time_emb):
        # first process the image 
        x = x.to(device)
        h = self.conv1(x)
        h = self.relu(h)
        t = self.time_mlp(time_emb)
        t = self.relu(t)
        # to broadcast the time
        t = t.view(t.shape[0],t.shape[1],1,1)
        # time "injection"
        h = h + t
        # decode back to output image size
        out = self.conv2(h)
        #        print(f'time block shape is {out.shape}')
        return out


















