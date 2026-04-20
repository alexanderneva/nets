# sinusoidal embeddings 
# https://arxiv.org/pdf/1706.03762
import torch
import math
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

time = torch.tensor([50,250,500,999])
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
embeddings = model(time)
print(embeddings)
print(embeddings.shape)

### plotting embeddings
import matplotlib.pyplot as plt
embeddings_np = embeddings.numpy()
plt.figure(figsize=(12,6))
for i,t in enumerate(time):
    plt.plot(embeddings_np[i], label =f"t = {t.item()}",alpha=0.8)
plt.title("Sinusoidal embeddings on 256 dimensions")
plt.xlabel("embedding dimension")
plt.ylabel("Between -1. 1.")
plt.legend(loc="upper right")
plt.grid(True,alpha=0.3)
plt.show()
plt.savefig("embeddings_squish.png",dpi=300)

fig, axs = plt.subplots(nrows=4,ncols=1,figsize=(12,10),sharex=True,sharey=True)
colours = ["red","blue","green","orange"]
for i, (t, ax) in enumerate(zip(time,axs)):
    ax.plot(embeddings_np[i],color=colours[i],alpha=0.9)
    ax.set_title(f"Time step t = {t.item()}",fontsize=11,loc='left')
    ax.grid(True,alpha=0.3)
    ax.set_ylabel("Value")
axs[-1].set_xlabel(f"Embedding dim {embeddings.dim}",fontsize=11)
plt.tight_layout()
plt.show()
plt.savefig('embeddings_stacked.png',dpi=300)


