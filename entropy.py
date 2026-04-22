#https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html
import torch
from torch.nn import NLLLoss

def nansum(x):
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

print(self_information(4))

def entropy(p):
    entropy = -p*torch.log2(p)
    out = nansum(entropy)
    return out

print(entropy(torch.tensor([0.1,0.1,0.1,0.7])))

def joint_entropy(p_xy):
    joint_ent = -p_xy*torch.log2(p_xy)
    out = nansum(joint_ent)
    return out

print(joint_entropy(torch.tensor([[0.1,0.5],[0.1,0.3]])))

def conditional_entropy(p_xy,p_x):
    p_y_given_x = p_xy / p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    out = nansum(cond_ent)
    return out

print(conditional_entropy(torch.tensor([[0.1,0.5],[0.2,0.2]]),
                    torch.tensor([0.2,0.8])))

### mutual information
def mutual_information(p_xy,p_x,p_y):
    p = p_xy / (p_x*p_y)
    mutual = p_xy * torch.log2(p)
    out = nansum(mutual)
    return out


print(mutual_information(torch.tensor([[0.1,0.5],[0.1,0.3]]),
                   torch.tensor([0.2,0.8]),
                   torch.tensor([0.75,0.25])))

def kl_divergence(p,q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()

p = torch.randint(1,10,size=(2,2))
q = torch.randint(1,10,size=(2,2))

print(f" p:{p} q:{q} \n{kl_divergence(p,q)}")

p = torch.tensor([[1,2],[3,4]])


### cross entropy

def cross_entropy(y_hat,y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()

labels = torch.tensor([0,2])
preds = torch.tensor([[0.3,0.6,0.1],[0.2,0.3,0.5]])
print(cross_entropy(preds,labels))
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds),labels)
print(loss)



