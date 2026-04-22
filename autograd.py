# https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
#print(f'Model parameters {model}')
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

prediction = model(data)
#print(f" Prediction { prediction }")
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
#print(x.grad)

# resnet freeze

for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512,10)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)
optim.step()

# https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.html
# back prop example
# f(u,v) = (u+v)^2
# u(a,b) = (a+b)^2, v(a,b) = (a-b)^2,
# a(w,x,y,z) = (w+x+y+z)^2, b(w,x,y,z)=(w+x-y-z)^2

w,x,y,z = -1,0,-2,1
a = (w+x+y+z)**2
b = (w+x-y-z)**2
u = (a+b)**2
v = (a-b)**2
f = (u+v)**2
print(f'f at w = {w} x = {x} y = {y} z ={z} is f {f}')
df_du = 2*(u+v)
df_dv = 2*(u+v)
du_da = 2*(a+b)
du_db = 2*(a+b)
dv_da = 2*(a-b)
dv_db = -2*(a-b)
da_dw = 2*(w+x+y+z)
db_dw = 2*(w+x-y-z)

# df/dw = df/du*du/dw + df/dv*dv/dw
# du/dw = du/da*da/dw + du/db*db/dw
# dv/dw = dv/da*da/dw + dv/db*db/dw

dv_dw = dv_da*da_dw + dv_db*db_dw
print(f'dv/dw is {dv_dw}')
du_dw = du_da*da_dw + du_db*db_dw
print(f'du/dw is {du_dw}')
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df_dw is {df_dw}')

##### alternative chain rule
# df/dw = df/da*da/dw + df/db*db/dw
# df/da = df/du*du/da + df/dv*dv/da
# df/db = df/du*du/db + df/dv*dv/db
##### adding other variables
# df/dx = df/da*da/dx + df/db*db/dx
# df/dy = df/da*da/dy + df/db*db/dy
# df/dz = df/da*da/dz + df/db*db/dz
##### 
df_da = df_du*du_da + df_dv*dv_da
df_db = df_du*du_db + df_dv*dv_db
print(f'df/da is {df_da} df/db is {df_db}')
da_dx = 2*(w+x+y+z)
da_dy = 2*(w+x+y+z)
da_dz = 2*(w+x+y+z)
db_dx = 2*(w+x-y-z)
db_dy = -2*(w+x-y-z)
db_dz = -2*(w+x-y-z)

df_dw = df_da*da_dw + df_db*db_dw
df_dx = df_da*da_dx + df_db*db_dx
df_dy = df_da*da_dy + df_db*db_dy
df_dz = df_da*da_dz + df_db*db_dz
print(f'Other direction computation df/dw: {df_dw}')
print(f'Other direction computation df/dx: {df_dx}')
print(f'Other direction computation df/dy: {df_dy}')
print(f'Other direction computation df/dz: {df_dz}')

#### implementing with torch

w = torch.tensor([-1.],requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.],requires_grad=True)
z = torch.tensor([1.],requires_grad=True)
a = (w+x+y+z)**2
b = (w+x-y-z)**2
u = (a+b)**2
v = (a-b)**2
f = (u+v)**2
f.backward()
print("Using torch")
print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, {z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, {z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, {z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, {z.data.item()} is {z.grad.data.item()}')

# https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
print(f'Loss {loss}')
print(f'Gradient for z = {z.grad_fn}')
print(f'Gradient for loss = {loss.grad_fn}')
loss.backward()
print(f'Gradient at w {w.grad}')
print(f'Gradient at b {b.grad}')
print("Detaching gardient from z")
z_det = z.detach()
print(z_det.requires_grad)

# jacobian product
# v^TJ
inp = torch.eye(4,5,requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out),retain_graph=True)
print(f"First call \n{inp.grad}")
out.backward(torch.ones_like(out),retain_graph=True)
print(f"Second call \n {inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out),retain_graph=True)
print(f"\n Call after zeroing \n {inp.grad}")

#https://docs.pytorch.org/tutorials/beginner/understanding_leaf_vs_nonleaf_tutorial.html
import torch.nn.functional as F
x = torch.ones(1,3)
W = torch.ones(3,2, requires_grad=True)
b = torch.ones(1,2,requires_grad=True)
y = torch.ones(1,2)

# forward
z = (x @ W) + b
y_pred = F.relu(z)
loss = F.mse_loss(y_pred, y)
print(f"x is leaf? {x.is_leaf=}")
print(f"z is leaf? {z.is_leaf=}")
loss.backward()
print(f"W {W} and W gradient \n {W.grad}")
print(f"b {b} and b gradient {b.grad}")

z = (x @ W) + b
y_pred = F.relu(z)
loss = F.mse_loss(y_pred, y)

# retain grad
z.retain_grad()
y_pred.retain_grad()
loss.retain_grad()

W.grad = None
b.grad = None

loss.backward()
print(f"W {W} and W gradient \n {W.grad}")
print(f"b {b} and b gradient \n {b.grad}")
print(f"b {z} and b gradient \n {z.grad}")
print(f"b {y} and b gradient \n {y.grad}")
print(f"b {loss} and b gradient \n {loss.grad}")



### adding logistic loss problem
### number of observations n, 
### number of features p
criterion = torch.nn.functional.binary_cross_entropy
n = 100
p = 20

X = torch.tensor(torch.rand(n,p))
theta = torch.randn(p,1,requires_grad=True)
z = X@theta
print(f"Shape of X: {X.shape}")
print(f'Shape of Xtheta: {(X@theta).shape}')
sigmoid = nn.Sigmoid()
h_theta_X = sigmoid(X@theta)
print(f"Shape of sigm(Xtheta) {h_theta_X}")

### dummy target
y = torch.randint(0,2,(n,1)).float()
#print(f"Loss between dummy and model is {loss}")
#loss.backward()
#print(f"The gradient of the loss w.r.t theta {theta.grad}")
def h_theta_X(theta):
    return sigmoid(X@theta)

print(f"theta {theta}")
lr = 0.01
Nsteps = 100
for step in range(Nsteps):
    theta = torch.randn(p,1,requires_grad=True)
    z = X@theta
    h_theta_X = sigmoid(z)
    loss = criterion(h_theta_X,y)
    loss.backward()
    with torch.no_grad():
        theta -= lr*theta.grad
        theta.grad.zero_()

print(f"Params after {Nsteps} {theta}")

### subgradient and proximal examples

m, p = 1000, 10
X = torch.randn(m,p)
true_theta = torch.tensor([5.,-3,2.]+[0.]*7).unsqueeze(1)
y = (torch.sigmoid(X @ true_theta) > 0.5).float()

## initial guess
theta = torch.randn(p,1,requires_grad=True)
lr = 0.01
l1_lambda=0.01
tau = lr *l1_lambda

def soft_threshold(v,tau):
    return torch.sign(v)*torch.relu(torch.abs(v)-tau)

print(f"Running for lambda {l1_lambda} and lr {lr}")
for step in range(1000):
    z = X @ theta
    predictions = torch.sigmoid(z)
    m = X.size(0)
    gradient = (1/m) *X.t() @ (predictions-y)
    ### initial smooth step
    theta_temp = theta - lr*gradient
    ### update weights
    with torch.no_grad():
        new_theta = soft_threshold(theta_temp,tau)
        theta.copy_(new_theta)

    if step % 100 == 0:
        bce_loss = -torch.mean(y*torch.log(predictions+1e-8)+(1-y)*torch.log(1-predictions +1e-8))
        l1_penalty = l1_lambda * torch.norm(theta,1)
        print(f"Step {step}: Total loss = {bce_loss + l1_penalty:.4f}")



def softMax(x):
    return torch.exp(x) / torch.sum(torch.exp(x))


class CustomEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx,logits,true_labels):
        probabilities = softMax(logits)
        loss = - torch.sum(true_labels * torch.log(probabilities))
        ctx.save_for_backward(probabilities,true_labels)
        return loss

    @staticmethod
    def backward(ctx,grad_output):
        probabilities, true_labels = ctx.saved_tensors
        grad_logits = (probabilities- true_labels) * grad_output
        grad_labels = None
        return grad_logits,grad_labels
        pass


 
