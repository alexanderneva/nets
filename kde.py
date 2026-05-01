import numpy as np
import matplotlib.pyplot as plt

def rbf(x,y,sigma=1):
    k = np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))
    return k 
def laplacian(x,y,sigma):
    k = np.exp(-np.linalg.norm(x-y)/sigma)
    return k

X = np.random.normal(size=(20,10))
# labels
y = np.random.randint(0,2,size=(20,1))

def kde(X,sigma=1):
    kde = np.zeros_like(X@X.T)
    rows = X.shape[0]
    for row_1 in range(rows):
        for row_2 in range(rows):
            kde[row_1,row_2]+=rbf(X[row_1,:],X[row_2,:],sigma=sigma)
    return kde
print(kde(X).shape)

#fig, (ax1,ax2,ax3) = plt.subplots(3)
#ax1.hist(kde(X,sigma=1))
#ax1.set_title("Test")
#ax2.hist(kde(X,sigma=2))
#ax3.hist(kde(X,sigma=2))
#
#plt.savefig('kde.jpg')
#print(kde(X).max(axis=1))
#
def naraya(x,y,X,h=1):
    length = np.sum(x - X,axis=1,keepdims=True) 
    return y.T@np.sum(x - X,axis=1,keepdims=True) / length 

def naraya_predict(X_train,y_train,x_query,kernel_func=rbf,sigma=1):
    weights = np.array([kernel_func(x_query,x_i,sigma) for x_i in X_train])
    num = y_train.T@weights
    den = np.sum(weights)
    return num / den

X_train = np.sort(5*np.random.rand(40,1),axis=0)
y_train = np.sin(X_train).ravel()
y_train += 0.2 *np.random.randn(40)

X_query = np.linspace(0,5,100)[:,np.newaxis]

y_pred_gaussian = [naraya_predict(X_train,y_train,x_q,kernel_func=rbf,sigma=0.2) for x_q in X_query]
y_pred_laplacian = [naraya_predict(X_train,y_train,x_q,kernel_func=laplacian,sigma=0.2) for x_q in X_query]

#plt.scatter(X_train,y_train)
#plt.plot(X_query,y_pred_gaussian)
#plt.plot(X_query,y_pred_laplacian)
#plt.title("Gaussian vs. laplacian kernel")
#plt.legend("upper right")
#plt.savefig('kr.jpg')
