import numpy as np
import matplotlib.pyplot as plt
import sklearn
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../djinn')
import djinn
import djinn_fns


def gaussian(x,p,mu,sig):
    c = 1./(sig*np.sqrt(2*np.pi))
    exp = -1*((x-mu)**2/(2*sig**2))**p
    return c*np.exp(exp)

def multivariate_gaussian(x,mu,sig):
    c = 1./(np.sqrt(np.linalg.det(sig)*(2*np.pi)**(len(x))))
    exp = -.5*((x-mu).T@np.linalg.inv(sig)@(x-mu))
    return c*np.exp(exp)


def ddx_gaussian(x,p,mu,sig):
    # from wolfram alpha:
    # d/dx(e^(-(2^(-p - 1/2) ((x - μ)^2/σ^2)^p)/(sqrt(π) σ))) = -(2^(1/2 - p) p σ e^(-(2^(-p - 1/2) ((x - μ)^2/σ^2)^p)/(sqrt(π) σ)) ((x - μ)^2/σ^2)^(p + 1))/(sqrt(π) (x - μ)^3)

    c = -(2**(1/2 - p)*p*sig*((x - mu)**2/sig**2)**(p + 1))/(np.sqrt(np.pi)*(x - mu)**3)
    exp = -(2**(-p - 1/2)*((x - mu)**2/sig**2)**p)/(np.sqrt(np.pi)*sig)
    return c*np.exp(exp)


num_pts = 1000
X1 = np.random.rand(num_pts)*4-2
X2 = np.random.rand(num_pts)*4-2
X = np.zeros((num_pts,2))
X[:,0] = X1
X[:,1] = X2

y = np.zeros((num_pts,2))
# y[:,0] = gaussian(X1,1,X2,1)
y[:,1] = gaussian(X1,3,X2,1)
for i in range(X.shape[0]):
    y[i,0] = multivariate_gaussian(X[i,:],[0,1],[[1,0],[0,2]])


fig =plt.figure()
ax = fig.gca(projection='3d')
plt.scatter(X[:,0],X[:,1],y[:,0])
# plt.plot(X,ddx_gaussian(x,1,0,1))
# plt.plot(X,ddx_gaussian(x,3,0,1))
plt.show()

# X=X.reshape(-1,1)
print(X)
print(y)

x_train,x_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=1)

print("djinn example")
modelname="input_2d_djinn_test"   # name the model
ntrees=1                 # number of trees = number of neural nets in ensemble
maxdepth=4               # max depth of tree -- optimize this for each data set
dropout_keep=1.0         # dropout typically set to 1 for non-Bayesian models

load_model = False

if load_model:
    model=djinn.load(model_name=modelname)
else:
    #initialize the model
    model=djinn.DJINN_Regressor(ntrees,maxdepth,dropout_keep)

    # find optimal settings: this function returns dict with hyper-parameters
    # each djinn function accepts random seeds for reproducible behavior
    optimal=model.get_hyperparameters(x_train, y_train, random_state=1)
    batchsize=optimal['batch_size']
    learnrate=optimal['learn_rate']
    epochs=optimal['epochs']

    # train the model with hyperparameters determined above
    model.train(x_train,y_train,epochs=epochs,learn_rate=learnrate, batch_size=batchsize,
                  display_step=1, save_files=True, file_name=modelname,
                  save_model=True,model_name=modelname, random_state=1)

# make predictions
m=model.predict(x_test) #returns the median prediction if more than one tree

#evaluate results
mse=sklearn.metrics.mean_squared_error(y_test,m)
mabs=sklearn.metrics.mean_absolute_error(y_test,m)
exvar=sklearn.metrics.explained_variance_score(y_test,m)
print('MSE',mse)
print('M Abs Err',mabs)
print('Expl. Var.',exvar)

g = model.gradient(x_test,input_idx=0,output_idx=0)
# h = model.hessian(x_test)


g_baseline = np.zeros(x_test.shape[0])
for i,mu in enumerate(x_test[:,1]):
    g_baseline[i]= ddx_gaussian(x_test[i,0],1,mu,1) #+ddx_gaussian(x_test,3,0,1)
# print(len(g))
# print(g)
# print(len(g[0]))
# print(sum(g[0]-g_baseline))
c = np.zeros((200,3))
c[:,0] = np.array(g[0])
c[:,1] = g_baseline
c[:,2] = g[0] - g_baseline
print(c)
# print(h)

#close model
model.close_model()
