import numpy as np
import matplotlib.pyplot as plt
import sklearn
try: from sklearn.model_selection import train_test_split
except: from sklearn.cross_validation import train_test_split

import sys,os
# print(os.path.dirname(os.path.abspath(__file__)))
# os.path.join('..','djinn/')
sys.path.append('../djinn')
import djinn
import djinn_fns


def gaussian(x,p,mu,sig):
    c = 1./(sig*np.sqrt(2*np.pi))
    exp = -1*((x-np.ones(len(x))*mu)**2/(2*sig**2))**p
    return c*np.exp(exp)

def ddx_gaussian(x,p,mu,sig):
    # from wolfram alpha:
    # d/dx(e^(-(2^(-p - 1/2) ((x - μ)^2/σ^2)^p)/(sqrt(π) σ))) = -(2^(1/2 - p) p σ e^(-(2^(-p - 1/2) ((x - μ)^2/σ^2)^p)/(sqrt(π) σ)) ((x - μ)^2/σ^2)^(p + 1))/(sqrt(π) (x - μ)^3)

    c = -(2**(.5 - p)*p*sig*((x - mu)**2/sig**2)**(p + 1))/(np.sqrt(np.pi)*(x - mu)**3)
    exp = -(2**(-p - .5)*((x - mu)**2/sig**2)**p)/(np.sqrt(np.pi)*sig)
    return c*np.exp(exp)


num_pts = 1000
X = np.linspace(-5,5,num_pts)
y = np.zeros((num_pts,2))
y[:,0] = gaussian(X,1,0,1)
y[:,1] = gaussian(X,3,0,1)

# plt.figure()
# plt.plot(X,y)
# plt.plot(X,ddx_gaussian(x,1,0,1))
# plt.plot(X,ddx_gaussian(x,3,0,1))
# plt.show()

X=X.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=1)

print("djinn example")
modelname="gaussian_djinn_test"   # name the model
ntrees=1                 # number of trees = number of neural nets in ensemble
maxdepth=4               # max depth of tree -- optimize this for each data set
dropout_keep=1.0         # dropout typically set to 1 for non-Bayesian models

load_model = True

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

g = model.gradient(x_test,output_idx=0)
h = model.hessian(x_test)

g_baseline = ddx_gaussian(x_test,1,0,1) #+ddx_gaussian(x_test,3,0,1)
# print(len(g))
# print(len(g[0]))
# print(sum(g[0]-g_baseline))
c = np.zeros((200,3))
c[:,0] = g[0][:,0]
c[:,1] = g_baseline[:,0]
c[:,2] = g[0][:,0] - g_baseline[:,0]
print(c)
# print(h)

#close model
model.close_model()
