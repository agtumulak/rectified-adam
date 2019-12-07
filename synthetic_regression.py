
# coding: utf-8

# In[2]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from sklearn.datasets.samples_generator import make_regression


# In[3]:


ntrain = 2000
ntest = 1000
batch_size = 128
X,y =  make_regression(n_samples=ntrain+ntest, n_features=500, random_state=0, noise = 1)
Xtrain = np.vstack([np.ones((ntrain)),X[:ntrain,:].T])
Xtest = np.vstack([np.ones((ntest)),X[ntrain:,:].T])
ytrain = y[:ntrain]
ytest = y[ntrain:]

wLS = np.linalg.inv(Xtrain.dot(Xtrain.T)).dot(Xtrain.dot(ytrain))


# In[4]:


def test_MSE(weights):
    return (1/ytest.shape[0])*(1/2)*np.linalg.norm(Xtest.T.dot(weights) - ytest)**2


# In[5]:


# Adam
# inputs:
# cost_func - cost function for objective function, f(theta)
# grad_func - function for gradient of objective, f'(theta)
# theta_0 - initial parameters, vector
# niters - number of iterations, scalar
# alpha - step size, scalar; default = 0.001
# beta1 - decay rate for first moment, scalar [0,1); default = 0.9
# beta2 - decay rate for second moment, scalar [0,1); default = 0.999
# epsilon - tolerance, scalar; default = 10e-8
#
# outputs:
# final_weights - weights after iterations
# cost_over_time - cost function evaluated at each iteration [0,niters]

def Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha = None,beta1 = None,beta2 = None,epsilon = None):
    
    # check parameters
    if alpha == None:
        alpha = 0.001
    if beta1 == None or beta1 < 0 or beta1 >= 1:
        beta1 = 0.9
    if beta2 == None or beta2 < 0 or beta2 >= 1:
        beta2 = 0.999
    if epsilon == None:
        epsilon = 10e-8
    
    # initial values
    m_vec = np.zeros(theta_0.shape)
    bias_correct_m = np.zeros(theta_0.shape)
    
    v_vec = np.zeros(theta_0.shape)
    bias_correct_v = np.zeros(theta_0.shape)
    
    cost_out = np.zeros(niters+1)
    cost_out[0] = cost_func(theta_0)
    
    mse_out = np.zeros(niters+1)
    mse_out[0] = test_MSE(theta_0)
    
    theta_vec = np.zeros(theta_0.shape)
    theta_vec = theta_0
    
    np.random.seed(0)
    nbatches = len(range(0,ntrain,batch_size))
    for tt in np.arange(1,niters+1):
        i_array = np.random.permutation(ntrain)
        for ii in np.arange(0,nbatches):
            batch_inds = i_array[ii:ii+batch_size]
            # get gradient
            grad_vec = grad_func(theta_vec,batch_inds)

            # first moment
            m_vec = beta1*m_vec + (1-beta1)*grad_vec

            # second moment
            v_vec = beta2*v_vec + (1-beta2)*(grad_vec**2)

            # bias-correction 
            bias_correct_m = m_vec/(1-beta1**tt)
            bias_correct_v = v_vec/(1-beta2**tt)

            # update weights
            theta_vec = theta_vec - alpha*bias_correct_m/(bias_correct_v**(1/2) + epsilon)

        # cost function
        cost_out[tt] = cost_func(theta_vec)
        mse_out[tt] = test_MSE(theta_vec)
    
    return theta_vec,cost_out,mse_out     


# In[6]:


# RAdam
# inputs:
# cost_func - cost function for objective function, f(theta)
# grad_func - function for gradient of objective, f'(theta)
# theta_0 - initial parameters, vector
# niters - number of iterations, scalar
# alpha - step size, vector
# beta1 - decay rate for first moment, scalar [0,1); default = 0.9
# beta2 - decay rate for second moment, scalar [0,1); default = 0.999
#
# outputs:
# final_weights - weights after iterations
# cost_over_time - cost function evaluated at each iteration [0,niters]

def RAdam(cost_func,grad_func,theta_0,niters,alpha,ntrain,batch_size,beta1 = None,beta2 = None, epsilon = None):
    
    # check parameters
    if beta1 == None or beta1 < 0 or beta1 >= 1:
        beta1 = 0.9
    if beta2 == None or beta2 < 0 or beta2 >= 1:
        beta2 = 0.999
    if epsilon == None:
        epsilon = 10e-8
    
    # initial values
    m_vec = np.zeros(theta_0.shape)
    bias_correct_m = np.zeros(theta_0.shape)
    
    v_vec = np.zeros(theta_0.shape)
    bias_correct_v = np.zeros(theta_0.shape)
    
    cost_out = np.zeros(niters+1)
    mse_out = np.zeros(niters+1)
    cost_out[0] = cost_func(theta_0)
    mse_out[0] = test_MSE(theta_0)
    
    theta_vec = np.zeros(theta_0.shape)
    theta_vec = theta_0
    
    rho_infinity = 2/(1-beta2) - 1
    
    np.random.seed(0)
    nbatches = len(range(0,ntrain,batch_size))
    for tt in np.arange(1,niters+1):
        i_array = np.random.permutation(ntrain)
        for ii in np.arange(0,nbatches):
            batch_inds = i_array[ii:ii+batch_size]
            # get gradient
            grad_vec = grad_func(theta_vec,batch_inds)

            # second moment
            v_vec = beta2*v_vec + (1-beta2)*(grad_vec**2)

            # first moment
            m_vec = beta1*m_vec + (1-beta1)*grad_vec

            # bias-correction first moment
            bias_correct_m = m_vec/(1-beta1**tt)

            # compute iteration SMA
            rho_iter = rho_infinity - 2*tt*beta2**tt/(1-beta2**tt)

            if rho_iter > 4:
                # variance is tractable
                bias_correct_v = ((v_vec/(1-beta2**tt))**(1/2) + epsilon)
                num_term = (rho_iter-4)*(rho_iter - 2)*rho_infinity
                denom_term = (rho_infinity-4)*(rho_infinity-2)*rho_iter
                rectification_term = (num_term/denom_term)**(1/2)
                # update weights
                theta_vec = theta_vec - alpha[tt-1]*rectification_term*bias_correct_m/bias_correct_v
            else:
                # update weights
                theta_vec = theta_vec - alpha[tt-1]*bias_correct_m

        # cost function
        cost_out[tt] = cost_func(theta_vec)
        mse_out[tt] = test_MSE(theta_vec)
    
    return theta_vec,cost_out,mse_out


# In[7]:


# normal SGD
def SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha = None):
    
    # check parameters
    if alpha == None:
        alpha = 0.001
    
    # initial values
    cost_out = np.zeros(niters+1)
    cost_out[0] = cost_func(theta_0)
    mse_out = np.zeros(niters+1)
    mse_out[0] = test_MSE(theta_0)
    theta_vec = np.zeros(theta_0.shape)
    theta_vec = theta_0
    
    np.random.seed(0)
    nbatches = len(range(0,ntrain,batch_size))
    for tt in np.arange(1,niters+1):
        i_array = np.random.permutation(ntrain)
        for ii in np.arange(0,nbatches):
            batch_inds = i_array[ii:ii+batch_size]
            # get gradient
            grad_vec = grad_func(theta_vec,batch_inds)

            # update weights
            theta_vec = theta_vec - alpha*grad_vec

        # cost function
        cost_out[tt] = cost_func(theta_vec)
        mse_out[tt] = test_MSE(theta_vec)
    
    return theta_vec,cost_out,mse_out        


# In[8]:


# adam / radam parameters
alpha_start = 0.1 # .01
alpha_start_2 = .03 # .05
alpha_start_3 = .01 # .001
alpha_start_4 = .005 # .003
alpha_start_5 = .003 # .005
decrease_cond = False

if decrease_cond:
    dmid = 0.1
    dend = 0.01
else:
    dmid = 1
    dend = 1

#theta_0 = np.zeros((X_Train_mat.shape[0]))
xxx = 0.01
theta_0 = xxx*np.ones((Xtrain.shape[0]))
niters = 1000
third_iter = np.int(np.floor(niters/3))
extra_iters = np.int(niters - third_iter*3)
radam_alph = np.concatenate([alpha_start*np.ones((third_iter+extra_iters)),                             dmid*alpha_start*np.ones((third_iter)),                             dend*alpha_start*np.ones((third_iter))])
radam_alph_2 = np.concatenate([alpha_start_2*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_2*np.ones((third_iter)),                             dend*alpha_start_2*np.ones((third_iter))])
radam_alph_3 = np.concatenate([alpha_start_3*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_3*np.ones((third_iter)),                             dend*alpha_start_3*np.ones((third_iter))])
radam_alph_4 = np.concatenate([alpha_start_4*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_4*np.ones((third_iter)),                             dend*alpha_start_4*np.ones((third_iter))])
radam_alph_5 = np.concatenate([alpha_start_5*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_5*np.ones((third_iter)),                             dend*alpha_start_5*np.ones((third_iter))])


# In[9]:


def cost_func(weights):
    return (1/2)*np.linalg.norm(Xtrain.T.dot(weights) - ytrain)**2


# In[10]:


def grad_func(weights,batch_inds):
    return (1/len(batch_inds))*Xtrain[:,batch_inds].dot(Xtrain[:,batch_inds].T.dot(weights) - ytrain[batch_inds])
# radam can blow up if step size isn't small enough?


# In[11]:


# run iters
a_w_1,a_c_1,a_mse_1 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start)
a_w_2,a_c_2,a_mse_2 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_2)
a_w_3,a_c_3,a_mse_3 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_3)
a_w_4,a_c_4,a_mse_4 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_4)
a_w_5,a_c_5,a_mse_5 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_5)


# In[12]:


r_w_1,r_c_1,r_mse_1 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph,ntrain,batch_size)
r_w_2,r_c_2,r_mse_2 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph_2,ntrain,batch_size)
r_w_3,r_c_3,r_mse_3 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph_3,ntrain,batch_size)
r_w_4,r_c_4,r_mse_4 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph_4,ntrain,batch_size)
r_w_5,r_c_5,r_mse_5 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph_5,ntrain,batch_size)


# In[13]:


s_w_1,s_c_1,s_mse_1 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start)
s_w_2,s_c_2,s_mse_2 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_2)
s_w_3,s_c_3,s_mse_3 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_3)
s_w_4,s_c_4,s_mse_4 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_4)
s_w_5,s_c_5,s_mse_5 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_5)


# In[29]:


print(alpha_start,np.linalg.norm(a_w_1 - wLS),np.linalg.norm(r_w_1 - wLS),np.linalg.norm(s_w_1 - wLS))
print()
print(alpha_start_2,np.linalg.norm(a_w_2 - wLS),np.linalg.norm(r_w_2 - wLS),np.linalg.norm(s_w_2 - wLS))
print()
print(alpha_start_3,np.linalg.norm(a_w_3 - wLS),np.linalg.norm(r_w_3 - wLS),np.linalg.norm(s_w_3 - wLS))
print()
print(alpha_start_4,np.linalg.norm(a_w_4 - wLS),np.linalg.norm(r_w_4 - wLS),np.linalg.norm(s_w_4 - wLS))
print()
print(alpha_start_5,np.linalg.norm(a_w_5 - wLS),np.linalg.norm(r_w_5 - wLS),np.linalg.norm(s_w_5 - wLS))


# In[15]:


plt.plot(a_c_2,label='adam')
plt.plot(r_c_2,label='radam')
plt.plot(s_c_2,label='sgd')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('cost vs epoch for α = {}'.format(alpha_start_2))
plt.legend()
plt.show()


# In[16]:


print(a_c_1[-1],r_c_1[-1],s_c_1[-1])
print(a_mse_1[-1],r_mse_1[-1],s_mse_1[-1])
print(a_mse_2[-1],r_mse_2[-1],s_mse_2[-1])
print(a_mse_3[-1],r_mse_3[-1],s_mse_3[-1])
print(a_mse_4[-1],r_mse_4[-1],s_mse_4[-1])
print(a_mse_5[-1],r_mse_5[-1],s_mse_5[-1])


# In[17]:


plt.plot(r_c_1,label='α = {}'.format(alpha_start))
plt.plot(r_c_2,label='α = {}'.format(alpha_start_2))
plt.plot(r_c_3,label='α = {}'.format(alpha_start_3))
plt.plot(r_c_4,label='α = {}'.format(alpha_start_4))
plt.plot(r_c_5,label='α = {}'.format(alpha_start_5))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('RAdam cost vs epoch')
plt.show()


# In[18]:


plt.plot(a_c_1,label='α = {}'.format(alpha_start))
plt.plot(a_c_2,label='α = {}'.format(alpha_start_2))
plt.plot(a_c_3,label='α = {}'.format(alpha_start_3))
plt.plot(a_c_4,label='α = {}'.format(alpha_start_4))
plt.plot(a_c_5,label='α = {}'.format(alpha_start_5))
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Adam cost vs epoch')
plt.show()


# In[28]:


plt.plot(a_mse_1,label='α = {}'.format(alpha_start))
plt.plot(a_mse_2,label='α = {}'.format(alpha_start_2))
plt.plot(a_mse_3,label='α = {}'.format(alpha_start_3))
plt.plot(a_mse_4,label='α = {}'.format(alpha_start_4))
plt.plot(a_mse_5,label='α = {}'.format(alpha_start_5))
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Adam MSE vs epoch')
plt.show()


# In[19]:


plt.plot(r_mse_1,label='α = {}'.format(alpha_start))
plt.plot(r_mse_2,label='α = {}'.format(alpha_start_2))
plt.plot(r_mse_3,label='α = {}'.format(alpha_start_3))
plt.plot(r_mse_4,label='α = {}'.format(alpha_start_4))
plt.plot(r_mse_5,label='α = {}'.format(alpha_start_5))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('RAdam MSE vs epoch')
plt.show()


# In[20]:


plt.plot(s_c_1,label='α = {}'.format(alpha_start))
plt.plot(s_c_2,label='α = {}'.format(alpha_start_2))
plt.plot(s_c_3,label='α = {}'.format(alpha_start_3))
plt.plot(s_c_4,label='α = {}'.format(alpha_start_4))
plt.plot(s_c_5,label='α = {}'.format(alpha_start_5))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('SGD cost vs epoch')
plt.show()


# In[21]:


plt.plot(s_mse_1,label='α = {}'.format(alpha_start))
plt.plot(s_mse_2,label='α = {}'.format(alpha_start_2))
plt.plot(s_mse_3,label='α = {}'.format(alpha_start_3))
plt.plot(s_mse_4,label='α = {}'.format(alpha_start_4))
plt.plot(s_mse_5,label='α = {}'.format(alpha_start_5))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('SGD MSE vs epoch')
plt.show()


# In[27]:


s_mse_5[-1]

