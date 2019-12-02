
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time


# In[2]:


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

def Adam(cost_func,grad_func,theta_0,niters,alpha = None,beta1 = None,beta2 = None,epsilon = None):
    
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
    theta_vec = np.zeros(theta_0.shape)
    theta_vec = theta_0
    
    for tt in np.arange(1,niters+1):
        # get gradient
        grad_vec = grad_func(theta_vec)
        
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
    
    return theta_vec,cost_out        


# In[3]:


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

def RAdam(cost_func,grad_func,theta_0,niters,alpha,beta1 = None,beta2 = None, epsilon = None):
    
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
    cost_out[0] = cost_func(theta_0)
    
    theta_vec = np.zeros(theta_0.shape)
    theta_vec = theta_0
    
    rho_infinity = 2/(1-beta2) - 1
    
    for tt in np.arange(1,niters+1):
        # get gradient
        grad_vec = grad_func(theta_vec)
        
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
    
    return theta_vec,cost_out        


# In[80]:


# make synthetic data
def f_x_1(x):
    return x**2+6*x+9


# In[81]:


# synthetic train and test
ntrain = 500
ntest = 100

noise_mean = 0
noise_sd_1 = 1
noise_sd_2 = 10
x_vec = np.random.normal(noise_mean,noise_sd_1,ntrain+ntest)
x2_vec = x_vec**2
fx_vec = f_x_1(x_vec) # true value (used to plot)


y_vec = fx_vec + np.random.normal(noise_mean,noise_sd_2,x_vec.shape[0]) # add gaussian

X_Train_mat = np.vstack([np.ones((1,ntrain)),x_vec[:ntrain].T,x2_vec[:ntrain].T])
X_Test_mat = np.vstack([np.ones((1,ntest)),x_vec[ntrain:].T,x2_vec[ntrain:].T])

y_train = y_vec[:ntrain]
y_test = y_vec[ntrain:]

x_train = x_vec[:ntrain]
x_test = x_vec[ntrain:]

fx_train = fx_vec[:ntrain]
fx_test = fx_vec[ntrain:]


# In[112]:


# adam / radam parameters
alpha_start = 0.05
decrease_cond = False

if decrease_cond:
    dmid = 0.1
    dend = 0.01
else:
    dmid = 1
    dend = 1

theta_0 = np.zeros((3))
niters = 5000
third_iter = np.int(np.floor(niters/3))
extra_iters = np.int(niters - third_iter*3)
radam_alph = np.concatenate([alpha_start*np.ones((third_iter+extra_iters)),                             dmid*alpha_start*np.ones((third_iter)),                             dend*alpha_start*np.ones((third_iter))])


# In[113]:


def cost_func(weights):
    return (1/2)*np.linalg.norm(X_Train_mat.T.dot(weights) - y_train)**2


# In[114]:


def grad_func(weights):
    return (1/X_Train_mat.shape[1])*X_Train_mat.dot(X_Train_mat.T.dot(weights) - y_train)
# radam can blow up if step size isn't small enough?


# In[115]:


# run iters
adam_weights,adam_cost = Adam(cost_func,grad_func,theta_0,niters,alpha=alpha_start)
radam_weights,radam_cost = RAdam(cost_func,grad_func,theta_0,niters,radam_alph)


# In[116]:


print(adam_weights)
print(radam_weights)


# In[117]:


plt.scatter(x_train,X_Train_mat.T.dot(adam_weights))
plt.scatter(x_train,fx_train)
plt.scatter(x_train,y_train)
plt.show()


# In[118]:


plt.scatter(x_train,X_Train_mat.T.dot(radam_weights))
plt.scatter(x_train,fx_train)
plt.scatter(x_train,y_train)
plt.show()


# In[119]:


plt.plot(adam_cost)
plt.plot(radam_cost)
plt.show()


# In[110]:


def test_MSE(weights):
    return (1/y_test.shape[0])*(1/2)*np.linalg.norm(X_Test_mat.T.dot(weights) - y_test)**2


# In[111]:


print(test_MSE(adam_weights))
print(test_MSE(radam_weights))

