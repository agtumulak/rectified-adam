
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time


# In[2]:


# make synthetic data
def cost_func(theta):
    return theta**2*(theta-6)*(theta-7)


# In[3]:


def grad_func(theta,batch_inds):
    return 4*theta**3-39*theta**2+84*theta


# In[4]:


def test_MSE(weights):
    return 1


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


niters = 5000
alpha_start = 0.01
radam_alph = alpha_start*np.ones((niters))
theta_0 = np.array([3])
a_w_1,a_c_1,a_mse_1 = Adam(cost_func,grad_func,theta_0,niters,1,1,alpha=alpha_start)
r_w_1,r_c_1,r_mse_1 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph,1,1)
s_w_1,s_c_1,s_mse_1 = SGD(cost_func,grad_func,theta_0,niters,1,1,alpha=alpha_start)


# In[9]:


a_w = []
r_w = []
s_w = []
for tht_0 in np.arange(-3.7,8.1,.1):
    tht_0_1 = np.array([tht_0])
    a_w_temp,temp,temp2 = Adam(cost_func,grad_func,tht_0_1,niters,1,1,alpha=alpha_start)
    a_w.append(a_w_temp)
    
    r_w_temp,temp,temp = RAdam(cost_func,grad_func,tht_0_1,niters,radam_alph,1,1)
    r_w.append(r_w_temp)

    s_w_temp,temp,temp2 = SGD(cost_func,grad_func,tht_0_1,niters,1,1,alpha=alpha_start)
    s_w.append(s_w_temp)


# In[14]:


plt.plot(np.arange(-3.7,8.1,.1),cost_func(np.arange(-3.7,8.1,.1)))
plt.title('f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


# In[17]:


plt.plot(np.arange(-3.7,8.1,.1),cost_func(np.arange(-3.7,8.1,.1)))
plt.title('zoomed in f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim([-20,50])
plt.xlim([-1,8])
plt.show()
# global min x = 6.538
# local min x = 0


# In[29]:


plt.plot(np.arange(-3.7,8.1,.1),a_w,label = "adam")
plt.plot(np.arange(-3.7,8.1,.1),r_w,label = "radam")
plt.plot(np.arange(-3.7,8.1,.1),s_w,label = "SGD")
plt.legend()
plt.title('Comparison of Convergence')
plt.ylabel('Optimal x-value')
plt.xlabel('Initialization x-value')
plt.show()


# In[28]:


print(np.arange(-3.7,8.1,.1),s_w)

