
# coding: utf-8

# In[4]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time


# In[95]:


# test out example

# read in X, y
read_in_Boston = np.loadtxt("boston-corrected.csv", delimiter=',', skiprows=1)
y_CMEDV = read_in_Boston[:,0] # n (506) length vector
y_train = y_CMEDV[0:46*10]
y_test = y_CMEDV[-46:]
X_mat = read_in_Boston[:,1:].T # d x n (13 x 506)
X_mat_train = X_mat[:,0:46*10]
X_mat_test = X_mat[:,-46:]
d,ntrain = X_mat_train.shape
ntest = y_test.shape[0]

# tools for sphering
II = np.identity(ntrain)
nones = np.ones((ntrain,1))
outones = np.outer(nones,nones)

# mean vector
mu_vec = (1/ntrain)*X_mat_train.dot(nones)

# sigma vector
rep_mu = np.outer(mu_vec,nones)
X_sub = X_mat_train - rep_mu
X_sub_square = np.square(X_sub)
sigma_vec = np.sqrt((1/(ntrain-1))*X_sub_square.dot(nones))
one_over_sigma = np.divide(1,sigma_vec)

# center matrix
proj_mat = II - (1/ntrain)*outones
XC_train = X_mat_train.dot(proj_mat)

# final sphering
X_sphere_train = np.diagflat(one_over_sigma).dot(XC_train)

# augment for bias term
X_sphere_aug_train = np.vstack([np.ones((1,ntrain)),X_sphere_train])

# for lambda = 0 apply to test data
# first sphere + augment test data
testones = np.ones((y_test.shape[0],1))
rep_mu = np.outer(mu_vec,testones)

XC_test = X_mat_test - rep_mu

X_test_sphere = np.diagflat(one_over_sigma).dot(XC_test)

X_sphere_aug_test = np.vstack([np.ones((1,y_test.shape[0])),X_test_sphere])

# least squares weights
wLS = np.linalg.inv(X_sphere_aug_train.dot(X_sphere_aug_train.T)).dot(X_sphere_aug_train.dot(y_train))


# In[96]:


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


# In[119]:


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


# In[18]:


# cost function
def cost_use(weights):
    results = X_sphere_aug_train.T.dot(weights)
    return sum((results - y_train)**2)/ntrain


# In[8]:


def grad_use(weights):
    grad = X_sphere_aug_train.dot(X_sphere_aug_train.T.dot(weights) - y_train)
    return grad


# In[116]:


alpha_adam = 0.05
beta1_adam = 0.9
beta2_adam = 0.999
epsilon_adam = 10e-8

theta_0 = np.zeros(wLS.shape)
niters = 5000

adam_weights,adam_cost = Adam(cost_use,grad_use,theta_0,niters,alpha = alpha_adam,beta1 = beta1_adam,beta2 = beta2_adam,epsilon = epsilon_adam)


# In[124]:


first_rate = 1e-5*np.ones((2175))
second_rate = 0.1*1e-5*np.ones(1105)
third_rate = 0.01*1e-5*np.ones(1720)
alpha_radam = np.hstack([first_rate,second_rate,third_rate])

full_alpha = 1e-5*np.ones((5000))

radam_weights,radam_cost = RAdam(cost_use,grad_use,theta_0,niters,alpha_radam,beta1=beta1_adam,beta2 = beta2_adam,epsilon = epsilon_adam)


# In[125]:


print("The learned weights from adam after",niters,"iterations are:")
print(adam_weights)

print("The learned weights from radam after",niters,"iterations are:")
print(radam_weights)

print("The least squares weights are:")
print(wLS)


# In[122]:


plt.plot(np.arange(0,5001),adam_cost,'b')
plt.title("adam training error vs iteration")
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.show()


# In[126]:


plt.plot(np.arange(0,5001),radam_cost,'b')
plt.title("radam training error vs iteration")
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.show()


# In[58]:


t_vec = np.arange(1,niters+1)
beta2t = beta2_adam**t_vec
term = 2*t_vec*beta2_adam**t_vec/(1-beta2t)
rho_infinity = 2/(1-beta2_adam) - 1


# In[60]:


print(rho_infinity)


# In[61]:


beta2_adam


# In[70]:


rrr = rho_infinity - term


# In[71]:


print(rrr[0:100])


# In[102]:


print(radam_cost[0:100])


# In[106]:


print(adam_cost[0:100])


# In[108]:




