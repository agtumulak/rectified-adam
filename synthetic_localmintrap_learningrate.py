
# coding: utf-8

# In[56]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time


# In[150]:


# make synthetic data
def f_x_1(x):
    return x**2*(x-6)*(x-7)


# In[151]:


# synthetic train and test
ntrain = 5000
batch_size = 256
ntest = 1000

noise_mean = 0
noise_sd_1 = 1
noise_sd_2 = 10
x_vec = np.random.normal(noise_mean,noise_sd_1,ntrain+ntest)
x2_vec = x_vec**2
x3_vec = x_vec**3
x4_vec = x_vec**4
fx_vec = f_x_1(x_vec) # true value (used to plot)


y_vec = fx_vec + np.random.normal(noise_mean,noise_sd_2,x_vec.shape[0]) # add gaussian

X_Train_mat = np.vstack([np.ones((1,ntrain)),x_vec[:ntrain].T,x2_vec[:ntrain].T,x3_vec[:ntrain].T,x4_vec[:ntrain].T])
X_Test_mat = np.vstack([np.ones((1,ntest)),x_vec[ntrain:].T,x2_vec[ntrain:].T,x3_vec[ntrain:].T,x4_vec[ntrain:].T])

y_train = y_vec[:ntrain]
y_test = y_vec[ntrain:]

x_train = x_vec[:ntrain]
x_test = x_vec[ntrain:]

fx_train = fx_vec[:ntrain]
fx_test = fx_vec[ntrain:]


# In[152]:


def test_MSE(weights):
    return (1/y_test.shape[0])*(1/2)*np.linalg.norm(X_Test_mat.T.dot(weights) - y_test)**2


# In[153]:


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


# In[154]:


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


# In[155]:


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


# In[156]:


# adam / radam parameters
alpha_start = 0.01
alpha_start_2 = .05
alpha_start_3 = .001
alpha_start_4 = .003
alpha_start_5 = .005
decrease_cond = False

if decrease_cond:
    dmid = 0.1
    dend = 0.01
else:
    dmid = 1
    dend = 1

#theta_0 = np.zeros((X_Train_mat.shape[0]))
xxx = 0.01
theta_0 = xxx*np.ones((X_Train_mat.shape[0]))
niters = 1000
third_iter = np.int(np.floor(niters/3))
extra_iters = np.int(niters - third_iter*3)
radam_alph = np.concatenate([alpha_start*np.ones((third_iter+extra_iters)),                             dmid*alpha_start*np.ones((third_iter)),                             dend*alpha_start*np.ones((third_iter))])
radam_alph_2 = np.concatenate([alpha_start_2*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_2*np.ones((third_iter)),                             dend*alpha_start_2*np.ones((third_iter))])
radam_alph_3 = np.concatenate([alpha_start_3*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_3*np.ones((third_iter)),                             dend*alpha_start_3*np.ones((third_iter))])
radam_alph_4 = np.concatenate([alpha_start_4*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_4*np.ones((third_iter)),                             dend*alpha_start_4*np.ones((third_iter))])
radam_alph_5 = np.concatenate([alpha_start_5*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_5*np.ones((third_iter)),                             dend*alpha_start_5*np.ones((third_iter))])


# In[157]:


def cost_func(weights):
    return (1/2)*np.linalg.norm(X_Train_mat.T.dot(weights) - y_train)**2


# In[158]:


def grad_func(weights,batch_inds):
    return (1/len(batch_inds))*X_Train_mat[:,batch_inds].dot(X_Train_mat[:,batch_inds].T.dot(weights) - y_train[batch_inds])
# radam can blow up if step size isn't small enough?


# In[159]:


# run iters
a_w_1,a_c_1,a_mse_1 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start)
a_w_2,a_c_2,a_mse_2 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_2)
a_w_3,a_c_3,a_mse_3 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_3)
a_w_4,a_c_4,a_mse_4 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_4)
a_w_5,a_c_5,a_mse_5 = Adam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_5)


# In[160]:


r_w_1,r_c_1,r_mse_1 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph,ntrain,batch_size)
r_w_2,r_c_2,r_mse_2 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph_2,ntrain,batch_size)
r_w_3,r_c_3,r_mse_3 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph_3,ntrain,batch_size)
r_w_4,r_c_4,r_mse_4 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph_4,ntrain,batch_size)
r_w_5,r_c_5,r_mse_5 = RAdam(cost_func,grad_func,theta_0,niters,radam_alph_5,ntrain,batch_size)


# In[161]:


s_w_1,s_c_1,s_mse_1 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start)
s_w_2,s_c_2,s_mse_2 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_2)
s_w_3,s_c_3,s_mse_3 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_3)
s_w_4,s_c_4,s_mse_4 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_4)
s_w_5,s_c_5,s_mse_5 = SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=alpha_start_5)


# In[162]:


print(a_w_1)
print(r_w_1)
print(s_w_1)


# In[129]:


plt.scatter(x_train,X_Train_mat.T.dot(a_w_1))
plt.scatter(x_train,fx_train)
plt.scatter(x_train,y_train)
plt.show()


# In[130]:


plt.scatter(x_train,X_Train_mat.T.dot(r_w_1))
plt.scatter(x_train,fx_train)
plt.scatter(x_train,y_train)
plt.show()


# In[131]:


plt.scatter(x_train,X_Train_mat.T.dot(s_w_1))
plt.scatter(x_train,fx_train)
plt.scatter(x_train,y_train)
plt.show()


# In[132]:


plt.plot(a_c_1)
plt.plot(r_c_1)
plt.plot(s_c_1)
plt.show()


# In[134]:


print(a_mse_1[-1],r_mse_1[-1],s_mse_1[-1])
print(a_mse_2[-1],r_mse_2[-1],s_mse_2[-1])
print(a_mse_3[-1],r_mse_3[-1],s_mse_3[-1])
print(a_mse_4[-1],r_mse_4[-1],s_mse_4[-1])
print(a_mse_5[-1],r_mse_5[-1],s_mse_5[-1])


# In[111]:


print(a_c_1[0:10])
print(r_c_1[0:10])


# In[107]:


# radam can blow up if step size is too big in first terms
# does not appear to be robust in terms of learning rate


# In[148]:


plt.plot(r_c_1)
plt.plot(r_c_2)
plt.plot(r_c_3)
plt.plot(r_c_4)
plt.plot(r_c_5)
plt.show()


# In[149]:


plt.plot(a_c_1)
plt.plot(a_c_2)
plt.plot(a_c_3)
plt.plot(a_c_4)
plt.plot(a_c_5)
plt.show()


# In[164]:


print(s_c_1[100:110])

