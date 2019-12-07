
# coding: utf-8

# In[6]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from sklearn.datasets.samples_generator import make_classification


# In[7]:


ntrain = 2000
ntest = 1000
batch_size = 128
X,y =  make_classification(n_samples=ntrain+ntest, n_features=500, random_state=0)
y[y==0] = -1
Xtrain = X[:ntrain,:].T
Xtest = X[ntrain:,:].T
ytrain = y[:ntrain]
ytest = y[ntrain:]

wLS = np.linalg.inv(Xtrain.dot(Xtrain.T)).dot(Xtrain.dot(ytrain))


# In[8]:


def test_accuracy(weights):
    output_labels = np.sign(Xtest.T.dot(weights))
    numcorrect = np.sum(output_labels == ytest)/np.shape(ytest)[0]
    return numcorrect


# In[9]:


print(test_accuracy(wLS))


# In[10]:


def sigmoid(x): # function for calculating exp(x)/(1+exp(x))
    z = np.exp(x)
    ans = z/(1+z)
    if np.isnan(ans): # used to make sure that really large abs(x) do not give nan answer
        if x > 0:
            return 1
        else:
            return 0
    else:
        return ans


# In[11]:


def grad_log(theta,batch_inds): # compute gradient for given theta
    regularize = 10
    grad = np.zeros((np.shape(Xtrain)[0]))
    regvec = 2*regularize*theta

    for aa in batch_inds:
        grad -= (ytrain[aa]*sigmoid(-ytrain[aa]*theta.T.dot(Xtrain[:,aa])))*Xtrain[:,aa]

    grad += regvec
    return grad


# In[12]:


def evaluate_cost_function(theta): # function to evaluate cost function at given theta
    regularize = 10
    temp = 0
    
    normterm = regularize*(theta.T.dot(theta))

    for aa in range(0,ntrain):
        temp += np.log(1+np.exp(-ytrain[aa]*theta.T.dot(Xtrain[:,aa])))

    return temp+normterm


# In[13]:


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
    
    accuracy_out = np.zeros(niters+1)
    accuracy_out[0] = test_accuracy(theta_0)
    
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
        accuracy_out[tt] = test_accuracy(theta_vec)
    
    return theta_vec,cost_out,accuracy_out     


# In[14]:


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
    accuracy_out = np.zeros(niters+1)
    cost_out[0] = cost_func(theta_0)
    accuracy_out[0] = test_accuracy(theta_0)
    
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
        accuracy_out[tt] = test_accuracy(theta_vec)
    
    return theta_vec,cost_out,accuracy_out


# In[15]:


# normal SGD
def SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha = None):
    
    # check parameters
    if alpha == None:
        alpha = 0.001
    
    # initial values
    cost_out = np.zeros(niters+1)
    cost_out[0] = cost_func(theta_0)
    accuracy_out = np.zeros(niters+1)
    accuracy_out[0] = test_accuracy(theta_0)
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
        accuracy_out[tt] = test_accuracy(theta_vec)
    
    return theta_vec,cost_out,accuracy_out        


# In[16]:


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
theta_0 = xxx*np.ones((Xtrain.shape[0]))
niters = 200
third_iter = np.int(np.floor(niters/3))
extra_iters = np.int(niters - third_iter*3)
radam_alph = np.concatenate([alpha_start*np.ones((third_iter+extra_iters)),                             dmid*alpha_start*np.ones((third_iter)),                             dend*alpha_start*np.ones((third_iter))])
radam_alph_2 = np.concatenate([alpha_start_2*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_2*np.ones((third_iter)),                             dend*alpha_start_2*np.ones((third_iter))])
radam_alph_3 = np.concatenate([alpha_start_3*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_3*np.ones((third_iter)),                             dend*alpha_start_3*np.ones((third_iter))])
radam_alph_4 = np.concatenate([alpha_start_4*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_4*np.ones((third_iter)),                             dend*alpha_start_4*np.ones((third_iter))])
radam_alph_5 = np.concatenate([alpha_start_5*np.ones((third_iter+extra_iters)),                             dmid*alpha_start_5*np.ones((third_iter)),                             dend*alpha_start_5*np.ones((third_iter))])


# In[17]:


# run iters
a_w_1,a_c_1,a_mse_1 = Adam(evaluate_cost_function,grad_log,theta_0,niters,ntrain,batch_size,alpha=alpha_start)
a_w_3,a_c_3,a_mse_3 = Adam(evaluate_cost_function,grad_log,theta_0,niters,ntrain,batch_size,alpha=alpha_start_3)
a_w_4,a_c_4,a_mse_4 = Adam(evaluate_cost_function,grad_log,theta_0,niters,ntrain,batch_size,alpha=alpha_start_4)
a_w_5,a_c_5,a_mse_5 = Adam(evaluate_cost_function,grad_log,theta_0,niters,ntrain,batch_size,alpha=alpha_start_5)


# In[18]:


r_w_1,r_c_1,r_mse_1 = RAdam(evaluate_cost_function,grad_log,theta_0,niters,radam_alph,ntrain,batch_size)
r_w_3,r_c_3,r_mse_3 = RAdam(evaluate_cost_function,grad_log,theta_0,niters,radam_alph_3,ntrain,batch_size)
r_w_4,r_c_4,r_mse_4 = RAdam(evaluate_cost_function,grad_log,theta_0,niters,radam_alph_4,ntrain,batch_size)
r_w_5,r_c_5,r_mse_5 = RAdam(evaluate_cost_function,grad_log,theta_0,niters,radam_alph_5,ntrain,batch_size)


# In[19]:


s_w_1,s_c_1,s_mse_1 = SGD(evaluate_cost_function,grad_log,theta_0,niters,ntrain,batch_size,alpha=alpha_start)
s_w_3,s_c_3,s_mse_3 = SGD(evaluate_cost_function,grad_log,theta_0,niters,ntrain,batch_size,alpha=alpha_start_3)
s_w_4,s_c_4,s_mse_4 = SGD(evaluate_cost_function,grad_log,theta_0,niters,ntrain,batch_size,alpha=alpha_start_4)
s_w_5,s_c_5,s_mse_5 = SGD(evaluate_cost_function,grad_log,theta_0,niters,ntrain,batch_size,alpha=alpha_start_5)


# In[20]:


print(max(abs(a_w_1 - wLS)))
print(max(abs(r_w_1 - wLS)))
print(max(abs(s_w_1 - wLS)))


# In[21]:


plt.plot(a_c_1,label='adam')
plt.plot(r_c_1,label='radam')
plt.plot(s_c_1,label='sgd')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('cost vs epoch for α = {}'.format(alpha_start))
plt.legend()
plt.show()


# In[32]:


print(a_c_1[-1],r_c_1[-1],s_c_1[-1])
print(alpha_start,a_mse_1[-1],r_mse_1[-1],s_mse_1[-1])
print(alpha_start_3,a_mse_3[-1],r_mse_3[-1],s_mse_3[-1])
print(alpha_start_4,a_mse_4[-1],r_mse_4[-1],s_mse_4[-1])
print(alpha_start_5,a_mse_5[-1],r_mse_5[-1],s_mse_5[-1])


# In[23]:


plt.plot(r_c_1,label='α = {}'.format(alpha_start))
#plt.plot(r_c_2,label='α = {}'.format(alpha_start_2))
plt.plot(r_c_3,label='α = {}'.format(alpha_start_3))
plt.plot(r_c_4,label='α = {}'.format(alpha_start_4))
plt.plot(r_c_5,label='α = {}'.format(alpha_start_5))
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('RAdam cost vs epoch')
plt.legend()
plt.show()


# In[24]:


plt.plot(a_c_1,label='α = {}'.format(alpha_start))
#plt.plot(a_c_2,label='α = {}'.format(alpha_start_2))
plt.plot(a_c_3,label='α = {}'.format(alpha_start_3))
plt.plot(a_c_4,label='α = {}'.format(alpha_start_4))
plt.plot(a_c_5,label='α = {}'.format(alpha_start_5))
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Adam cost vs epoch')
plt.show()


# In[25]:


plt.plot(s_c_1,label='α = {}'.format(alpha_start))
#plt.plot(a_c_2,label='α = {}'.format(alpha_start_2))
plt.plot(s_c_3,label='α = {}'.format(alpha_start_3))
plt.plot(s_c_4,label='α = {}'.format(alpha_start_4))
plt.plot(s_c_5,label='α = {}'.format(alpha_start_5))
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('SGD cost vs epoch')
plt.show()


# In[26]:


plt.plot(r_mse_1,label='α = {}'.format(alpha_start))
#plt.plot(r_mse_2,label='α = {}'.format(alpha_start_2))
plt.plot(r_mse_3,label='α = {}'.format(alpha_start_3))
plt.plot(r_mse_4,label='α = {}'.format(alpha_start_4))
plt.plot(r_mse_5,label='α = {}'.format(alpha_start_5))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.title('RAdam test accuracy vs epoch')
plt.ylim([0.5,1])
plt.show()


# In[27]:


plt.plot(a_mse_1,label='α = {}'.format(alpha_start))
#plt.plot(a_mse_2,label='α = {}'.format(alpha_start_2))
plt.plot(a_mse_3,label='α = {}'.format(alpha_start_3))
plt.plot(a_mse_4,label='α = {}'.format(alpha_start_4))
plt.plot(a_mse_5,label='α = {}'.format(alpha_start_5))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.title('Adam test accuracy vs epoch')
plt.ylim([0.5,1])
plt.show()


# In[28]:


plt.plot(s_mse_1,label='α = {}'.format(alpha_start))
#plt.plot(r_mse_2,label='α = {}'.format(alpha_start_2))
plt.plot(s_mse_3,label='α = {}'.format(alpha_start_3))
plt.plot(s_mse_4,label='α = {}'.format(alpha_start_4))
plt.plot(s_mse_5,label='α = {}'.format(alpha_start_5))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.title('SGD test accuracy vs epoch')
plt.ylim([0.5,1])
plt.show()

