#!/usr/bin/env python
from ipdb import set_trace as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

a0 = np.array(2.2)
mean = np.array([1, 2, 3])
cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
ones = np.ones(3)
# a0 = np.array(0.5)
# mean_df = pd.read_hdf('day_mean.hdf5', 'mean')
# mean = mean_df.values
# cov_df = pd.read_hdf('day_cov.hdf5', 'cov')
# cov = cov_df.values
# ones = np.ones(len(mean))

def loss(theta):
    w, l1, l2 = theta[:-2], theta[-2], theta[-1]
    f = cov @ w - l1 * mean - l2 * ones
    c1 = a0 - w.T @ mean
    c2 = 1. - w.T @ ones
    return f.T @ f + c1 * c1 + c2 * c2


def grad_loss(theta, _):
    w, l1, l2 = theta[:-2], theta[-2], theta[-1]
    f = cov @ w - l1 * mean - l2 * ones
    c1 = a0 - w.T @ mean
    c2 = 1. - w.T @ ones
    # gradient w.r.t. w
    grad_w = 2. * (cov @ f - c1 * mean - c2 * ones)
    # gradient w.r.t. l1
    grad_l1 = - 2. * mean.T @ f
    grad_l2 = - 2. * ones.T @ f
    return np.append(grad_w, [grad_l1, grad_l2])


def SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=0.01):
    # initial values
    cost_out = np.zeros(niters+1)
    cost_out[0] = cost_func(theta_0)
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
    return theta_vec,cost_out


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

    return theta_vec,cost_out


def RAdam(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha = 0.001, beta1 = None,beta2 = None, epsilon = None):

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
                theta_vec = theta_vec - alpha*rectification_term*bias_correct_m/bias_correct_v
            else:
                # update weights
                theta_vec = theta_vec - alpha*bias_correct_m

        # cost function
        cost_out[tt] = cost_func(theta_vec)

    return theta_vec,cost_out

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=1, ncols=3)
    names = ['RAdam', 'Adam', 'SGD']
    optimizers = [RAdam, Adam, SGD]
    for axis, name, optimizer in zip(axes, names, optimizers):
        iters = 10000
        axis.set_title(name)
        axis.set_ylim(top=2.5, bottom=0)
        axis.set_xlim(left=1, right=iters)
        axis.grid(linewidth=0.1)
        for alpha in [0.001]:
            theta_0 = np.zeros(len(mean)+2)
            theta_vec, cost_out = optimizer(loss, grad_loss, theta_0, iters, 1, 1, alpha=alpha)
            print(name)
            print(theta_vec)
            # axis.plot(cost_out, label=alpha)
        axis.legend()

plt.show()
import ipdb; ipdb.set_trace()
pass
