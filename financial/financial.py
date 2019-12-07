#!/usr/bin/env python
from ipdb import set_trace as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# a0 = np.array(2.2)
# mean = np.array([1, 2, 3])
# cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# ones = np.ones(3)
a0 = np.array(0.0005)
mean_df = pd.read_hdf('day_mean.hdf5', 'mean')
mean = mean_df.values
cov_df = pd.read_hdf('day_cov.hdf5', 'cov')
cov = cov_df.values
ones = np.ones(len(mean))

indices = np.argsort(np.diag(cov))[:-100]
mean = mean[indices]
cov = cov[np.ix_(indices, indices)]
ones = np.ones(len(mean))

inv_cov = np.linalg.inv(cov)
a = mean.T @ inv_cov @ mean
b = mean.T @ inv_cov @ ones
c = ones.T @ inv_cov @ ones
constraints = np.linalg.inv(np.array([[a, b], [b, c]])) @ np.array([a0, 1])
l1_0, l2_0 = constraints[0], constraints[1]
w_0 = l1_0 * inv_cov @ mean + l2_0 * inv_cov @ ones
print(l1_0)
print(l2_0)
st()
pass

def loss(theta):
    w, th1, th2 = theta[:-2], theta[-2], theta[-1]
    f = cov @ w - np.exp(th1) * mean - np.exp(th2) * ones
    c1 = a0 - w.T @ mean
    c2 = 1. - w.T @ ones
    return f.T @ f + c1 * c1 + c2 * c2


def grad_loss(theta, _):
    w, th1, th2 = theta[:-2], theta[-2], theta[-1]
    f = cov @ w - np.exp(th1) * mean - np.exp(th2) * ones
    c1 = a0 - w.T @ mean
    c2 = 1. - w.T @ ones
    # gradient w.r.t. w
    grad_w = 2. * (cov @ f - c1 * mean - c2 * ones)
    # gradient w.r.t. l1
    grad_th1 = - 2. * np.exp(th1) * mean.T @ f
    grad_th2 = - 2. * np.exp(th2) * ones.T @ f
    return np.append(grad_w, [grad_th1, grad_th2])


def SGD(cost_func,grad_func,theta_0,niters,ntrain,batch_size,alpha=0.01):
    # initial values
    cost_out = np.zeros(niters+1)
    cost_out[0] = cost_func(theta_0)
    theta_vec = np.zeros(theta_0.shape)
    theta_vec = theta_0
    rss_out = np.zeros(niters+1)
    rss_out[0] = np.linalg.norm(w_0 - theta_0[:-2])
    l1_out, l2_out = np.zeros(niters+1), np.zeros(niters+1)
    l1_out[0], l2_out[0] = np.exp(theta_0[-2]), np.exp(theta_0[-1])
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

        # rss
        rss_out[tt] = np.linalg.norm(w_0 - theta_vec[:-2])

        # dual variables
        l1_out[tt], l2_out[tt] = np.exp(theta_vec[-2]), np.exp(theta_vec[-1])

    return theta_vec, cost_out, rss_out, l1_out, l2_out


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
    rss_out = np.zeros(niters+1)
    rss_out[0] = np.linalg.norm(w_0 - theta_0[:-2])
    l1_out, l2_out = np.zeros(niters+1), np.zeros(niters+1)
    l1_out[0], l2_out[0] = np.exp(theta_0[-2]), np.exp(theta_0[-1])

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

        # rss
        rss_out[tt] = np.linalg.norm(w_0 - theta_vec[:-2])

        # dual variables
        l1_out[tt], l2_out[tt] = np.exp(theta_vec[-2]), np.exp(theta_vec[-1])

    return theta_vec,cost_out, rss_out, l1_out, l2_out


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

    rss_out = np.zeros(niters+1)
    rss_out[0] = np.linalg.norm(w_0 - theta_0[:-2])

    l1_out, l2_out = np.zeros(niters+1), np.zeros(niters+1)
    l1_out[0], l2_out[0] = np.exp(theta_0[-2]), np.exp(theta_0[-1])

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

        # rss
        rss_out[tt] = np.linalg.norm(w_0 - theta_vec[:-2])

        # dual variables
        l1_out[tt], l2_out[tt] = np.exp(theta_vec[-2]), np.exp(theta_vec[-1])

    return theta_vec,cost_out, rss_out, l1_out, l2_out

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True, figsize=(10,8))
    names = ['RAdam', 'Adam', 'SGD']
    optimizers = [RAdam, Adam, SGD]
    alphas = [0.003, 0.001, 0.0003]
    for col, (rss_axis, l1_axis, l2_axis, loss_axis, name, optimizer) in enumerate(zip(
            axes[0], axes[1], axes[2], axes[3], names, optimizers)):
        iters = 50000
        if col == 0:
            rss_axis.set_ylabel(r'$|| w_{t} - w^{*} ||_{2}$')
            l1_axis.set_ylabel(r'$\lambda_{1}$')
            l2_axis.set_ylabel(r'$\lambda_{2}$')
            loss_axis.set_ylabel('Loss')
        rss_axis.set_title(name)
        rss_axis.set_ylim(top=2.0, bottom=0)
        rss_axis.set_xlim(left=1, right=iters)
        rss_axis.grid(linewidth=0.1)
        l1_axis.grid(linewidth=0.1)
        l1_axis.set_ylim(top=0.1, bottom=-0.1)
        l2_axis.grid(linewidth=0.1)
        l2_axis.set_ylim(top=0.1, bottom=-0.1)
        loss_axis.set_ylim(top=1.0, bottom=0.0)
        loss_axis.set_xlim(left=1, right=iters)
        loss_axis.grid(linewidth=0.1)
        loss_axis.set_xlabel('Epoch')
        for alpha in alphas:
            theta_0 = np.ones(len(mean)+2) / len(mean)
            theta_0[-1] = np.log(0.1)
            theta_0[-2] = np.log(0.1)
            theta_vec, cost_out, rss, l1, l2 = optimizer(loss, grad_loss, theta_0, iters, 1, 1, alpha=alpha)
            rss_axis.plot(rss, label=alpha, linewidth=0.2)
            l1_axis.plot(l1, label=alpha, linewidth=0.2)
            l1_axis.plot(np.full(len(l1), l1_0), color='k', linestyle=':', linewidth=0.2)
            l2_axis.plot(l2, label=alpha, linewidth=0.2)
            l2_axis.plot(np.full(len(l2), l2_0), color='k', linestyle=':', linewidth=0.2)
            loss_axis.plot(cost_out, label=alpha, linewidth=0.2)
            weights = theta_vec[:-2]
            print(np.exp(theta_vec[-2]), np.exp(theta_vec[-1]), weights.sum(), weights.T @ mean)
        rss_axis.legend()
        l1_axis.legend()
        l2_axis.legend()
        loss_axis.legend()
plt.tight_layout()
plt.savefig('markowitz.pdf')
