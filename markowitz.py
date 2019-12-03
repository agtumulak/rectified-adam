#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:02:18 2019

@author: bryanrabotnick
"""
import numpy as np
import pandas as pd
import scipy.optimize as sci
import scipy.io as sio
import matplotlib.pyplot as plt
import time
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


# In[5]:


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





df = pd.read_csv('history_60d.csv')
df = df.loc[~df.duplicated(subset=['date', 'symbol']),:] # drop duplicates
stockClose = pd.DataFrame.to_numpy(df.pivot(index='date', columns='symbol', values='close'))[:,:40]
stockOpen = pd.DataFrame.to_numpy(df.pivot(index='date', columns='symbol', values='open'))[:,:40]
dailyReturns = 100*(stockClose - stockOpen)/stockOpen
averageDailyReturns = (np.ones(42)@dailyReturns/42)
deviationMatrix = dailyReturns - averageDailyReturns
covarianceMatrix = np.zeros((40, 40)).astype(float)
for x in range(40):
    for y in range(40):
        xVal = deviationMatrix[:,x]
        yVal = deviationMatrix[:,y]
        covarianceMatrix[x, y] = np.dot(xVal, yVal)/42
        #covarianceMatrix[x, y] /= np.sqrt((np.dot(xVal, xVal)/42))
        #covarianceMatrix[x, y] /= np.sqrt((np.dot(yVal, yVal)/42))
data = pd.read_csv("history_60d.csv")
endData = data.loc[data['date'] == '2019-04-18']
closeData = endData[['close']]
closeData = pd.DataFrame.to_numpy(endData.get("close"))#[:7900]
openData = pd.DataFrame.to_numpy(endData.get("open"))#[:7900]

#endDataAve = data.loc[data['date'] == '2019-03-18']
#closeDataAve = endDataAve[['close']]
#closeDataAve = pd.DataFrame.to_numpy(endDataAve.get("close"))[:7900]
#fakeMeanVec = (100*(closeData - closeDataAve)/closeDataAve)
#percent rate of return
#rateOfReturnVector = 100*(closeData - openData)/openData
#rateOfReturnVector = rateOfReturnVector[:10]
#mean = np.dot(np.ones(rateOfReturnVector.size),rateOfReturnVector)/rateOfReturnVector.size
#covariance = np.matmul((rateOfReturnVector - mean).T, (rateOfReturnVector - mean))/mean
#c = rateOfReturnVector - mean #THIS IS THE WRONG MEAN. WILL NEED TO DEFINE A LOOP TO MAKE A NEW VECTOR TO GO THROUGH TIME SERIES AND FIND AVERAGE RETURN
##cmean = np.dot(np.ones(c.size),c)/c
#covariance = np.outer(c, c)/rateOfReturnVector.size
rateOfReturnVector = averageDailyReturns
covariance = covarianceMatrix
portfolioWeights = np.ones(rateOfReturnVector.size)/rateOfReturnVector.size
#portfolioWeights[.05, .05, .05, .05, .05, .05, .2, .05, .05, .4]
#lossFunction = (1/2)*(portfolioWeights.T@covariance@portfolioWeights)
def loss(w):
    return (1/2)*(w.T@covariance@w)
def c1(w):
    return w.T@np.ones(w.size) - 1
a0 = 25563
def c2(w):
    return w.T@rateOfReturnVector - a0
constraint = [{'type':'eq', 'fun': c1},{'type':'eq', 'fun': c2}]
#gradient = covariance@weights
#niters = 50000
y = sci.minimize(loss,portfolioWeights,constraints = constraint)
y = y['x']
print(np.ones(portfolioWeights.size)@y)
print(y@rateOfReturnVector)

a = rateOfReturnVector.T@np.linalg.inv(covariance)@rateOfReturnVector
b = rateOfReturnVector.T@np.linalg.inv(covariance)@np.ones(rateOfReturnVector.size)
cc = np.ones(rateOfReturnVector.size).T@np.linalg.inv(covariance)@np.ones(rateOfReturnVector.size)
matPreInv = np.arange(4).astype(float)
matPreInv = [a, b, b, cc]
matPreInv = np.reshape(matPreInv, (2, 2))
matPostInv = np.linalg.inv(matPreInv)
mulGenMat = [a0, 1]
lagrangeMultiplierVector = matPostInv@mulGenMat
#weightOutput, cost = Adam(cost_use, grad_use, portfolioWeights, niters)
#optimalWeightsPerhaps = 
part1 = np.linalg.inv(covariance)@rateOfReturnVector
part1 *= lagrangeMultiplierVector[0]
part2 = np.linalg.inv(covariance)@np.ones(rateOfReturnVector.size)
part2 *= lagrangeMultiplierVector[1]
optimalWeights = part1 + part2

print(np.ones(optimalWeights.size)@optimalWeights)
print(optimalWeights@rateOfReturnVector)
def loss2(weights):
    #returnVal = 0
    originalComponent = (1/2)*(weights.T@covariance@weights)
    l1Component = (a0 - weights@rateOfReturnVector)
    l1Component *= lagrangeMultiplierVector[0]
    l2Component = (1 - weights@np.ones(weights.size))
    l2Component *= lagrangeMultiplierVector[1]
    return (originalComponent + l1Component + l2Component)
    #return (1/2)*(weights.T@covariance@weights) + lagrangeMultiplierVector[0]*(a0 - weights@rateOfReturnVector) + lagrangeMultiplierVector[1]*(1 - weights@np.ones(weights.size))

portfolioWeights2 = np.ones(rateOfReturnVector.size)/rateOfReturnVector.size
h = sci.minimize(loss2,portfolioWeights2)
h = h['x']
print(np.ones(h.size)@h)
print(h@rateOfReturnVector)
#portfolioWeights3 = np.ones(rateOfReturnVector.size)/rateOfReturnVector.size
#niters = 5000
#def cost_use(weights):
#    
#    return (1/2)*(weights.T@covariance@weights) + lagrangeMultiplierVector[0]*(a0 - weights@rateOfReturnVector) + lagrangeMultiplierVector[1]*(1 - weights@np.ones(weights.size))
#def grad_use(weights):
#    return covariance@weights
#w, cost = Adam(cost_use, grad_use, portfolioWeights3, niters)
#def loss2(weights):
#    return (1/2)*(weights.T@covariance@weights) - lagrangeMultiplierVector[0]*(a0 - weights@rateOfReturnVector) - lagrangeMultiplierVector[1]*(1 - weights@np.ones(weights.size))
#print(np.ones(w.size)@w)
#print(w@rateOfReturnVector)

#startData = data.loc[data['date'] == '2019-02-20']
#justSymbolsEndDate = endData['symbol']
#justSymbolsStartDate = startData['symbol']
#intersection = pd.Series(np.intersect1d(justSymbolsEndDate,justSymbolsStartDate))

