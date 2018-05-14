from __future__ import division
import GPy
import GPyOpt
import numpy as np 
import os.path
from termcolor import colored
from numpy import linalg as LA
import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import math
from scipy.stats import norm
import GPy
import pprint 
from scipy.spatial import ConvexHull
from tqdm import *
import time
import random

FDIM = 2
MAX_ITER = 200
COUNTER = 0
MIN = -5
MAX = 5
EPS = 10**-6
RANKER_SET = np.array([[0.5,0.5]])
logDicx = dict();
logDicy = dict();
Informed_X = [] 
Informed_Y = []
logger_compare = []
logger_f = []

def findPareto(y,ch):
    if ch == 'min':
        sortedx = y[:,np.argsort(y[0,:])]
        sizex = sortedx.shape
        tempmin = float('inf')
        paretoset = np.empty((0,sizex[0]), float)
        for i in range(sizex[1]):
            if ((sortedx[1,i]) < tempmin):
                paretoset = np.append(paretoset, np.array([sortedx[:,i]]), axis=0)
                tempmin = sortedx[1,i]
    return paretoset.T

def compareRanks(compare,with_):
    global logger
    sizew = with_.shape[0]
    disMat = np.zeros(shape=(1,sizew))
    for j in range(0,sizew):
        disMat[0,j] = LA.norm(compare - with_[j,:])
    
    return (disMat.min())

def showMe(x):
    
    global Informed_Y,Informed_X
    np.array(x)
    def Mf(xx):
        f1 = (xx[0,0])**2
        f2 = (xx[0,0]-2)**2
        return f1,f2
    
    [f1,f2] = Mf(x)   
    Informed_X.append([x[0,0],x[0,1],x[0,2]])
    Informed_Y.append([f1,f2])
    B_Informed_Y = np.array(Informed_Y)
    PY = findPareto(B_Informed_Y.T,'min')
    print(PY.shape[1] / B_Informed_Y.shape[0])
    #plt.plot(B_Informed_Y[:,0],B_Informed_Y[:,1],'ob',markersize=12)
    plt.plot(PY[0,:],PY[1,:],'*y',markersize=7)
    plt.show()
    
def f(x):

    showMe(x) 
    global COUNTER,RANKER_SET;
    #x[0,0] = (x[0,0]+MAX)/(MAX-MIN)
    #x[0,0] = (x[0,0])/(MAX-MIN)
    x[0,0] = x[0,0]/MAX
    x[0,1] = x[0,1]/(x[0,1]+x[0,2])
    x[0,2] = x[0,2]/(x[0,1]+x[0,2])
    
    f1 = (x[0,0])**2
    f2 = (x[0,0]-2)**2
    
    lC = (compareRanks(x[0,1:3],RANKER_SET))
    if lC<EPS:
        lC = EPS
    logger_f.append(x[0,1]*f1 + x[0,2]*f2)
    logger_compare.append(1/lC)
    #Result = (x[0,1]*f1 + x[0,2]*f2)/max(logger_f) + (1/lC)/max(logger_compare)
    Result = (x[0,1]*f1 + x[0,2]*f2) + (1/lC)/max(logger_compare)
    RANKER_SET = np.vstack((RANKER_SET,np.array([x[0,1],x[0,2]])))
    logDicx[COUNTER] = x;
    logDicy[COUNTER] = Result;

    print('----------------------------------------\n')
    print('Result:  ', Result)
    print('\n Best X: ', logDicx[min(logDicy, key=logDicy.get)])
    print('\n Best Y: ', logDicy[min(logDicy, key=logDicy.get)])
    print('----------------------------------------\n')
    
    COUNTER = COUNTER + 1
    return Result
    
    
bounds = [
    {'name': 'x0', 'type': 'continuous', 'domain': (MIN,MAX)},
    {'name': 'x1', 'type': 'continuous', 'domain': (0.5,1.0)},
    {'name': 'x2', 'type': 'continuous', 'domain': (0.0,0.5)}
]

ker2 = GPy.kern.RBF(input_dim=3, lengthscale=0.1, ARD=True)+GPy.kern.White(input_dim=3)

myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, 
                                             kernel=ker2, 
                                             model_type = 'GP',
                                             acquisition_type = 'EI')

myBopt.run_optimization(max_iter=400)


print(myBopt.x_opt) 
print(myBopt.fx_opt) 
myBopt.plot_acquisition()