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

INPUT_DIM = 2
OUTPUT_DIM = 1
MAX_ITER = 200
EPSILON = 10**-6

class DataComplex:
    
    data = np.empty((0,INPUT_DIM))
    rankers = np.empty((0,INPUT_DIM))
    def __init__(self, iData,iRank):
        self.data = iData
        self.rankers = iRank
    def newData(self,newPoint):
        self.data = np.append(self.data,[newPoint],axis=0)
    def newRank(self,newRank):
        self.rankers = np.append(self.rankers,[newRank],axis=0)
    

def mPareto(y):

    pFlag = 0
    pSet = np.empty((0,y.shape[1]))
    sortedx = y[:,np.argsort(y[0,:])]
    uSortedx = np.empty((0,y.shape[1]))
    tMin = float('inf')
       
    if np.unique(sortedx[:,0]).shape[0] != sortedx[:,0].shape[0]:
        print('Some points in a row...\nHandling that....\n')
        pFlag = 1
    if pFlag:
        U = np.unique(sortedx[:,0])
        for val in U:
            uSortedx = np.append(uSortedx,np.array([np.min(sortedx[sortedx[:,0] == val],axis=0)]),axis=0)
        for val in uSortedx:
            if (val[1] <= tMin):
                pSet = np.append(pSet,np.array([val]),axis=0)
                tMin = val[1]
        return pSet               
    else:
        for val in sortedx:
            if (val[1] <= tMin):
                pSet = np.append(pSet,np.array([val]),axis=0)
                tMin = val[1]
        return pSet
        
    

#############################################################
X = np.array([[1,2],[2,2.02],[2,1.5],[3,1],[4,0.9],[5,0.5],[2.5,1.8],
              [3.5,1],[1,1.8],[1,1.5],[2.5,1.2],[3,2],[1,1.6],
              [1.5,1.4],[4.5,0.8]])

plt.plot(X[:,0],X[:,1],'ob')


XX = mPareto(X)
plt.plot(XX[:,0],XX[:,1],'*r',markersize=13)
plt.show()

datapointer = DataComplex(X,np.array([X[-1]]))
datapointer.newData(X[-1])
datapointer.newRank(X[-1])
#############################################################