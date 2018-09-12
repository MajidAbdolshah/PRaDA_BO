from __future__ import division
import GPy
from scipy.stats import multivariate_normal
from scipy.stats import mvn
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
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal
import random

INPUT_DIM = 1
INITIAL = 5
OUTPUT_DIM = 2
MAX_ITER = 200
EPSILON = 10**-6
REF = [10,10]
LEN_SCALE = 0.1
EST_RANGE = 100

class DataComplex:
    data = np.empty((0,INPUT_DIM))
    outputs = np.empty((0,OUTPUT_DIM))
    def __init__(self, iData,iOut):
        self.data = iData
        self.outputs = iOut
    def newData(self,newPoint):
        self.data = np.append(self.data,[newPoint],axis=0)
    def newOut(self,newPoint):
        self.outputs = np.append(self.outputs,[newPoint],axis=0)
    
def mPareto(y):
    pFlag = 0
    pSet = np.empty((0,y.shape[1]))
    sortedx = y[y[:,0].argsort()]
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
        
def f(x,fNum):
    def firstfun(x):
        return x**2
    def secondfun(x):
        return (x-2)**2
    options = {0 : firstfun,
               1 : secondfun,}
    return options[fNum](x)
    
    
def function(Xp):
    tmp_1 = f(Xp,0)
    tmp_2 = f(Xp,1)
    res_ = np.hstack((tmp_1,tmp_2))
    return res_

    
    
    

## CheckedOK
def initvals_(bounds):
    print("Initialing Data As:\n")
    gData = np.zeros([INITIAL,INPUT_DIM])
    for i in range(0,INITIAL):
        for j in range(0,INPUT_DIM):
            gData[i,j] = random.uniform(bounds['min'][0], bounds['max'][0])
    gDataY = function(gData)
    return gData,gDataY
    
    
        
    
    
def info(*arg):
    for i in range(len(arg)):
        print("Shape of "+str(i)+" is "+str(arg[i].shape))


def ctKernel(no,dInput,*arg):
    names_ = {}
    for i in range(no):
        text_ = "ker" + str(i)
        names_[text_] = GPy.kern.RBF(input_dim=dInput,lengthscale=arg[i],ARD=False)+GPy.kern.White(input_dim=dInput)
    return names_

        
def trainModel(X,Y,Ker,eVal):
    model_ = GPy.models.GPRegression(X,Y,Ker)
    model_.optimize(max_f_eval = eVal)    
    return model_

def testModel(model_,x):
    [mu_per,sig_per] = model_.predict(x,full_cov=1)
    print(mu_per,sig_per)
    return mu_per[0,0],sig_per[0,0]
    
def samplePareto(par):
    infoDic = {}
    uPartsX = np.sort(par[:,0])
    uMapX = np.concatenate([[0],uPartsX,[REF[0]]])
    uPartsY = np.sort(par[:,1])
    uMapY = np.concatenate([[0],uPartsY,[REF[1]]])
    
    cells_ = np.empty((0,OUTPUT_DIM*2))
    for i in range(0,len(uMapX)-1):
        for j in range(0,len(uMapY)-1):
            
            pos_st = np.array([uMapX[i],uMapY[j]])
            pos_en = np.array([uMapX[i+1],uMapY[j+1]])
            
            pos_ = np.matrix(np.append(pos_st,pos_en,axis=0))
            mid_point = np.array([(pos_[0,0]+pos_[0,2])/2,
                                  (pos_[0,1]+pos_[0,3])/2])
            cells_ = np.vstack([cells_,pos_])
            infoDic[repr(pos_)] = ruPareto(mid_point,par)
    return cells_,infoDic
            
    
def ruPareto(x,par):
    for val in par:
        if(x[0]>val[0] and x[1]>val[1]):
            return False
    return True
    
def dvt_mu(xs,Data,Kernels,yReal):
    
    len_matrix = -np.linalg.pinv(np.identity(INPUT_DIM)*LEN_SCALE**2)
    XsT = (Data - xs).T
    tmpI = np.dot(len_matrix,XsT)
    KxsX = Kernels['ker1'].K(xs,Data).T
    tmpII = np.dot(np.linalg.pinv(Kernels['ker1'].K(Data,Data)),np.matrix(yReal[:,0]).T)
    tmpIII = np.multiply(KxsX,tmpII)
    res_ = np.dot(tmpI,tmpIII)
    
    print(tmpI.shape)
    print(KxsX.shape)
    print(tmpII.shape)
    print(tmpIII.shape)
    print(res_.shape)

    return res_
    
def dvt_var(xs,Data,Kernels):
    
    sizeD = Data.shape[0]
    sizeX = xs.shape[1]
    
    len_matrix = np.linalg.inv(np.identity(INPUT_DIM)*LEN_SCALE**2)
    #len_matrix = np.linalg.inv(np.identity(OUTPUT_DIM)*LEN_SCALE)
    KXX_m1 = np.linalg.pinv(Kernels['ker1'].K(Data,Data))
    KxsX = Kernels['ker1'].K(xs,Data)
    KXxs = Kernels['ker1'].K(Data,xs)
    X_xs = (Data-xs).T
    
    sumup_ = np.zeros([sizeX,sizeX])


    Alis_ = 0
    for i in range(0,sizeD):
        for j in range(0,sizeD):
            
            tmpI = np.dot(np.dot(np.matrix(X_xs[:,i]).T,np.matrix(X_xs[:,j])),len_matrix**2)
            tmp1 = KXX_m1[i,j]*KxsX[0,i]*KxsX[0,j]
            Alis_ += tmp1*tmpI
    
    return (len_matrix + Alis_)       
            
def Helpme_(points):
    grid_,dic = samplePareto(points)
    for val in grid_:
        if np.invert(dic[repr(val)]):
            plt.plot(val[0,2],val[0,3],'*r',markersize=20)
    plt.plot(anotherY[:,0],anotherY[:,1],'*b',markersize=15)
    plt.show()        

def cell_point_dom(points,cell):

    res_ = []
    for i in range(0,len(points)):
        if ((cell[0,0] >= points[i,0]) and (cell[0,1] >= points[i,1])):
            res_.append(i)
    return res_
            
def Expected_HVI(points,weights):
    
    grid_,pMap_ = samplePareto(points)
    fSum = 0
    wPoints = {}
    Weightsdic = {}
    for val in grid_:
        if np.invert(pMap_[repr(val)]):
            wPoints[repr(val)] = cell_point_dom(points,val)
            if (len(wPoints[repr(val)])):
                exWeights = 1 - np.prod(np.take(weights,wPoints[repr(val)]))
            else:
                exWeights = 1
            #print(exWeights)MultiplyB
            
            fSum += exWeights*(val[0,3] - val[0,1])*(val[0,2] - val[0,0])
            Weightsdic[repr(val)] = exWeights       
    #print(wPoints)
    return fSum,Weightsdic
    

def findWeight(model,dim_):
    cols_ = 10
    start_ = 0
    end_ = EST_RANGE
    sum_up = 0
    
    for i in range(1,end_):
        tmp_1 = np.arange(start_,i,i/cols_)
        tmp_2 = np.empty(cols_)
        tmp_2.fill(i)
        points_ = np.array(list(zip(tmp_1,tmp_2)))
        print(points_)
        #print(mvn.pdf(points_))
    

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

INITIAL = 60
bounds = dict()
bounds  = {'min': [-10],'max':[10]}
xData,yData = initvals_(bounds)
info(xData,yData)
Kernels = ctKernel(OUTPUT_DIM,INPUT_DIM,LEN_SCALE,LEN_SCALE)
anotherY = mPareto(yData)
grid_,dic = samplePareto(anotherY)
Helpme_(anotherY)
x = np.array([[0.4]])

mod1 = trainModel(xData,np.matrix(yData[:,0]).T,Kernels['ker0'],400)
mod2 = trainModel(xData,np.matrix(yData[:,1]).T,Kernels['ker1'],400)
[mu_,sigma_] = testModel(mod1,x)
[mu__,sigma__] = testModel(mod2,x)

print(dvt_mu(x,xData,Kernels,yData))
print(dvt_var(x,xData,Kernels))

'''

x = np.array([[0.4,0.3,0.6]])


sWeights = np.random.rand(1,anotherY.shape[0])
EHVI,WI = Expected_HVI(anotherY,sWeights)
print("Expected Hypervolume Improvement: {}".format(EHVI))

mu_ = np.array(dvt_mu(x,xData,Kernels,yReal)).reshape(3,)
cov_ = dvt_var(x,xData,Kernels)
print(mu_)
print(cov_)


mvn = multivariate_normal(mu_,cov_) #create a multivariate Gaussian object with specified mean and covariance matrix
p = findWeight(mvn,2)
print(p)
#evaluate the probability density at x
'''