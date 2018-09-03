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
from sklearn.preprocessing import scale
import random

INPUT_DIM = 1
INITIAL = 5
OUTPUT_DIM = 2
MAX_ITER = 200
EPSILON = 10**-6
REF = [10,10]
LEN_SCALE = 0.1

class DataComplex:
    data = np.empty((0,INPUT_DIM+OUTPUT_DIM))
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
    
    
def function(XpR):
    fVal = []
    for val in XpR:
        x = val[0:INPUT_DIM]
        r = val[INPUT_DIM:INPUT_DIM+OUTPUT_DIM]
        fSum = 0
        for i in range(0,OUTPUT_DIM):
            fSum = fSum + (r[i]*f(x,i))
        fVal.append(fSum)
    return np.array(fVal)

def initvals_(bounds):
    print("Initialing Data As:\n")
    gData = np.zeros([INITIAL,INPUT_DIM+OUTPUT_DIM])
    for i in range(0,INITIAL):
        for j in range(0,INPUT_DIM):
            gData[i,j] = random.uniform(bounds['min'][0], bounds['max'][0])
        ranktmp_ = []
        for k in range(INPUT_DIM,INPUT_DIM+OUTPUT_DIM):
            ranktmp_.append(random.uniform(0,1))
        ranktmp_.sort()
        for k in range(INPUT_DIM,INPUT_DIM+OUTPUT_DIM):
            gData[i,k] = ranktmp_[k-INPUT_DIM]
        
    gDataY = function(gData)
    
    return gData,gDataY
    
def extractF(x):
    yOut = np.zeros([len(x),OUTPUT_DIM])
    print(yOut.shape)
    for i in range(yOut.shape[0]):
        for j in range(yOut.shape[1]):
            yOut[i,j] = f(x[i,0:INPUT_DIM],j)
    return yOut
    
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
    
    len_matrix = -np.linalg.pinv(np.identity(INPUT_DIM+OUTPUT_DIM)*LEN_SCALE**2)
    XsT = (Data - xs).T
    tmpI = np.dot(len_matrix,XsT)
    KxsX = Kernels['ker1'].K(xs,Data).T
    tmpII = np.dot(np.linalg.pinv(Kernels['ker1'].K(Data,Data)),np.matrix(yReal[:,0]).T)
    tmpIII = np.multiply(KxsX,tmpII)
    res_ = np.dot(tmpI,tmpIII)
    '''
    print(tmpI.shape)
    print(KxsX.shape)
    print(tmpII.shape)
    print(tmpIII.shape)
    print(res_.shape)
    '''
    return res_
    
def dvt_var(xs,Data,Kernels):
    
    sizeD = Data.shape[0]
    sizeX = xs.shape[1]
    
    len_matrix = np.linalg.inv(np.identity(INPUT_DIM+OUTPUT_DIM)*LEN_SCALE**2)
    #len_matrix = np.linalg.inv(np.identity(OUTPUT_DIM)*LEN_SCALE)
    KXX_m1 = np.linalg.pinv(Kernels['ker1'].K(Data,Data))
    KxsX = Kernels['ker1'].K(xs,Data)
    KXxs = Kernels['ker1'].K(Data,xs)
    X_xs = (Data-xs).T
    
    sumup_ = np.zeros([sizeX,sizeX])
        
    '''
    print("----len_matrix-----")
    print(len_matrix)
    print("----KXX_m1-----")
    print(KXX_m1)
    print("----KxsX-----")
    print(KxsX)
    print("-----KXxs----")
    print(KXxs)
    print("------X_xs---")
    print(X_xs)
    '''

    Alis_ = 0
    for i in range(0,sizeD):
        for j in range(0,sizeD):
            
            tmpI = np.dot(np.dot(np.matrix(X_xs[:,i]).T,np.matrix(X_xs[:,j])),len_matrix**2)
            tmp1 = KXX_m1[i,j]*KxsX[0,i]*KxsX[0,j]
            Alis_ += tmp1*tmpI
    
    
    
    return (len_matrix + Alis_)       
            
         


def cell_point_dom(points,cell):
    
    #print(cell)
    #print()
    #print(points)
    res_ = []
    for i in range(0,len(points)):
        if ((cell[0,0] >= points[i,0]) and (cell[0,1] >= points[i,1])):
            res_.append(i)
    return res_
            
    

def Expected_HVI(points,weights):
    
    grid_,pMap_ = samplePareto(points)
    wPoints = {}
    for val in grid_:
        if (~pMap_[repr(val)]):
            wPoints[repr(val)] = cell_point_dom(points,val)
            
            
    print(wPoints)    
    
    
    
    
    '''
    plt.plot(points[:,0],points[:,1],'.b')
    grid_,dic = samplePareto(points)
    for val in grid_:
        print(val)
        mean_ = np.array([(val[0,0]+val[0,2])/2,(val[0,1]+val[0,3])/2])
        if (dic[repr(val)]):
            print("Hi")
            plt.plot(mean_[0],mean_[1],'*g')
        print(mean_)
        
    plt.show()
    print(grid_.shape,dic)
    '''
    
    
    
    

    
    
    

    
#############################################################
INITIAL = 5
bounds = dict()
bounds  = {'min': [-10],'max':[10]}
xData,yData = initvals_(bounds)
yReal = extractF(xData)
info(xData,yData,yReal)
Kernels = ctKernel(OUTPUT_DIM,INPUT_DIM,LEN_SCALE,LEN_SCALE)


anotherY = mPareto(yReal)
#print(anotherY)
grid_,dic = samplePareto(anotherY)



x = np.array([[0.4,0.3,0.6]])


####
#mod1 = trainModel(xData,np.matrix(yReal[:,0]).T,Kernels['ker0'],400)
#mod2 = trainModel(xData,np.matrix(yReal[:,1]).T,Kernels['ker1'],400)
#mod1 = trainModel(xData,yData,Kernels['ker0'],400)
#mod2 = trainModel(xData,yData,Kernels['ker1'],400)


sWeights = np.random.rand(1,anotherY.shape[0])
Expected_HVI(anotherY,sWeights)
#print(cell_point_dom(anotherY,np.array([[3,3,4,4]])))
#print("**********")
#print(grid_)

#mu_ = (dvt_mu(x,xData,Kernels,yReal))
#print(mu_)
#print(dvt_var(x,xData,Kernels))

