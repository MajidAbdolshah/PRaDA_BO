from __future__ import division
from scipy.stats import norm
import warnings
from copy import deepcopy
import GPy
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
from tqdm import tqdm
import sys

INPUT_DIM = 1
INITIAL = 5
OUTPUT_DIM = 2
MAX_ITER = 200
EPSILON = 10**-6
REF = [10,10]
LEN_SCALE = 0.5
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

def print_dots(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def print_fancy(string,val):
    print_dots(string)
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print()

def mPareto(y):
    pFlag = 0
    pSet = np.empty((0,y.shape[1]))
    sortedx = y[y[:,0].argsort()]
    uSortedx = np.empty((0,y.shape[1]))
    tMin = float('inf')

    if np.unique(sortedx[:,0]).shape[0] != sortedx[:,0].shape[0]:
        #print('Some points in a row...\nHandling that....\n')
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

def parY_X(y,ypar):

    size_y = y.shape[0]
    size_yp = ypar.shape[0]
    indx = []
    for i in range(size_y):
        for j in range(size_yp):
            if (y[i,0] == ypar[j,0] and y[i,1] == ypar[j,1]):
                indx.append(i)
    return indx

def findXpareto(xData,yData,yPareto):
    indx = parY_X(yData,yPareto)
    Xres = np.empty([INPUT_DIM,])
    for val in indx:
        Xres = np.vstack((Xres,xData[val,:]))

    return Xres[1:]

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

def initvals_(bounds,INITIAL):
    str_ = "Initializing " + str(INITIAL) + " Data"
    print_fancy(str_,0.1)
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
    print_fancy('Initializing Kernel',0.1)
    names_ = {}
    for i in range(no):
        text_ = "ker" + str(i)
        names_[text_] = GPy.kern.RBF(input_dim=dInput,variance=1.5,lengthscale=arg[i],ARD=False)#+GPy.kern.White(input_dim=dInput)
    return names_

def trainModel(X,Y,Ker,eVal):
    global Kernels
    model_ = GPy.models.GPRegression(X,Y,Kernels[Ker])
    #model_.optimize_restarts(num_restarts = eVal)
    #print('______________________________')
    #print(Kernels['ker0'].variance)
    #model_.optimize(max_f_eval = eVal)
    return model_,Kernels[Ker]

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
    Jparet_ = np.empty((0,OUTPUT_DIM*2))
    not_Jparet_ = np.empty((0,OUTPUT_DIM*2))
    for i in range(0,len(uMapX)-1):
        for j in range(0,len(uMapY)-1):

            pos_st = np.array([uMapX[i],uMapY[j]])
            pos_en = np.array([uMapX[i+1],uMapY[j+1]])

            pos_ = np.matrix(np.append(pos_st,pos_en,axis=0))
            mid_point = np.array([(pos_[0,0]+pos_[0,2])/2,
                                  (pos_[0,1]+pos_[0,3])/2])
            cells_ = np.vstack([cells_,pos_])
            infoDic[repr(pos_)] = ruPareto(mid_point,par)
            if ruPareto(mid_point,par):
                Jparet_ = np.vstack([Jparet_,pos_])
            else:
                not_Jparet_ = np.vstack([not_Jparet_,pos_])


    return cells_,Jparet_,not_Jparet_,infoDic


def ruPareto(x,par):
    for val in par:
        if(x[0]>val[0] and x[1]>val[1]):
            return False
    return True

def dvt_mu(xs,Data,Kernels,yReal,Ker_,Kinv):
    #len_matrix = -np.linalg.inv(np.identity(INPUT_DIM)*LEN_SCALE**2)
    len_matrix = -(np.identity(INPUT_DIM)*(1/(LEN_SCALE**2)))
    XsT = (Data - xs).T
    tmpI = np.dot(len_matrix,XsT)
    KxsX = Kernels[Ker_].K(xs,Data).T
    #tmpII = np.dot(np.linalg.pinv(Kernels[Ker_].K(Data,Data)),np.matrix(yReal).T)
    tmpII = np.dot(Kinv,np.matrix(yReal).T)
    tmpIII = np.multiply(KxsX,tmpII)
    res_ = np.dot(tmpI,tmpIII)
    return res_

def dvt_var(xs,Data,Kernels,Ker_,Kinv):
    sizeD = Data.shape[0]
    len_matrix = (np.identity(INPUT_DIM)*(1/(LEN_SCALE**2)))
    KXX_m1 = Kinv#np.linalg.pinv(Kernels[Ker_].K(Data,Data))
    KxsX = Kernels[Ker_].K(xs,Data)
    X_xs = (Data-xs).T

    Alis_ = 0
    for i in range(0,sizeD):
        for j in range(0,sizeD):
            tmpI = np.dot(np.dot(np.matrix(X_xs[:,i]).T,np.matrix(X_xs[:,j])),len_matrix**2)
            tmp1 = KXX_m1[i,j]*KxsX[0,i]*KxsX[0,j]
            Alis_ += tmp1*tmpI

    return (len_matrix + Alis_)


def improved_dvt_var(xs,Data,Kernels,Ker_,Kinv):
    len_matrix = (np.identity(INPUT_DIM)*(1/(LEN_SCALE**2)))
    first_ = len_matrix
    second_ = np.dot((Kernels[Ker_].K(xs,Data)*(Data-xs).T).T,len_matrix)
    third_ = Kinv
    
    result = first_ + np.dot(np.dot(second_.T,third_),second_)
    return result
    
        

def Helpme_(points):
    grid_,J,NJ,dic = samplePareto(points)
    for val in grid_:
        if np.invert(dic[repr(val)]):
            plt.plot(val[0,2],val[0,3],'*r',markersize=15)
    plt.plot(anotherY[:,0],anotherY[:,1],'*b',markersize=10)
    plt.show()

def cell_point_dom(points,cell):
    res_ = []
    for i in range(0,len(points)):
        if ((cell[0,0] >= points[i,0]) and (cell[0,1] >= points[i,1])):
            res_.append(i)
    return res_

def Expected_HVI(points,weights,Not_Jgrid_):
    if (len(points) != len(weights)):
        sys.exit("WARNING WEIGHTs AND POINTS?")
    #grid_,Jgrid_,Not_Jgrid_,pMap_ = samplePareto(points)
    #print("================")
    #print(Not_Jgrid_)
    #print(Not_Jgrid__)
    #print("================")
    #time.sleep(2)
    
    w_area = 0
    j_area = 0
    wPoints = {}
    Weightsdic = {}
    for val in Not_Jgrid_:
        wPoints[repr(val)] = cell_point_dom(points,val)
        if (len(wPoints[repr(val)])):
            exWeights = 1 - np.prod(np.take(1-weights,wPoints[repr(val)]))
        else:
            exWeights = 1
        area_ = (val[0,3] - val[0,1])*(val[0,2] - val[0,0])
        w_area += exWeights*area_
        j_area += area_
        Weightsdic[repr(val)] = exWeights
    return w_area

def createPoints(deepness,breadth):
    points_ = {}
    for i in range(1,deepness+1):
        #temp_x = np.arange(0,i/3,i/(breadth*3))
        #temp_x = np.arange(0,i,i/(breadth))
        temp_x = np.linspace(0, i/2, num=breadth)
        temp_y = np.zeros(temp_x.shape)+i
        temp_merge = np.vstack((temp_x,temp_y))
        points_[i] = temp_merge

    for key in points_:
        if key!=1:
            points_[1] = np.hstack((points_[1],points_[key]))
    return points_[1].T

def findWeight(Mu,Sigma,deepness,breadth,points_):
    mvnMu = np.array([Mu[0][0,0], Mu[1][0,0]])
    mvnSigma = np.array([[Sigma[0][0,0],0],[0,Sigma[1][0,0]]])
    Rotationdict = createRot()
    sumProb = 0
    log = []
    #points_ = createPoints(deepness,breadth)
    #spoints_ = points_.shape[1]
    for key in Rotationdict:
        mvnMuRnine = np.dot(Rotationdict[key],mvnMu)
        mvnSigmaRnine = np.dot(Rotationdict[key].T,np.dot(mvnSigma,Rotationdict[key]))
        sumProb += np.sum(multivariate_normal.pdf(points_,mvnMuRnine,mvnSigmaRnine))
        log.append(sumProb)
    return sumProb/(deepness*breadth)


def createRot():
    Rotationdict = {}
    Rotationdict[0] = np.array([[0,-1],[1,0]])
    Rotationdict[1] = np.array([[0,1],[-1,0]])
    return Rotationdict

def findIntegral(Mu,Sigma):
    mvnMu = np.array([Mu[0][0,0], Mu[1][0,0]])
    mvnSigma = np.array([[Sigma[0][0,0],0],[0,Sigma[1][0,0]]])
    Rotationdict = createRot()
    sumProb = 0
    print(mvnMu)
    print(mvnSigma)
    for key in Rotationdict:
        print(key)
        mvnMuRnine = np.dot(Rotationdict[key],mvnMu)
        mvnSigmaRnine = np.dot(Rotationdict[key].T,np.dot(mvnSigma,Rotationdict[key]))
        print(mvnMuRnine)
        print(mvnSigmaRnine)
        low = np.array([0, 0])
        upp = np.array([math.inf, math.inf])
        p1,i = mvn.mvnun(low,upp,mvnMuRnine,mvnSigmaRnine)
        print(p1)
        sumProb += p1
    return sumProb

def plot_me(yPareto,Probs):
    grid_,J,NJ,dic = samplePareto(yPareto)
    for val in grid_:
        '''
        if dic[repr(val)]:
            plt.plot(val[0,2],val[0,3],'or',marker='.',markersize=14)
        else:
            plt.plot(val[0,2],val[0,3],'og',marker='.',markersize=14)
        '''
    for i in range((yPareto.shape[0])):
        plt.plot(yPareto[i,0],yPareto[i,1],'ob',marker='*',markersize=12)
        plt.text(yPareto[i,0]-0.5,yPareto[i,1]-0.5,np.round(Probs[i],5),size=14)
    plt.show()

def Derivatives(x,xPareto,dataset,Kernels,Kinv_0,Kinv_1):
    Mu = {}
    Sigma = {}
    Mu[0] = (dvt_mu(x,dataset.data,Kernels,dataset.outputs[:,0],'ker0',Kinv_0))
    Mu[1] = (dvt_mu(x,dataset.data,Kernels,dataset.outputs[:,1],'ker1',Kinv_1))
    Sigma[0] = (improved_dvt_var(x,dataset.data,Kernels,'ker0',Kinv_0))
    Sigma[1] = (improved_dvt_var(x,dataset.data,Kernels,'ker1',Kinv_1))
    return Mu,Sigma

def sampleXs(X,Y,bound):
    ind0_0 = np.where(Y[0,:] >= bound[0])[0]
    ind1_0 = np.where(Y[1,:] >= bound[1])[0]
    fIndx_0 = list(set(ind0_0) & set(ind1_0))

    ind0_1 = np.where(Y[0,:] <= bound[2])[0]
    ind1_1 = np.where(Y[1,:] <= bound[3])[0]
    fIndx_1 = list(set(ind0_1) & set(ind1_1))

    finalInd = list(set(fIndx_0) & set(fIndx_1))
    return X.T[finalInd],Y.T[finalInd]

def WeightPoints(xPareto,dataset,Kernels,Kinv_0,Kinv_1,points_):
    Wlog = []
    RPareto = []
    for val in xPareto:
        x = np.array([val])
        Mu,Sigma = Derivatives(x,xPareto,dataset,Kernels,Kinv_0,Kinv_1)
        tmp_ = findWeight(Mu,Sigma,DEEP,BREAD,points_)
        Wlog.append(tmp_)
        RPareto.append([function(x)[0,0],function(x)[0,1]])
    Weights = np.array(Wlog)
    RPareto = np.array(RPareto)
    return Weights,RPareto

def Generate_bounded(x1,x2,y1,y2,num):
    first_ = np.random.uniform(low = x1, high = x2, size=(num,))
    second_ = np.random.uniform(low = y1, high = y2, size=(num,))
    return np.vstack((first_,second_)).T

def AQFunc(bounds,Weights,RPareto):
    print("(***)")


if __name__ == '__main__':

    ################# PARAMETERS
    INITIAL = 16
    COUNTER = 3
    DEEP = 100
    BREAD = 100
    COUNTER = 40
    ctn = 0
    MAXSAMPLE = 10**6
    bounds = dict()
    bounds  = {'min': [-5],'max':[5]}

    ################# GENERATE DATASET
    start_time = time.time()
    xData = np.loadtxt('xData.txt', dtype=float)
    xData = xData.reshape(len(xData),1)
    yData = np.loadtxt('yData.txt', dtype=float)
    #xData,yData = initvals_(bounds,INITIAL)
    dataset = DataComplex(xData,yData)
    yPareto = mPareto(dataset.outputs)
    xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
    initial_pareto = yPareto
    points_ = createPoints(DEEP,BREAD)
    print("_____________ INFO ________________")
    info(dataset.data,dataset.outputs,yPareto,xPareto)
    X,Y = initvals_(bounds,MAXSAMPLE)
    X,Y = X.T,Y.T
    Kernels = ctKernel(OUTPUT_DIM,INPUT_DIM,LEN_SCALE,LEN_SCALE)
    print("_____________________________")
    print("Data Initialized in %s seconds; OK!\n" % round(time.time() - start_time,5))


    while ctn < COUNTER:

        ################# TRAIN THE MODEL
        start_time = time.time()
        mod1,Kernels['ker0'] = trainModel(dataset.data,np.matrix(dataset.outputs[:,0]).T,'ker0',40)
        mod2,Kernels['ker1'] = trainModel(dataset.data,np.matrix(dataset.outputs[:,1]).T,'ker1',40)
        Kinv_0 = np.linalg.pinv(Kernels['ker0'].K(dataset.data,dataset.data))
        Kinv_1 = np.linalg.pinv(Kernels['ker1'].K(dataset.data,dataset.data))
        print("_____________________________")
        print("GP trained in %s seconds; OK!\n" % round(time.time() - start_time,5))

        #################  FIND THE PARETO
        start_time = time.time()
        yPareto = mPareto(dataset.outputs)
        xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
        if (len(xPareto)!=len(yPareto)):
            sys.exit("Size of X pareto is not same as Y pareto!")
        print("_____________________________")
        print("Found the Pareto in %s seconds; OK!\n" % round(time.time() - start_time,5))

        #################  READY TO LUNCH THE LOOP
        start_time = time.time()
        copyPareto = deepcopy(yPareto)
        copyxPareto = deepcopy(xPareto)
        SlidingY = np.empty(shape = (0,OUTPUT_DIM))
        grid_,Jgrid_,Ngridholder_,pMap_ = samplePareto(yPareto)
        for valg in Jgrid_:
            if (valg[0,0]>valg[0,1]) and (valg[0,2]>valg[0,3]):
                yBatch = Generate_bounded(valg[0,0],valg[0,2],valg[0,1],valg[0,3],2)
                SlidingY = np.vstack((SlidingY,yBatch))

        if (SlidingY.shape[0] > 64):
            indices_ = [random.randint(0, SlidingY.shape[0]-1) for p in range(0, 64)]
            SlidingY = SlidingY[indices_]
            #plt.plot(SlidingY[:,0],SlidingY[:,1],"*g")
            #plt.show()
            
        Faster = {}
        for k in (range(SlidingY.shape[0])):
            AddY = np.vstack((yPareto,np.array([SlidingY[k,0],SlidingY[k,1]])))
            ParAddY = mPareto(AddY)
            Fast_1,Fast_2,Fast_3,Fast_4 = samplePareto(ParAddY)
            Faster[k] = Fast_3    
        print("Number of grids to handle: %s" % SlidingY.shape[0])
        x_log = []
        imp_log = []
        indices_ = [random.randint(0, X.shape[1]-1) for p in range(0, 100)]
        optimizerX = X.T[indices_]
        print("_____________________________")
        print("Data Prep for main loop launch in %s seconds; OK!\n" % round(time.time() - start_time,5))
        
        for i in tqdm(range(len(optimizerX))):
            x_log.append([optimizerX[i,:]])
            Total_HVI_diff = 0
            y1,Sigy1 = testModel(mod1,np.array([optimizerX[i,:]]))
            y2,Sigy2 = testModel(mod2,np.array([optimizerX[i,:]]))
            temp_x_pareto = np.vstack((copyxPareto,np.array([optimizerX[i,:]])))
            New_Weights_,New_Paretos_ = WeightPoints(temp_x_pareto,dataset,Kernels,Kinv_0,Kinv_1,points_)
            slide_size = len(SlidingY)
            for j in (range(slide_size)):
                temp_pareto = np.vstack((New_Paretos_[:-1],np.array([SlidingY[j,0],SlidingY[j,1]])))
                found_temp_pareto = mPareto(temp_pareto)
                usef_weights = New_Weights_[parY_X(temp_pareto,found_temp_pareto)]

                Probs_dim1 = norm(y1, Sigy1).pdf(SlidingY[j,0])
                Probs_dim2 = norm(y2, Sigy2).pdf(SlidingY[j,1])
                EHVI_New = Expected_HVI(found_temp_pareto,usef_weights,Faster[j])*Probs_dim1*Probs_dim2
                Total_HVI_diff += EHVI_New
            imp_log.append(Total_HVI_diff)
        indx = imp_log.index(max(imp_log))
        Best_x = x_log[indx][0]
        Best_y = function(np.array([Best_x]))[0,0],function(np.array([Best_x]))[0,1]
        if Best_x not in dataset.data:
            dataset.newData(Best_x)
            dataset.newOut(Best_y)
        else:
            sys.exit("Couldn't go on, almost the same point! Why?!")
        ctn += 1

        plt.plot(yPareto[:,0],yPareto[:,1],"or",markersize = 8,label='solutions we found')
        plt.plot(initial_pareto[:,0],initial_pareto[:,1],"*b",markersize = 15,label='initial observation')
        plt.legend('Model length', 'Data length',loc='upper center', shadow=True)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.legend()
        plt.title('r2>(2)*r1')
        plt.savefig('2_32_lONG_times.pdf', format='pdf', dpi=1000)
        plt.show()











    '''
        start_time = time.time()
        yPareto = mPareto(dataset.outputs)
        xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
        print("==============PARETO SHAPE================")
        print(yPareto.shape)
        print("==========================================")
        Weights_,Paretos_ = WeightPoints(xPareto,dataset,Kernels,Kinv_0,Kinv_1)
        EHVI = Expected_HVI(Paretos_,Weights_)
        print("_____________________________ %s " % EHVI)
        print("Hypervolume found in %s seconds; OK!\n" % round(time.time() - start_time,3))
        '''

    '''
    
    
            print("==============================")
            print("size of X is: ",dataset.data.shape)
            print("size of Y is: ",dataset.outputs.shape)
            print("==============================")
    if np.array_equal(np.round(mPareto(temp_pareto),5),np.round(yPareto,5)):
        pass
    else:
    plot_me(Paretos_,Weights_)
    ############################################################# CALLING DERIVATIVES
    Wlog = []
    RPareto = []

    ############################################################# ROTATE THEM UP FIND INT
    for val in xPareto:
        x = np.array([val])
        Mu,Sigma = Derivatives(x,xPareto,dataset,Kernels)
        tmp_ = findWeight(Mu,Sigma,DEEP,BREAD)
        Wlog.append(tmp_)
        RPareto.append([function(x)[0,0],function(x)[0,1]])
        plt.plot(function(x)[0,0],function(x)[0,1],'ob',marker='*',markersize=12)
        plt.text(function(x)[0,0]-0.5,function(x)[0,1]-0.5,str(np.round(tmp_,5)),size=14)

    print('\n______________Adding up the weights_______________')
    Weights = np.array(Wlog)
    RPareto = np.array(RPareto)


    #AQFunc(Weights,RPareto)
    EWHI = Expected_HVI(RPareto,Weights)
    print(EWHI)
    '''






    '''
    AQFunc(mod1,mod2,Kernels,xPareto,yPareto,dataset)
    ############################################################# FIND FOR EVERYPOINT
    print('\n______________Adding up the weights_______________')
    Weights = np.array(Wlog)
    res1,res2,res3 = Expected_HVI(yPareto,Weights)
    plot_me(yPareto,Wlog)
    plot_me(RPareto,Wlog)

    dataset.newData(np.array([1]))
    dataset.newOut(np.array([9.0,9.0]))
    print(dataset.outputs)


    def sampleGenerator(X,mod1,mod2):
        Y = np.empty((OUTPUT_DIM,X.shape[1]))
        for i in range(X.shape[1]):
            mu_1,sig_1 = testModel(mod1,np.array([[X[0,i]]]))
            mu_2,sig_2 = testModel(mod2,np.array([[X[0,i]]]))
            Y[0,i] = mu_1
            Y[1,i] = mu_2
        print(np.min(Y[0,:]))
        print(np.min(Y[1,:]))
            #print(Y.shape)

sampleGenerator(X,mod1,mod2)
    '''
            #print("found_temp_pareto ",found_temp_pareto.shape)
            #temp_x_pareto = list(map(temp_x_pareto.__getitem__, parY_X(temp_pareto,found_temp_pareto)))
    '''
            plt.plot(copyPareto[:,0],copyPareto[:,1],'ob',markersize=20)
            print(valy[0],valy[1])
            plt.plot(valy[0],valy[1],'*r',markersize=20)
            plt.plot(found_temp_pareto[:,0],found_temp_pareto[:,1],'og')

            plt.axis((-1,10,-1,10))
            plt.show()newOut

            #print("Not the same, initiate the process!~!")

            print("-------------")
            #print(New_Weights_)
    '''
