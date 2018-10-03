from __future__ import division
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
import sys
from george.kernels import ExpSquaredKernel

INPUT_DIM = 1
INITIAL = 5
OUTPUT_DIM = 2
MAX_ITER = 200
EPSILON = 10**-6
REF = [10,10]
LEN_SCALE = 0.1
EST_RANGE = 256

NUM = 32

X = np.random.uniform(-3.,3.,(NUM,1))
Y = X**2# + np.random.randn(NUM,1)*0.005

kernel = GPy.kern.RBF(input_dim=1, variance=10, lengthscale=0.5)
m = GPy.models.GPRegression(X, Y,kernel)
#m.optimize_restarts(num_restarts = 10)
print(m.kern)
#m.optimize(max_f_eval = 1)

print(m.predict(np.array([[0.5]]),kern=kernel))
#plt.plot(X,Y,'.-r')
m.plot()
plt.show()
### Needs some plotting
'''
from george import kernels
import numpy as np
import matplotlib.pyplot as pl
import george

x = 10 * np.sort(np.random.rand(15))
yerr = 0.01 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.xlim(0, 10)
pl.ylim(-1.45, 1.45)
pl.xlabel("x")
pl.ylabel("y");


kernel = np.var(y) * kernels.ExpSquaredKernel(0.5)
gp = george.GP(kernel)
gp.compute(x, yerr)

x_pred = np.linspace(0, 10, 500)
pred, pred_var = gp.predict(y, x_pred, return_var=True)

pl.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                color="k", alpha=0.2)
pl.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.plot(x_pred, np.sin(x_pred), "--g")
pl.xlim(0, 10)
pl.ylim(-1.45, 1.45)
pl.xlabel("x")
pl.ylabel("y");
pl.show()
'''