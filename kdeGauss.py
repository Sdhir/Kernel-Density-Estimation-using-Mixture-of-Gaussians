# -*- coding: utf-8 -*-
"""
Enlitic - Work sample on 

 Kernel density estimation with Mixture of Gaussians
 
 Coded by Sudhir Sornapudi
 Email: ssbw5@mst.edu
"""

"""
 KDE Model with Gaussian kernel
"""

import numpy as np
from numpy.matlib import repmat
from math import pi
from decimal import *
getcontext().prec = 7 # Precision for decimal

"""
Model defined according to the problem statement
parameters: Sigma is constant and Mu is equivalent to X_A, i.e., Train data
"""
def model(X_A, X_B, sigma):
    len_t = len(X_A)
    len_v = len(X_B)
    d = len(X_A[0])
    mu = X_A.astype(np.float64)
    p_x = 0
    t2 = -Decimal(0.5*d)*(Decimal(2*pi*(sigma**2)).ln())
    ln_k = (Decimal(len_t)).ln()
    for i in range(len_t):
        x = repmat(X_B[i], len_t, 1)
        x = x.astype(np.float64)
        t1 = np.sum((-(x-mu)**2)/(2*(sigma**2)),axis=1)
        sum_ele = 0
        for t in range(len_v):
            temp = (Decimal(t1[t]).exp())
            sum_ele += temp
        log_p = t2 - ln_k + sum_ele.ln()
        p_x += log_p
    L = p_x/len_v
    return L
    