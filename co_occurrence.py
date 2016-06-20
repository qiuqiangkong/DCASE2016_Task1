'''
SUMMARY:  extract co-occurrence feature and dump
AUTHOR:   Qiuqiang Kong
Created:  2016.06.06
Modified: -
--------------------------------------
'''
import config as cfg
import prepareData as ppData
import csv
import cPickle
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)


def co_occurrence_matrix( X, levels, theta ):
    M = np.zeros( (len(levels), len(levels)) )
    (a,b) = X.shape
    d = 1
    for i1 in xrange(a):
        for i2 in xrange(b):
            if theta=='0':
                p1 = X[i1,i2]
                v = i2-d
                if v>=0:
                    p2 = X[ i1, v ]
                    M[ levels[p1], levels[p2] ] += 1.
                v = i2+d
                if v<b:
                    p3 = X[ i1, v ]
                    M[ levels[p1], levels[p3] ] += 1.
            if theta=='90':
                p1 = X[i1,i2]
                v = i1-d
                if v>=0:
                    p2 = X[ v, i2 ]
                    M[ levels[p1], levels[p2] ] += 1.
                v = i1+d
                if v<a:
                    p3 = X[ v, i2 ]
                    M[ levels[p1], levels[p3] ] += 1.
    P = M / np.sum(M)
    return M, P

