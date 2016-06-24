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
import stats
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)


def co_occurrence_matrix( X, bgn, fin, interval, theta ):
    def _val_to_loct( val ):
        return int( ( val - bgn ) / interval ) 
    
    X = np.clip( X, bgn, fin )
    
    N = int( ( fin - bgn ) / interval ) + 1
    M = np.zeros( (N, N) )
    (a,b) = X.shape
    d = 1
    for i1 in xrange(a):
        for i2 in xrange(b):
            if theta=='0':
                p1 = _val_to_loct( X[i1,i2] )
                v = i2-d
                if v>=0:
                    p2 = _val_to_loct( X[ i1, v ] )
                    M[ p1, p2 ] += 1.
                v = i2+d
                if v<b:
                    p3 = _val_to_loct( X[ i1, v ] )
                    M[ p1, p3 ] += 1.
            if theta=='90':
                p1 = _val_to_loct( X[i1,i2] )
                v = i1-d
                if v>=0:
                    p2 = _val_to_loct( X[ v, i2 ] )
                    M[ p1, p2 ] += 1.
                v = i1+d
                if v<a:
                    p3 = _val_to_loct( X[ v, i2 ] )
                    M[ p1, p3 ] += 1.
    P = M / np.sum(M)
    return M, P

def energy( P ):
    N = len( P )
    sum = 0
    for i1 in xrange( N ):
        for i2 in xrange( N ):
            sum += np.power( P[i1,i2], 2 )
    return sum
    
def mean( P, type ):
    if type=='row':
        p = np.sum( P, axis=1 )
    if type=='col':
        p = np.sum( P, axis=0 )
    mu = stats.mean(p)
    return mu

def variance( P, type ):
    if type=='row':
        p = np.sum( P, axis=1 )
    if type=='col':
        p = np.sum( P, axis=0 )
    mu = stats.variance(p)
    return mu

def correlation( P ):
    mu_x = mean( P, type='row' )
    mu_y = mean( P, type='col' )
    sigma_x = np.sqrt( variance( P, type='row' ) )
    sigma_y = np.sqrt( variance( P, type='col' ) )
    N = len( P )
    sum = 0.
    for i1 in xrange( N ):
        for i2 in xrange( N ):
            sum += i1*i2*P[i1,i2] - mu_x*mu_y
    sum /= (sigma_x*sigma_y)
    return sum
    
def inertia( P ):
    N = len( P )
    sum = 0.
    for i1 in xrange( N ):
        for i2 in xrange( N ):
            sum += np.power( i1 - i2, 2 ) * P[i1,i2]
    return sum
    
def absolute_value( P ):
    N = len( P )
    sum = 0.
    for i1 in xrange( N ):
        for i2 in xrange( N ):
            sum += np.abs( i1 - i2 ) * P[i1,i2]
    return sum
    
def inverse_difference( P ):
    N = len( P )
    sum = 0.
    for i1 in xrange( N ):
        for i2 in xrange( N ):
            sum += P[i1,i2] / ( 1 + np.power(i1-i2, 2) )
    return sum
    
def entropy( P ):
    N = len( P )
    sum = 0.
    for i1 in xrange( N ):
        for i2 in xrange( N ):
            sum += P[i1,i2] * np.log( P[i1,i2] + 1e-8 )
    sum = -sum
    return sum
    

x = np.array([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]])
#x = np.array([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]]) + 0.5
bgn, fin, interval = -2, 3, 0.3
M, P = co_occurrence_matrix( x, bgn, fin, interval, theta='0' )

'''
print energy(P)
print mean( P, 'row' )
print variance( P, 'col' )
print correlation( P )
print inertia(P)
print absolute_value(P)
print inverse_difference(P)
print entropy(P)
'''