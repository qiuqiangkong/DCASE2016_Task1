# statistics of distribution
import numpy as np

def mean( p ):
    sum = 0
    for i1 in xrange( len(p) ):
        sum += i1 * p[i1]
    return sum
    
def variance( p ):
    mu = mean( p )
    sum = 0
    for i1 in xrange( len(p) ):
        sum += np.power( i1 - mu, 2 ) * p[i1]
    return sum
    
def skewness( p ):
    mu = mean( p )
    var = variance( p )
    sum = 0
    for i1 in xrange( len(p) ):
        sum += np.power( i1 - mu, 3 ) * p[i1]
    sum *= np.power( np.sqrt( var ), -3 )
    return sum

def kurtosis( p ):
    mu = mean( p )
    var = variance( p )
    sum = 0
    for i1 in xrange( len(p) ):
        sum += np.power( i1 - mu, 4 ) * p[i1]
    sum *= np.power( np.sqrt( var ), -4 )
    sum -= 3
    return sum
    
def energy( p ):
    sum = 0
    for i1 in xrange( len(p) ):
        sum += np.power( p[i1], 2 )
    return sum
    
def entropy( p ):
    sum = 0
    for i1 in xrange( len(p) ):
        sum += p[i1] * np.log( p[i1] + 1e-8 )
    sum = -sum
    return sum