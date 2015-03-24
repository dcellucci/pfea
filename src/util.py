#!/usr/bin/env python
from __future__ import division
from numpy import *
from scipy.spatial import cKDTree
import networkx as nx
import itertools

def magnitudes(v):
	return sqrt(sum(v**2,axis=-1))
def close(a,b):
  return absolute(a-b)<1e-5
def unique_points(a,tol=1e-5,leafsize=10):
    '''Use KDTree to do uniqueness check within tolerance.
    '''
    pairs = cKDTree(a,leafsize=leafsize).query_pairs(tol)  #pairs of (i,j) where i<j and d(a[i]-a[j])<tol    
    components = map( sorted, nx.connected_components(nx.Graph(data=list(pairs))) ) #sorted connected components of the proximity graph
    idx = delete( arange(shape(a)[0]),  list(itertools.chain(*[c[1:] for c in components])) ) #all indices of a, except nodes past first in each component
    inv = arange(shape(a)[0])
    for c in components: inv[c[1:]]=c[0]
    inv = searchsorted(idx,inv) 
    return idx,inv
def unique_reduce(nodes,*beamsets):
    #reduced_idx,reduced_inv = unique_rows(nodes)
    reduced_idx,reduced_inv = unique_points(nodes)
    return nodes[reduced_idx],tuple([reduced_inv[bs] for bs in beamsets])
def rotation_matrix(axis, theta):
    axis = asarray(axis)
    theta = asarray(theta)
    axis = axis/sqrt(dot(axis, axis))
    a = cos(theta/2)
    b, c, d = -axis*sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return asarray([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
RX90=rotation_matrix([1,0,0],pi/2.)
RY90=rotation_matrix([0,1,0],pi/2.)
RZ90=rotation_matrix([0,0,1],pi/2.)
def V3(x,y,z):
    return asarray([x,y,z])
def dots(A,V):
    return inner(A,V).T