#!/usr/bin/env python
from __future__ import division
from numpy import *
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import networkx as nx
import itertools
import cvxopt as co
import csv

'''
def V3(x,y,z):
    return asarray([x,y,z])
def dots(A,V):
    return inner(A,V).T
def magnitudes(v):
	return sqrt(sum(v**2,axis=-1))
def close(a,b):
  return absolute(a-b)<1e-5
def mid(a):
    return .5*(amax(a)+amin(a))
def avg(a,axis=0):
    return sum(a,axis=axis)/shape(a)[axis]
def extremes(a):
    return array([amin(a),amax(a)])
def span(a):
    return amax(a)-amin(a)
def sqr_magnitude(x):
    return sum(x*x,axis=-1)
def combine_topologies(node_sets,seg_sets):
    assert(len(node_sets)==len(seg_sets))
    all_nodes = vstack(tuple(node_sets))
    offsets = cumsum( [0] + map(lambda x: shape(x)[0], node_sets)[:-1] )
    seg_lists = tuple([os+bs for bs,os in zip(seg_sets,offsets)])
    return all_nodes,seg_lists
def subdivide_topology(nodes,segs):
    n_nodes = shape(nodes)[0]; n_seg = shape(segs)[0]
    nodes = vstack((nodes,.5*sum(nodes[segs],axis=1)))
    segs_1 = hstack((segs[:,0,None],arange(n_seg)[...,None]+n_nodes))
    segs_2 = hstack((arange(n_seg)[...,None]+n_nodes,segs[:,1,None]))
    return nodes,vstack((segs_1,segs_2))

def unique_points(a,tol=1e-5,leafsize=10):
    #Use KDTree to do uniqueness check within tolerance.
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

def line_plane_intersection(P0,N,l,l0=array([0,0,0])):
    #plane through p0 normal to N, line is l0 + t*l
    #return distance from l0
    return dot(P0-l0,N)/dot(l,N)

'''

def coord_trans(x_n1,x_n2,L,p):
    # Find the coordinate transform from local beam coords
    # to the global coordinate frame
    # x_n1  : x,y,z position of the first node n1
    # x_n2  : x,y,z position of the second node n2
    # L     : length of beam (without accounting for node radius)
    # p     : the roll angle in radians
    L = np.linalg.norm(x_n1-x_n2)
    Cx = (x_n2[0]-x_n1[0])/L
    Cy = (x_n2[1]-x_n1[1])/L
    Cz = (x_n2[2]-x_n1[2])/L

    t = np.zeros(9)

    Cp = cos(p)
    Sp = sin(p)

    # We assume here that the GLOBAL Z AXIS IS VERTICAL.

    if ( fabs(Cz) == 1.0 ):
        t[2] =  Cz;
        t[3] = -Cz*Sp;
        t[4] =  Cp;
        t[6] = -Cz*Cp;
        t[7] = -Sp;
    else:
        den = sqrt ( 1.0 - Cz*Cz );

        t[0] = Cx;
        t[1] = Cy;
        t[2] = Cz;

        t[3] = 1.0*(-Cx*Cz*Sp - Cy*Cp)/den;    
        t[4] = 1.0*(-Cy*Cz*Sp + Cx*Cp)/den;
        t[5] = Sp*den;

        t[6] = 1.0*(-Cx*Cz*Cp + Cy*Sp)/den;
        t[7] = 1.0*(-Cy*Cz*Cp - Cx*Sp)/den;
        t[8] = Cp*den;

    return t

def atma(t,m):
    a = co.matrix(0.0,(12,12))

    #More efficient assignment possible
    for i in range(0,4):
        a[3*i,3*i] = t[0];
        a[3*i,3*i+1] = t[1];
        a[3*i,3*i+2] = t[2];
        a[3*i+1,3*i] = t[3];
        a[3*i+1,3*i+1] = t[4];
        a[3*i+1,3*i+2] = t[5];
        a[3*i+2,3*i] = t[6];
        a[3*i+2,3*i+1] = t[7];
        a[3*i+2,3*i+2] = t[8];
    
    m = co.matrix(np.dot(np.dot(a.T,m),a))

    return m

def swap_Matrix_Rows(M,r1,r2):
    r1 = int(r1)
    r2 = int(r2)
    M[[r1,r2],:] = M[[r2,r1],:]
    
def swap_Matrix_Cols(M,c1,c2):
    c1 = int(c1)
    c2 = int(c2)
    M[:,[c1,c2]] = M[:,[c2,c1]]
    
def swap_Vector_Vals(V,i1,i2):
    V[[i1,i2]] = V[[i2,i1]]

def gen_Node_map(nodes,constraints):
    # we want to generate the map between the input K 
    # and the easily-solved K
    ndof = len(nodes)*6
    index = ndof-len(constraints)
    
    indptr = np.array(range(ndof))
    data = np.array([1.0]*ndof)
    row = np.array(range(ndof))
    col = np.array(range(ndof))
    
    cdof_list = []
    
    for constraint in constraints:
        cdof_list.append(constraint["node"]*6+constraint["DOF"])
        
    for c_id in cdof_list:
        if c_id < ndof-len(constraints):
            not_found = True
            while not_found:
                if index in cdof_list:
                    index = index+1
                else:
                    col[c_id] = index
                    col[index] = c_id
                    not_found = False
                    index=index+1


    return co.spmatrix(data,row,col)

def plotLattice(nodes,frames,res_displace,scale):
    # Function to plot the intial lattice configuration
    # and the final version of the lattice configuration
    #
    # Input:    nodes - Initial node location
    #           frames - node frames
    #           res_displace - displacement of nodes
    #           scale - scaling parameter
    
    #intialize arrays
    xs = []
    ys = []
    zs = []
    
    rxs = []
    rys = []
    rzs = []
    
    #create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    frame_coords = []

    #poplate x, y, and z start and displacement arrays
    for i,node in enumerate(nodes):
	xs.append(node[0])
	ys.append(node[1])
	zs.append(node[2])
	rxs.append(node[0]+res_displace[i][0]*scale)
	rys.append(node[1]+res_displace[i][1]*scale)
	rzs.append(node[2]+res_displace[i][2]*scale)

    # Add frame
    for i,frame in enumerate(frames):
	nid1 = int(frame[0])
	nid2 = int(frame[1])
	start = [xs[nid1],ys[nid1],zs[nid1]]
	end   = [xs[nid2],ys[nid2],zs[nid2]]
	rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
	rend   = [rxs[nid2],rys[nid2],rzs[nid2]]
	ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
	ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=0.3)

    #plot
    ax.scatter(xs,ys,zs, color='r',alpha=0.1)
    ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
    plt.show()


def writeCSV(nodes,res_displace,filename):
    #Might need to eventually add rotational displacement
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['X Start,Y Start,Z Start,X Rot,Y Rot,Z Rot,X Disp,Y Disp,Z Disp'])
        for i,node in enumerate(nodes):
            # Note that for now I have it so the rotation is always zero to intialize. I think that the current iteration of the solver has this requirement as well
            # it might be useful to adapt this later
            spamwriter.writerow([node[0]]+[node[1]]+[node[2]]+[0.0]+[0.0]+[0.0]+[res_displace[i][0]]+[res_displace[i][1]]+[res_displace[i][2]])

