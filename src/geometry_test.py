import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import frame3dd
import subprocess
import pfea
import cProfile
from math import *

#geometry files

import cuboct
import cuboct_buckle
import kelvin
import pschwarz
import octet

######  #######  #####   #####  ######  ### ######  ####### ### ####### #     #
#     # #       #     # #     # #     #  #  #     #    #     #  #     # ##    #
#     # #       #       #       #     #  #  #     #    #     #  #     # # #   #
#     # #####    #####  #       ######   #  ######     #     #  #     # #  #  #
#     # #             # #       #   #    #  #          #     #  #     # #   # #
#     # #       #     # #     # #    #   #  #          #     #  #     # #    ##
######  #######  #####   #####  #     # ### #          #    ### ####### #     #

# Use this file to test geometries, to make sure everything lines up
# This file should just generate the node/frame positions for a given lattice, and plot them
# good way to test functionality


####### ######     #    #     # #######
#       #     #   # #   ##   ## #
#       #     #  #   #  # # # # #
#####   ######  #     # #  #  # #####
#       #   #   ####### #     # #
#       #    #  #     # #     # #
#       #     # #     # #     # #######

######  ####### ######  #     # #          #    ####### ### ####### #     #
#     # #     # #     # #     # #         # #      #     #  #     # ##    #
#     # #     # #     # #     # #        #   #     #     #  #     # # #   #
######  #     # ######  #     # #       #     #    #     #  #     # #  #  #
#       #     # #       #     # #       #######    #     #  #     # #   # #
#       #     # #       #     # #       #     #    #     #  #     # #    ##
#       ####### #        #####  ####### #     #    #    ### ####### #     #


#Temporary Material Matrix - NxNxN cubic grid (corresponding to cubic-octahedra)
# at the moment:
# 1's correspond to material being there
# 0's correspond to no material


mat_matrix = [[[0,0,0],
			   [0,0,0],
			   [0,0,0]],
			  [[0,0,0],
			   [0,1,0],
			   [0,0,0]],
			  [[0,0,0],
			   [0,0,0],
			   [0,0,0]]]

subdiv = 2
zheight = subdiv
subdiv_beam = False
#Subdivide the material matrix
if subdiv > 1:
	dims = np.shape(mat_matrix)
	new_mat_matrix = np.zeros(((dims[0]-2)*subdiv+2,(dims[1]-2)*subdiv+2,(dims[2]-2)*zheight+2))

	for i in range(1,dims[0]-1):
		for j in range(1,dims[1]-1):
			for k in range(1,dims[2]-1):
				if mat_matrix[i][j][k] == 1:
					dex = ((i-1)*subdiv+1,(j-1)*subdiv+1,(k-1)*zheight+1)
					for l in range(0,subdiv):
						for m in range(0,subdiv):
							for n in range(0,zheight):
								new_mat_matrix[l+dex[0]][m+dex[1]][n+dex[2]] = 1


	mat_matrix = new_mat_matrix

# Material Properties

#Physical Voxel Properties
vox_pitch = 0.01/subdiv #m

#Node Map Population
#Referencing the geometry-specific file.

#nodes,frames,node_frame_map,dims = kelvin.from_material(mat_matrix,vox_pitch)
#nodes,frames,node_frame_map,dims = cuboct_buckle.from_material(mat_matrix,vox_pitch)
#nodes,frames,node_frame_map,dims = pschwarz.from_material(mat_matrix,vox_pitch)
nodes,frames,node_frame_map,dims = octet.from_material(mat_matrix,vox_pitch)
#print(node_frame_map)
nodes = np.array(nodes)
print(len(nodes))
print(len(frames))
if subdiv_beam:
	num_bsub = 2
	#frame_props["Le"] = frame_props["Le"]*1.0/(num_bsub+1)
	newframes = list(frames)
	tdex = len(nodes)
	for i,frame in enumerate(frames):
		st_nid = frame[0]
		en_nid = frame[1]
		stnode = nodes[st_nid]
		nodevec = nodes[en_nid]-nodes[st_nid]
		print(nodevec)
		#add nodes in the middle
		idvals = [frame[0]]
		for sb in range(num_bsub):
			nodes =  np.vstack((nodes,[nodes[st_nid]+1.0*(sb+1.0)/(num_bsub+1.0)*nodevec]))
			idvals.append(tdex)
			tdex = tdex+1
		idvals.append(frame[1])
		newframes[i][:] = [idvals[0],idvals[1]]
		for bs in range(1,num_bsub+1):
			newframes.append([idvals[bs],idvals[bs+1]])

	frames = newframes
#print(frames,np.shape(nodes))
#Set up figure plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#nodes = np.array(nodes)
xs = np.array(nodes.T[0])
ys = np.array(nodes.T[1])
zs = np.array(nodes.T[2])

#This maintains proper aspect ratio for the 3d plot
max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xs.max()+xs.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ys.max()+ys.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zs.max()+zs.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

#plot all of the frames
for i,frame in enumerate(frames):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)

#plot the nodes
ax.scatter(xs,ys,zs)

#show it
plt.show()
