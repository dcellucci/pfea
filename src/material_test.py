import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import frame3dd
import subprocess
import pfea
import cProfile

import dschwarz
import cuboct
import cubic

from math import *


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
                                                                            

#Temporary Material Matrix - NxNxN grid 
# at the moment:
# 1's correspond to material being there
# 0's correspond to no material

'''
size = 3

size_x = size
size_y = size
size_z = size
mat_matrix = []
for i in range(0,size_x+2):
	tempcol = []
	for j in range(0,size_y+2):
		tempdep = [1]*(size_z+1)
		tempdep.append(0)
		tempdep[0] = 0
		if(i*j*(i-(size_x+1))*(j-(size_y+1)) == 0):
			tempdep = [0]*(size_z+2)
		tempcol.append(tempdep)
	mat_matrix.append(tempcol)
'''

mat_matrix = [[[0,0,0],
			   [0,0,0],
			   [0,0,0]],
			  [[0,0,0],
			   [0,1,0],
			   [0,0,0]],
			  [[0,0,0],
			   [0,0,0],
			   [0,0,0]]]
subdiv = 9

subdiv = subdiv*2

#Subdivide the material matrix
dims = np.shape(mat_matrix)
new_mat_matrix = np.zeros(((dims[0]-2)*subdiv+2,(dims[1]-2)*subdiv+2,(dims[2]-2)*subdiv+2))

for i in range(1,dims[0]-1):
	for j in range(1,dims[1]-1):
		for k in range(1,dims[2]-1):
			if mat_matrix[i][j][k] == 1:
				dex = ((i-1)*subdiv+1,(j-1)*subdiv+1,(k-1)*subdiv+1)
				for l in range(0,subdiv):
					for m in range(0,subdiv):
						for n in range(0,subdiv):
							new_mat_matrix[l+dex[0]][m+dex[1]][n+dex[2]] = 1

subdiv=subdiv/2.0
#print(new_mat_matrix)

mat_matrix = new_mat_matrix

# Material Properties

#Physical Voxel Properties
vox_pitch = 0.1016#/subdiv #m


node_radius = 0 

#STRUT PROPERTIES
#Physical Properties
#Assuming a square, Ti6Al4V strut that is 0.5 mm on a side
#Using Newtons, meters, and kilograms as the units

frame_props = {"nu"  : 0.35, #poisson's ratio
			   "d1"	 : 0.00238, #m
			   "d2"	 : 0.00238, #m
			   "th"  : 0,
			   "E"   :  3000000000, #N/m^2,
			   "rho" :  1400, #kg/m^3
			   "beam_divisions" : 0,
			   "cross_section"  : 'rectangular',
			   "roll": 0}

#Alternate frame properties to ensure that circular vs. rectangular struts
#Dont matter (much)
'''
frame_props = {"nu"  : 0.33, #poisson's ratio
			   "d1"	 : 0.000564/subdiv, #m
			   "d2"	 : 0.000564/subdiv, #m
			   "th"	 : 0.000564/2.0/subdiv, #m
			   "E"   :  116000000000, #N/m^2
			   "G"   :   42000000000,  #N/m^2
			   "rho" :  4420, #kg/m^3
			   "beam_divisions" : 0,
			   "cross_section"  : 'circular',
			   "roll": 0} #0.56*vox_pitch}#
'''

#Node Map Population
#Referencing the geometry-specific file. 

nodes,frames,node_frame_map,dims = dschwarz.alt_from_material(mat_matrix,vox_pitch)
frame_props["Le"] = dschwarz.alt_frame_length(vox_pitch)
#nodes,frames,node_frame_map,dims = cuboct.from_material(mat_matrix,vox_pitch)
#frame_props["Le"] = cuboct.frame_length(vox_pitch)
#nodes,frames,node_frame_map,dims = cubic.from_material(mat_matrix,vox_pitch)
#frame_props["Le"] = cubic.frame_length(vox_pitch)



#rotation of the structure so it's properly oriented
#Using Rodrigues' rotation formula
rotate = True

if rotate:
	k = [1,0,0]
	K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
	angle = -54.4/180.0*np.pi
	R = np.identity(3)+np.sin(angle)*K+(1-np.cos(angle))*np.dot(K,K) 

	newnodes = np.zeros(np.shape(nodes))
	for i,node in enumerate(nodes):
		#print(node,np.dot(R,node))
		#jitter = np.random.rand(3)/10000.0/subdiv
		#jitter[2] = 0 
		newnodes[i,:] = np.dot(R,node)#+jitter

	#Rezero the z-location of the structure
	meanval = [np.mean(newnodes.T[0]),np.mean(newnodes.T[1]),np.mean(newnodes.T[2])]
	
	newnodes.T[0] = newnodes.T[0]-meanval[0]
	newnodes.T[1] = newnodes.T[1]-meanval[1]
	newnodes.T[2] = newnodes.T[2]-meanval[2]
	#maxval = np.max(newnodes.T[2])

	tmp = dims[2]
	dims[2] = dims[1]
	dims[1] = tmp

	nodes = newnodes

	#cropping
	newnodes = np.zeros(np.shape(nodes))
	nodemap = np.zeros(len(nodes))
	newdex = 0
	
	for i,node in enumerate(nodes):
		if any(node >= 0.5*subdiv*vox_pitch) or any(node <= -0.5*subdiv*vox_pitch): 
			nodemap[i] = -1
		else:
			newnodes[newdex][0:3]= node
			nodemap[i] = newdex
			newdex = newdex+1
	#print(nodemap,newnodes[0:newdex])
	newframes = []
	for i, frame in enumerate(frames):
		if not(nodemap[frame[0]] == -1 or nodemap[frame[1]] == -1):
			newframes.append([nodemap[frame[0]],nodemap[frame[1]]])

	nodes = newnodes[0:newdex]
	frames = newframes

#renormalize beam cross section to keep the same relative density

tar_den = 0.01

frame_props["d1"] = np.sqrt(tar_den*(vox_pitch**3*subdiv**3)/(len(frames)*frame_props["Le"]))
frame_props["d2"] = frame_props["d1"]
rel_den = len(frames)*frame_props["d1"]*frame_props["d2"]*frame_props["Le"]/(vox_pitch**3*subdiv**3)

print(rel_den)

minval = np.min(nodes.T[2])
maxval = np.max(nodes.T[2])

strain = 0.001 #percent
strain_disp = (maxval-minval)*strain
#Constraint and load population
constraints = []
loads = []
#Constraints are added based on simple requirements right now
strains = np.zeros(6)
strains[2] = -strain_disp
minval = np.min(nodes.T[2])
maxval = np.max(nodes.T[2])
for i,node in enumerate(nodes):
	#print(node)
	if np.abs(node[2]-minval) < 0.005:
		for dof in range(6):
			constraints.append({'node':i,'DOF':dof, 'value':0})
	if np.abs(node[2]-maxval) < 0.005:
		for dof in range(6):
			constraints.append({'node':i,'DOF':dof, 'value':strains[dof]})

#print(len(constraints),len(nodes)*6)
		
'''
for x in range(1,subdiv+1):
	for y in range(1,subdiv+1):
		#The bottom-most nodes are constrained to neither translate nor
		#rotate
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':2, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':5, 'value':0})		
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':2, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':5, 'value':0})
		#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':3, 'value':0})
		#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':4, 'value':0})
		#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':5, 'value':0})
		
		#The top most nodes are assigned a z-axis load, as well as being
		#constrained to translate in only the z-direction.
		
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':2, 'value':-strain_disp})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':5, 'value':0})
		
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':2, 'value':-strain_disp})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':5, 'value':0})

		loads.append(      {'node':node_frame_map[x][y][subdiv][1],'DOF':2, 'value':-5.0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':5, 'value':0})

		loads.append(      {'node':node_frame_map[x][y][subdiv][2],'DOF':2, 'value':-5.0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':5, 'value':0})
		
'''
#frames = cuboct.remove_frame([(int(size_x/2.0)+1,int(size_y/2.0)+1,int(size_z/2.0)+1),2],node_frame_map,frames)
#dframes = cuboct.remove_frame([(int(size_x/2.0)+1,int(size_y/2.0)+1,int(size_z/2.0)+1),5],node_frame_map,dframes)



 #####  ### #     #    ####### #     # ####### ######  #     # ####### 
#     #  #  ##   ##    #     # #     #    #    #     # #     #    #    
#        #  # # # #    #     # #     #    #    #     # #     #    #    
 #####   #  #  #  #    #     # #     #    #    ######  #     #    #    
      #  #  #     #    #     # #     #    #    #       #     #    #    
#     #  #  #     #    #     # #     #    #    #       #     #    #    
 #####  ### #     #    #######  #####     #    #        #####     #    
                                                                       


#Group frames with their characteristic properties.
out_frames = [(np.array(frames),{'E'   : frame_props["E"],
								 'rho' : frame_props["rho"],
								 'nu'  : frame_props["nu"],
								 'd1'  : frame_props["d1"],
								 'd2'  : frame_props["d2"],
								 'th'  : frame_props["th"],
								 'beam_divisions' : frame_props["beam_divisions"],
								 'cross_section'  : frame_props["cross_section"],
								 'roll': frame_props["roll"],
								 'loads':{'element':0},
								 'prestresses':{'element':0},
								 'Le': frame_props["Le"]})]

#Format node positions
out_nodes = np.array(nodes)

#Global Arguments 
global_args = {'frame3dd_filename': "test", 'length_scaling':1,"using_Frame3dd":False,"debug_plot":True, "gravity" : [0,0,0]}

if global_args["using_Frame3dd"]:
	frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
	subprocess.call("frame3dd {0}.csv {0}.out".format(global_args["frame3dd_filename"]), shell=True)
	res_nodes, res_reactions = frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
	res_displace = frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])
else:
	res_displace,C,Q = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)
	#cProfile.run('res_displace,C,Q = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)')
	#def_displace = pfea.analyze_System(out_nodes, global_args, def_frames, constraints,loads)
	print("skipping")

tot_force = 0
num_forced = 0

xdim = np.max(nodes.T[0])-np.min(nodes.T[0])
ydim = np.max(nodes.T[1])-np.min(nodes.T[1])

notskipping = True
if notskipping:
	for constraint in constraints:
		if constraint["value"] != 0:
			tot_force = tot_force + C[constraint["node"]*6+constraint["DOF"]]
			num_forced+=1
	#struct_den = len(frames)*(0.5*frame_props["d1"])**2*np.pi*frame_props["Le"]/(vox_pitch**3*subdiv**3)
	print(tot_force/(xdim*ydim)/strain,0.01)

if global_args["debug_plot"]:
	### Right now the debug plot only does x-y-z displacements, no twisting
	xs = []
	ys = []
	zs = []

	rxs = []
	rys = []
	rzs = []

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_aspect('equal')
	frame_coords = []
	factor = 100

	for i,node in enumerate(nodes):
		xs.append(node[0])
		ys.append(node[1])
		zs.append(node[2])
		if notskipping:
			rxs.append(node[0]+res_displace[i][0]*factor)
			rys.append(node[1]+res_displace[i][1]*factor)
			rzs.append(node[2]+res_displace[i][2]*factor)

	frame_args = out_frames[0][1]
	if notskipping:
		st_nrg = 0.5*frame_args["Le"]/frame_args["E"]*(Q.T[0]**2/frame_args["Ax"]+Q.T[4]**2/frame_args["Iy"]+Q.T[5]**2/frame_args["Iz"])
		qmax = np.max(st_nrg)

	for i,frame in enumerate(frames):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		if notskipping:
			rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
			rend   = [rxs[nid2],rys[nid2],rzs[nid2]]
		#	ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=(1.0*st_nrg[i]/qmax)**2)

		#ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
	
	xs = np.array(xs)
	ys = np.array(ys)
	zs = np.array(zs)
	max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
	Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xs.max()+xs.min())
	Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ys.max()+ys.min())
	Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zs.max()+zs.min())
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
	   ax.plot([xb], [yb], [zb], 'w')
	
	lxs = []
	lys = []
	lzs = []
	cxs = []
	cys = []
	czs = []

	for load in loads:
		lid = int(load["node"])
		lxs.append(nodes[lid][0])
		lys.append(nodes[lid][1])
		lzs.append(nodes[lid][2])

	for constraint in constraints:
		cid = int(constraint["node"])		
		cxs.append(nodes[cid][0])
		cys.append(nodes[cid][1])
		czs.append(nodes[cid][2])
	#ax.scatter(xs,ys,zs, color='r',alpha=0.1)
	#ax.scatter(lxs,lys,lzs,color='m',alpha=1.0,s=40)
	ax.scatter(cxs,cys,czs,color='k',alpha=1.0,s=40)
	#ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
	plt.show()
	#print(frames)
