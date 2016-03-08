import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cuboct
import pfea
import frame3dd
import subprocess
import json
from math import *
import DM23dd 
import pickle

mat_matrix = DM23dd.mat_matrix_from_json('data/hut.json')

subdiv = 1
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

#print(new_mat_matrix)

mat_matrix = new_mat_matrix
# Material Properties

#Physical Voxel Properties
vox_pitch = 0.2828/subdiv #m
node_radius = 0 

#STRUT PROPERTIES
#Physical Properties
#A pultruded carbon fiber 5mm OD 3mm ID tube  
#Using Newtons, meters, and kilograms as the units
frame_props = {"nu"  : 0.3, #poisson's ratio
			   "d1"	 : 0.005/subdiv, #m
			   "th"	 : 0.001/subdiv, #m
			   "E"   :  135000000000, #N/m^2
			   "rho" :  1600, #kg/m^3
			   "beam_divisions" : 0,
			   "cross_section"  : 'circular',
			   "roll": 0,
			   "Le":vox_pitch/sqrt(2.0),
			   "node_mass":0.01615} 

#Node Map Population
#Referencing the geometry-specific cuboct.py file. 

#node_frame_map = np.zeros((len(mat_matrix),len(mat_matrix[0]),len(mat_matrix[0][0]),6))
nodes,frames,node_frame_map = cuboct.from_material(mat_matrix,vox_pitch)

#rotation of the structure so it's properly oriented
#Using Rodrigues' rotation formula
k = [-1./sqrt(2.),1./sqrt(2.),0]
K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
angle = -125.264/180.0*np.pi
R = np.identity(3)+np.sin(angle)*K+(1-np.cos(angle))*np.dot(K,K) 

newnodes = np.zeros((np.shape(nodes)[0],3))
for i,node in enumerate(nodes):
	#print(node,np.dot(R,node))
	newnodes[i,:] = np.dot(R,node)

#Rezero the z-location of the structure
minval = np.min(newnodes.T[2])
newnodes.T[2] = newnodes.T[2]-minval
maxval = np.max(newnodes.T[2])

nodes = newnodes
#Constraint and load population
constraints = []
loads = []
benjamin = 0
benjamin = 686.0/(6.0*subdiv)

meanloc = np.array([np.mean(nodes.T[0]),np.mean(nodes.T[1])])
#Constraints are added based on simple requirements right now
for i,node in enumerate(newnodes):
	if node[2] < 0.05:
		#if np.linalg.norm(meanloc-node[0:2])-1.27 < 0.01:
			#vec = -0.001*(meanloc-node[0:2])/np.linalg.norm(meanloc-node[0:2])
		constraints.append({'node':i,'DOF':2, 'value':0})
		constraints.append({'node':i,'DOF':0, 'value':0})
		constraints.append({'node':i,'DOF':1, 'value':0})
		#print(np.linalg.norm(meanloc-node[0:2]))
		#constraints.append({'node':i,'DOF':3, 'value':0})
		#constraints.append({'node':i,'DOF':4, 'value':0})
		#constraints.append({'node':i,'DOF':5, 'value':0})
	#if np.abs(node[2]-maxval) < 0.05:
	#	loads.append({'node':i,'DOF':2, 'value':-0.1})

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
								 'th'  : frame_props["th"],
								 'beam_divisions' : frame_props["beam_divisions"],
								 'cross_section'  : frame_props["cross_section"],
								 'roll': frame_props["roll"],
								 'loads':{'element':0},
								 'prestresses':{'element':0},
								 'Le': frame_props["Le"],
								 'node_mass':frame_props["node_mass"]})]

#Format node positions
out_nodes = np.array(newnodes)

#Global Arguments 
global_args = {'frame3dd_filename': "hut1_s{0}_mt".format(subdiv), 
			   'length_scaling':1,
			   'using_Frame3dd':False,
			   'debug_plot':True,
			   'gravity':(0,0,-9.81)}

if global_args["using_Frame3dd"]:
	frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
	subprocess.call("frame3dd {0}.csv {0}.out".format(global_args["frame3dd_filename"]), shell=True)
	res_nodes, res_reactions = frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
	res_displace = frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])

else:
	res_displace,C,Q = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)
	#cProfile.run('res_displace = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)')
	#def_displace = pfea.analyze_System(out_nodes, global_args, def_frames, constraints,loads)
	with open("output/{0}_disp.pk1".format(global_args["frame3dd_filename"]), 'wb') as outfile:
		pickle.dump(res_displace,outfile)
	with open("output/{0}_C.pk1".format(global_args["frame3dd_filename"]), 'wb') as outfile:
		pickle.dump(C,outfile)
	with open("output/{0}_Q.pk1".format(global_args["frame3dd_filename"]), 'wb') as outfile:
		pickle.dump(Q,outfile)
	with open("output/{0}_nodes.pk1".format(global_args["frame3dd_filename"]), 'wb') as outfile:
		pickle.dump(nodes,outfile)
	with open("output/{0}_frames.pk1".format(global_args["frame3dd_filename"]), 'wb') as outfile:
		pickle.dump(out_frames,outfile)
	with open("output/{0}_constraints.pk1".format(global_args["frame3dd_filename"]), 'wb') as outfile:
		pickle.dump(constraints,outfile)
	with open("output/{0}_loads.pk1".format(global_args["frame3dd_filename"]), 'wb') as outfile:
		pickle.dump(loads,outfile)
	with open("output/{0}_nfm.pk1".format(global_args["frame3dd_filename"]), 'wb') as outfile:
		pickle.dump(node_frame_map,outfile)
	#np.savetxt("output/{0}_constraints.out".format(global_args["frame3dd_filename"]),constraints)
	#np.savetxt("output/{0}_loads.out".format(global_args["frame3dd_filename"]),loads)

if global_args["debug_plot"]:
	### Right now the debug plot only does x-y-z displacements, no twisting
	xs = []
	ys = []
	zs = []

	rxs = []
	rys = []
	rzs = []

	lxs = []
	lys = []
	lzs = []

	cxs = []
	cys = []
	czs = []


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax = fig.add_subplot(111)
	plt.axis('equal')
	#ax.set_aspect('equal')
	frame_coords = []
	factor = 25
	for i,node in enumerate(nodes):
		xs.append(node[0])
		ys.append(node[1])
		zs.append(node[2])
		rxs.append(node[0]+res_displace[i][0]*factor)
		rys.append(node[1]+res_displace[i][1]*factor)
		rzs.append(node[2]+res_displace[i][2]*factor)

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

	xs = np.array(xs)
	ys = np.array(ys)
	zs = np.array(zs)
	rys = np.array(rys)
	#rys = rys+(np.mean(ys)-np.mean(rys))

	max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
	Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xs.max()+xs.min())
	Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ys.max()+ys.min())
	Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zs.max()+zs.min())
	# Comment or uncomment following both lines to test the fake bounding box:
	#for xb, yb, zb in zip(Xb, Yb, Zb):
	#   ax.plot([xb], [yb], [zb], 'w')

	for i,frame in enumerate(frames):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
		rend   = [rxs[nid2],rys[nid2],rzs[nid2]]

		#ax.plot([start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
		#ax.plot([rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=0.3)
	
		ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
		ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=0.3)
	
	'''
	for dframe in dframes:
		nid1 = int(dframe[0])
		nid2 = int(dframe[1])
		dstart = [dxs[nid1],rys[nid1],rzs[nid1]]
		dend   = [dxs[nid2],rys[nid2],rzs[nid2]]
		ax.plot([dstart[0],dend[0]],[dstart[1],dend[1]],[dstart[2],dend[2]],color='b', alpha=0.1)
	'''	

	ax.scatter(xs,ys,zs, color='r',alpha=0.1)
	ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
	ax.scatter(lxs,lys,lzs,color='m',alpha=1.0,s=40)
	ax.scatter(cxs,cys,czs,color='k',alpha=1.0,s=40)
	
	#ax.scatter(ys,zs,color='r', alpha=0.1,s=10)
	#ax.scatter(rys,rzs,color='b',alpha=0.3,s=10)
	#ax.scatter(lys,lzs,color='m',alpha=1.0,s=40)
	#ax.scatter(cys,czs,color='k',alpha=1.0,s=40)
	plt.show()

	#print(frames)
