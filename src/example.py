import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import frame3dd
import subprocess
import pfea
import cProfile
import dschwarz
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
                                                                            

#Temporary Material Matrix - NxNxN cubic grid (corresponding to cubic-octahedra)
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
subdiv = 2
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
vox_pitch = 0.01/subdiv #m


node_radius = 0 

#STRUT PROPERTIES
#Physical Properties
#Assuming a square, Ti6Al4V strut that is 0.5 mm on a side
#Using Newtons, meters, and kilograms as the units

frame_props = {"nu"  : 0.33, #poisson's ratio
			   "d1"	 : 0.0005/subdiv, #m
			   "d2"	 : 0.0005/subdiv, #m
			   "th"  : 0,
			   "E"   :  116000000000, #N/m^2
			   "G"   :   42000000000,  #N/m^2
			   "rho" :  4420, #kg/m^3
			   "beam_divisions" : 0,
			   "cross_section"  : 'rectangular',
			   "roll": 0,
			   "Le":0.56*vox_pitch}#1.0*vox_pitch/sqrt(2.0)} 
'''
frame_props = {"nu"  : 0.33, #poisson's ratio
			   "d1"	 : 0.000564/subdiv, #m
			   "th"	 : 0.000564/2.0/subdiv, #m
			   "E"   :  116000000000, #N/m^2
			   "G"   :   42000000000,  #N/m^2
			   "rho" :  4420, #kg/m^3
			   "beam_divisions" : 0,
			   "cross_section"  : 'circular',
			   "roll": 0,
			   "Le":1.0*vox_pitch/sqrt(2.0)} 
'''

#Node Map Population
#Referencing the geometry-specific cuboct.py file. 
#Future versions might have different files?

node_frame_map = np.zeros((subdiv,subdiv,subdiv,6))
print(node_frame_map.shape)
nodes,frames,node_frame_map = dschwarz.from_material(mat_matrix,vox_pitch)

strain = 0.0001
strain_disp = subdiv*vox_pitch*strain
#Constraint and load population
constraints = []
loads = []
#Constraints are added based on simple requirements right now
for x in range(1,subdiv+1):
	for y in range(1,subdiv+1):
		#The bottom-most nodes are constrained to neither translate nor
		#rotate
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':2, 'value':0})		
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][3],'DOF':2, 'value':0})
		#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':3, 'value':0})
		#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':4, 'value':0})
		#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':5, 'value':0})
		
		#The top most nodes are assigned a z-axis load, as well as being
		#constrained to translate in only the z-direction.
		#loads.append(      {'node':node_frame_map[x][y][size_z][5],'DOF':2, 'value':-5.0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':2, 'value':-strain_disp})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':2, 'value':-strain_disp})
		#constraints.append({'node':node_frame_map[x][y][subdiv][5],'DOF':3, 'value':0})
		#constraints.append({'node':node_frame_map[x][y][subdiv][5],'DOF':4, 'value':0})
		#constraints.append({'node':node_frame_map[x][y][subdiv][5],'DOF':5, 'value':0})
		#constraints.append({'node':node_ids[5],'DOF':2, 'value':0})

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

tot_force = 0
for constraint in constraints:
	if constraint["value"] != 0:
		tot_force = tot_force + C[constraint["node"]*6+constraint["DOF"]]
print(tot_force/(subdiv*subdiv*vox_pitch*vox_pitch)/strain)


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
	factor = 1000

	print(matplotlib.projections.get_projection_names())
	for i,node in enumerate(nodes):
		xs.append(node[0])
		ys.append(node[1])
		zs.append(node[2])
		rxs.append(node[0]+res_displace[i][0]*factor)
		rys.append(node[1]+res_displace[i][1]*factor)
		rzs.append(node[2]+res_displace[i][2]*factor)

	frame_args = out_frames[0][1]
	st_nrg = 0.5*frame_args["Le"]/frame_args["E"]*(Q.T[0]**2/frame_args["Ax"]+Q.T[4]**2/frame_args["Iy"]+Q.T[5]**2/frame_args["Iz"])
	qmax = np.max(st_nrg)

	for i,frame in enumerate(frames):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
		rend   = [rxs[nid2],rys[nid2],rzs[nid2]]

		ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
		ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=0.5)#(1.0*st_nrg[i]/qmax)**2)
	'''
	for dframe in dframes:
		nid1 = int(dframe[0])
		nid2 = int(dframe[1])
		dstart = [dxs[nid1],rys[nid1],rzs[nid1]]
		dend   = [dxs[nid2],rys[nid2],rzs[nid2]]
		ax.plot([dstart[0],dend[0]],[dstart[1],dend[1]],[dstart[2],dend[2]],color='b', alpha=0.1)
	'''	

	#ax.scatter(xs,ys,zs, color='r',alpha=0.1)
	ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
	plt.show()
	#print(frames)
