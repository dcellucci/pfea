####################################################################
####################################################################
## The 'font' is Banner from http://www.network-science.de/ascii/ ##
####################################################################
####################################################################


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import frame3dd
import subprocess
import pfea
import cProfile
from math import *


## Python Script for Converting DM files to .3dd Files

      #  #####  ####### #     #    ### #     # ######  ####### ######  ####### 
      # #     # #     # ##    #     #  ##   ## #     # #     # #     #    #    
      # #       #     # # #   #     #  # # # # #     # #     # #     #    #    
      #  #####  #     # #  #  #     #  #  #  # ######  #     # ######     #    
#     #       # #     # #   # #     #  #     # #       #     # #   #      #    
#     # #     # #     # #    ##     #  #     # #       #     # #    #     #    
 #####   #####  ####### #     #    ### #     # #       ####### #     #    #    
'''                                                                               
import json
from pprint import pprint

with open('data/DM assembly.json') as data_file:    
    data = json.load(data_file)
'''


 #####                      #######              
#     # #    # #####        #     #  ####  ##### 
#       #    # #    #       #     # #    #   #   
#       #    # #####  ##### #     # #        #   
#       #    # #    #       #     # #        #   
#     # #    # #    #       #     # #    #   #   
 #####   ####  #####        #######  ####    #   
                                                 
######                                                           
#     #  ####  #####  #    # #        ##   ##### #  ####  #    # 
#     # #    # #    # #    # #       #  #    #   # #    # ##   # 
######  #    # #    # #    # #      #    #   #   # #    # # #  # 
#       #    # #####  #    # #      ######   #   # #    # #  # # 
#       #    # #      #    # #      #    #   #   # #    # #   ## 
#        ####  #       ####  ###### #    #   #   #  ####  #    # 
                                                                 



#Temporary Material Matrix - NxNxN cubic grid (corresponding to cubic-octahedra)
# at the moment:
# 1's correspond to material being there
# 0's correspond to no material

size_x = 6
size_y = 6
size_z = 6
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



# Material Properties

#Physical Voxel Properties
vox_pitch = 0.01 #m

#NODE PROPERTIES
#The location of nodes within a cubic unit cell
#Units are relative to the voxel pitch (see above)
node_locs = [[0.0,0.5,0.5],
			 [0.5,0.0,0.5],  
			 [0.5,0.5,0.0],
			 [1.0,0.5,0.5],
			 [0.5,1.0,0.5],
			 [0.5,0.5,1.0]]
node_radius = 0 

#STRUT PROPERTIES
#Physical Properties
#Assuming a square, Ti6Al4V strut that is 0.5 mm on a side
#Using Newtons, meters, and kilograms as the units
frame_props = {"nu"  : 0.33, #poisson's ratio
			   "dx"	 : 0.0005, #m
			   "dy"	 : 0.0005, #m
			   "E"   :  116000000000, #N/m^2
			   "G"   :   42000000000,  #N/m^2
			   "rho" :  4420, #kg/m^3
			   "beam_divisions" : 0,
			   "cross_section"  : 'rectangular',
			   "roll": 0,
			   "Le":vox_pitch/sqrt(2.0)} 

#Geometric Properties
#References Node_locs, maps frame number to
#indices in node_locs corresponding to end-points
#The order of both the pairs and the IDs corresponds to 
frame_locs = [[0,1],
			  [0,2],
			  [0,5],
			  [0,4],
			  [1,2],
			  [1,5],
			  [1,3],
			  [2,4],
			  [2,3],
			  [5,3],
			  [5,4],
			  [4,3]]

#Node Map Population
#This builds the map of node IDs for each voxel.
#The format is node_frame_map[x_coord_vox][y_coord_vox][z_coord_vox][id]
#Since A voxel can share six nodes with its neighbors, sorting out which
#voxel shares which nodes with which can be confusing
#My approach right now is to assign IDs in a raster x>y>z, so there is
#always a consistent notion of what nodes have been assigned, and which
#are free.

#ASIDE:
#There are perhaps more efficient ways of distributing id's so that
#the resulting stiffness matrix is as diagonal as possible 
#That is, over all elements Ei with node ids ni1 and ni2, Sum(abs(ni2-ni1)) 
#is minimized. 
#For large numbers of nodes, this would be *extremely* useful.

#To make ID assignment more compact, a 1-voxel border of empty voxels
#surrounds the material matrix. That way, we can treat edges, corners and
#vacancies as the same problem, from a node-assignment perspective.

#TODO
#Replace node, and frame lists with numpy arrays

node_frame_map = np.zeros((size_x+2,size_y+2,size_z+2,6))

nodes = []
frames = []
constraints = []
loads = []
cur_node_id = 0

for i in range(1,size_x+1):
	for j in range(1,size_y+1):
		for k in range(1,size_z+1):
			node_ids = [0]*6
			if(mat_matrix[i][j][k] == 1):
				if(mat_matrix[i-1][j][k] == 0):
					nodes.append([(i+node_locs[0][0]-1)*vox_pitch,
								  (j+node_locs[0][1]-1)*vox_pitch, 
								  (k+node_locs[0][2]-1)*vox_pitch])
					node_ids[0] = cur_node_id
					cur_node_id = cur_node_id+1
				else:
					node_ids[0] = node_frame_map[i-1][j][k][3]

				if(mat_matrix[i][j-1][k] == 0):
					nodes.append([(i+node_locs[1][0]-1)*vox_pitch,
								  (j+node_locs[1][1]-1)*vox_pitch, 
								  (k+node_locs[1][2]-1)*vox_pitch])
					node_ids[1] = cur_node_id
					cur_node_id = cur_node_id+1
				else:
					node_ids[1] = node_frame_map[i][j-1][k][4]

				if(mat_matrix[i][j][k-1] == 0):
					nodes.append([(i+node_locs[2][0]-1)*vox_pitch,
								  (j+node_locs[2][1]-1)*vox_pitch, 
								  (k+node_locs[2][2]-1)*vox_pitch])
					node_ids[2] = cur_node_id
					cur_node_id = cur_node_id+1
				else:
					node_ids[2] = node_frame_map[i][j][k-1][5]

				for q in range(3,6):
					nodes.append([(i+node_locs[q][0]-1)*vox_pitch,
								  (j+node_locs[q][1]-1)*vox_pitch, 
								  (k+node_locs[q][2]-1)*vox_pitch])
					node_ids[q] = cur_node_id
					cur_node_id = cur_node_id+1
				
				node_frame_map[i][j][k][0:6] = node_ids

				### Frame Population
				#Once The node IDs for a voxel have been found, we populate
				#A list with the frame elements that compose the octahedron
				#contained within a voxel
				for q in range(0,12):
					frames.append([node_ids[frame_locs[q][0]],
								   node_ids[frame_locs[q][1]]])
				#Constraints are added based on simple requirements right now
				#The bottom-most nodes are constrained to neither translate nor
				#rotate
				if k == 1:
					constraints.append({'node':node_ids[2],'DOF':0, 'value':0})
					constraints.append({'node':node_ids[2],'DOF':1, 'value':0})
					constraints.append({'node':node_ids[2],'DOF':2, 'value':0})
					constraints.append({'node':node_ids[2],'DOF':3, 'value':0})
					constraints.append({'node':node_ids[2],'DOF':4, 'value':0})
					constraints.append({'node':node_ids[2],'DOF':5, 'value':0})
				#The top most nodes are assigned a z-axis load, as well as being
				#constrained to translate in only the z-direction.
				if k == size_z:
					loads.append({'node':node_ids[5],'DOF':2,'value':-5.0})
					constraints.append({'node':node_ids[5],'DOF':0, 'value':0})
					constraints.append({'node':node_ids[5],'DOF':1, 'value':0})
					#constraints.append({'node':node_ids[5],'DOF':2, 'value':0})




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
								 'd1'  : frame_props["dx"],
								 'd2'  : frame_props["dy"],
								 'beam_divisions' : frame_props["beam_divisions"],
								 'cross_section'  : frame_props["cross_section"],
								 'roll': frame_props["roll"],
								 'loads':{'element':0},
								 'prestresses':{'element':0},
								 'Le': frame_props["Le"]})]

#Format node positions
out_nodes = np.array(nodes)

#Global Arguments 
global_args = {'frame3dd_filename': "test", 'length_scaling':1000,"using_Frame3dd":False,"debug_plot":False}

if global_args["using_Frame3dd"]:
	frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
	subprocess.call("frame3dd {0}.csv {0}.out".format(global_args["frame3dd_filename"]), shell=True)
	res_nodes, res_reactions = frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
	res_displace = frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])
else:
	#res_displace = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)
	res_displace = cProfile.run('pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)')


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
	for i,node in enumerate(nodes):
		xs.append(node[0])
		ys.append(node[1])
		zs.append(node[2])
		rxs.append(node[0]+res_displace[i][0])
		rys.append(node[1]+res_displace[i][1])
		rzs.append(node[2]+res_displace[i][2])

	for frame in frames:
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = nodes[nid1]
		end = nodes[nid2]
		rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
		rend = [rxs[nid1],rys[nid1],rzs[nid1]]
		ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
		ax.plot([rxs[nid1],rxs[nid2]],[rys[nid1],rys[nid2]],[rzs[nid1],rzs[nid2]],color='b',alpha=0.3)



	ax.scatter(xs,ys,zs, color='r',alpha=0.1)
	ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
	plt.show()
	#print(frames)
