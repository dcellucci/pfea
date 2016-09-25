import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess
import cProfile
from math import *

import pfea
import pfea.frame3dd
import pfea.solver
import pfea.geom.cuboct as cuboct


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

size_x = 3
size_y = 3
size_z = 4

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


#Node Map Population
#Referencing the geometry-specific cuboct.py file. 
#Future versions might have different files?

node_frame_map = np.zeros((size_x,size_y,size_z,6))
nodes,frames,node_frame_map,temp = cuboct.from_material(mat_matrix,vox_pitch)


num_nodes = size_x*size_y*2 + size_x + size_y

#pressure = 1000000 #N/m^2

#node_load = pressure*size_x*size_y*vox_pitch*vox_pitch/num_nodes

#print(node_load)

node_load = (size_z-1)*vox_pitch*0.01

#Constraint and load population
constraints = []
loads = []
#The test article terminates at the half-voxel, so the plane of nodes at the halfway point are constrained. 

#for x in range(1,size_x+1):
#	for y in range(1,size_y+1):
#		frames, nodes,node_frame_map = cuboct.remove_node([x,y,1,2],node_frame_map,frames,nodes)
#		frames, nodes,node_frame_map = cuboct.remove_node([x,y,size_z,5],node_frame_map,frames,nodes)

for x in range(1,size_x+1):
	for y in range(1,size_y+1):
		#The bottom-most nodes are constrained to neither translate nor
		#rotate
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':2, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':5, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':2, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':5, 'value':0})


		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':2, 'value':node_load})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':5, 'value':0})

		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':2, 'value':node_load})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':5, 'value':0})

		#The top most nodes are assigned a z-axis load, as well as being
		#constrained to translate in only the z-direction.
		#loads.append(      {'type':'disp','node':node_frame_map[x][y][size_z][0],'DOF':2, 'value':node_load})
		#loads.append(      {'type':'disp','node':node_frame_map[x][y][size_z][1],'DOF':2, 'value':node_load})

		if x == size_x:
			constraints.append({'node':node_frame_map[x][y][1][3],'DOF':0, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][3],'DOF':1, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][3],'DOF':2, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][3],'DOF':3, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][3],'DOF':4, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][3],'DOF':5, 'value':0})

			constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':0, 'value':0})
			constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':1, 'value':0})
			constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':2, 'value':node_load})
			constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':3, 'value':0})
			constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':4, 'value':0})
			constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':5, 'value':0})
			
			#loads.append(      {'type':'disp','node':node_frame_map[x][y][size_z][3],'DOF':2, 'value':node_load})

		if y == size_y:
			constraints.append({'node':node_frame_map[x][y][1][4],'DOF':0, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][4],'DOF':1, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][4],'DOF':2, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][4],'DOF':3, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][4],'DOF':4, 'value':0})
			constraints.append({'node':node_frame_map[x][y][1][4],'DOF':5, 'value':0})

			constraints.append({'node':node_frame_map[x][y][size_z][4],'DOF':0, 'value':0})
			constraints.append({'node':node_frame_map[x][y][size_z][4],'DOF':1, 'value':0})
			constraints.append({'node':node_frame_map[x][y][size_z][4],'DOF':2, 'value':node_load})
			constraints.append({'node':node_frame_map[x][y][size_z][4],'DOF':3, 'value':0})
			constraints.append({'node':node_frame_map[x][y][size_z][4],'DOF':4, 'value':0})
			constraints.append({'node':node_frame_map[x][y][size_z][4],'DOF':5, 'value':0})

			#loads.append(      {'type':'disp','node':node_frame_map[x][y][size_z][4],'DOF':2, 'value':node_load})
		
		'''
		frames = cuboct.remove_frame([(x,y,1),1],node_frame_map,frames)
		frames = cuboct.remove_frame([(x,y,1),4],node_frame_map,frames)
		frames = cuboct.remove_frame([(x,y,1),7],node_frame_map,frames)
		frames = cuboct.remove_frame([(x,y,1),8],node_frame_map,frames)

		frames = cuboct.remove_frame([(x,y,size_z),2],node_frame_map,frames)
		frames = cuboct.remove_frame([(x,y,size_z),5],node_frame_map,frames)
		frames = cuboct.remove_frame([(x,y,size_z),9],node_frame_map,frames)
		frames = cuboct.remove_frame([(x,y,size_z),10],node_frame_map,frames)
		'''
		#constraints.append({'node':node_ids[5],'DOF':2, 'value':0})

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
global_args = {'frame3dd_filename': "test", 'length_scaling':1,"using_Frame3dd":False,"debug_plot":True,"gravity":[0,0,0]}

if global_args["using_Frame3dd"]:
	frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
	subprocess.call("/Users/dcellucci/Box\ Sync/KCS_NSTRF/Science/Structure_Work/Frame3DD/src/frame3dd {0}.csv {0}.out".format(global_args["frame3dd_filename"]), shell=True)
	res_nodes, res_reactions = frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
	res_displace = frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])
else:
	#displace = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)
	cProfile.run('res_displace,C,Q = pfea.solver.analyze_System(out_nodes, global_args, out_frames, constraints,loads)')
	#def_displace = pfea.analyze_System(out_nodes, global_args, def_frames, constraints,loads)
#print(res_displace[node_frame_map[int(size_x/2.0)+1][int(size_y/2.0)+1][size_z][4]][2])
#print(res_displace[node_frame_map[int(size_x/2.0)+1][int(size_y/2.0)+1][size_z][4]][2]/(size_z*vox_pitch))

top_f_mag = np.zeros((size_x,size_y,3))
tot_force = 0
#print(np.shape(res_reactions))

for i in range(1,size_x+1):
	for j in range(1,size_y+1):
		force_vec = res_displace[int(node_frame_map[i][j][size_z][0])]
		#print(nodes[int(node_frame_map[i][j][size_z][0])])
		#print(force_vec)
		tot_force = tot_force + np.sqrt(force_vec.dot(force_vec))
		force_vec = res_displace[node_frame_map[i][j][size_z][1]]
		tot_force = tot_force + np.sqrt(force_vec.dot(force_vec))
		if i == size_x:
			force_vec = res_displace[node_frame_map[i][j][size_z][3]]
			tot_force = tot_force + np.sqrt(force_vec.dot(force_vec))
		if j == size_y:
			force_vec = res_displace[node_frame_map[i][j][size_z][4]]
			tot_force = tot_force + np.sqrt(force_vec.dot(force_vec))

print(tot_force/(size_x*size_y*vox_pitch*vox_pitch))

if global_args["debug_plot"]:
	### Right now the debug plot only does x-y-z displacements, no twisting
	xs = []
	ys = []
	zs = []

	rxs = []
	rys = []
	rzs = []

	dxs = []
	dys = []
	dzs = []

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_aspect('equal')
	frame_coords = []

	#print(matplotlib.projections.get_projection_names())
	for i,node in enumerate(nodes):
		if(node[2] == (size_z-0.5)*vox_pitch):
			dxs.append(node[0])
			dys.append(node[1])
			dzs.append(node[2])
		else:
			xs.append(node[0])
			ys.append(node[1])
			zs.append(node[2])
		
		rxs.append(node[0]+res_displace[i][0])
		rys.append(node[1]+res_displace[i][1])
		rzs.append(node[2]+res_displace[i][2])

	'''
	for i,frame in enumerate(frames):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
		rend   = [rxs[nid2],rys[nid2],rzs[nid2]]

		ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
		ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=0.3)
	
	for dframe in dframes:
		nid1 = int(dframe[0])
		nid2 = int(dframe[1])
		dstart = [dxs[nid1],rys[nid1],rzs[nid1]]
		dend   = [dxs[nid2],rys[nid2],rzs[nid2]]
		ax.plot([dstart[0],dend[0]],[dstart[1],dend[1]],[dstart[2],dend[2]],color='b', alpha=0.1)
	'''	

	ax.scatter(xs,ys,zs, color='r',alpha=0.1)
	ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
	ax.scatter(dxs,dys,dzs,color='m')
	plt.show()
	#print(frames)
