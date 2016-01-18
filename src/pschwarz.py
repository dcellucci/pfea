import numpy as np


#Geometric Properties

#The location of nodes within a cubic unit cell
#Units are relative to the voxel pitch
uc_dims = np.array([1.0,1.0,1.0])

node_locs = [[0.0 ,0.25,0.5 ],
			 [0.0 ,0.5 ,0.25],
			 [0.0 ,0.5 ,0.75],
			 [0.0 ,0.75,0.5 ],
			 [0.25,0.0 ,0.5 ],
			 [0.25,0.25,0.25],
			 [0.25,0.25,0.75],
			 [0.25,0.5 ,0.0 ],
			 [0.25,0.5 ,1.0 ],
			 [0.25,0.75,0.25],
			 [0.25,0.75,0.75],
			 [0.25,1.0 ,0.5 ],
			 [0.5 ,0.0 ,0.25],
			 [0.5 ,0.0 ,0.75],
			 [0.5 ,0.25,0.0 ],
			 [0.5 ,0.25,1.0 ],
			 [0.5 ,0.75,0.0 ],
			 [0.5 ,0.75,1.0 ],
			 [0.5 ,1.0 ,0.25],
			 [0.5 ,1.0 ,0.75],
			 [0.75,0.0 ,0.5 ],
			 [0.75,0.25,0.25],
			 [0.75,0.25,0.75],
			 [0.75,0.5 ,0.0 ],
			 [0.75,0.5 ,1.0 ],
			 [0.75,0.75,0.25],
			 [0.75,0.75,0.75],
			 [0.75,1.0 ,0.5 ],
			 [1.0 ,0.25,0.5 ],
			 [1.0 ,0.5 ,0.25],
			 [1.0 ,0.5 ,0.75],
			 [1.0 ,0.75,0.5 ]]  

#References Node_locs, maps frame number to
#indices in node_locs corresponding to end-points
#The order of both the pairs and the IDs corresponds to 
#the Frame3dd convention of assigning endpoints
#see http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html
#Section 7.3

frame_locs = [[0 ,5 ],
			  [0 ,6 ],
			  [1 ,5 ],
			  [1 ,9 ],
			  [2 ,6 ],
			  [2 ,10],
			  [3 ,9 ],
			  [3 ,10],
			  [4 ,5 ],
			  [4 ,6 ],
			  [5 ,7 ],
			  [5 ,12],
			  [5 ,14],
			  [6 ,8 ],
			  [6 ,13],
			  [6 ,15],
			  [7 ,9 ],
			  [8 ,10],
			  [9 ,11],
			  [9 ,16],
			  [9 ,18],
			  [10,11],
			  [10,17],
			  [10,19],
			  [12,21],
			  [13,22],
			  [14,21],
			  [15,22],
			  [16,25],
			  [17,26],
			  [18,25],
			  [19,26],
			  [20,21],
			  [20,22],
			  [21,23],
			  [21,28],
			  [21,29],
			  [22,24],
			  [22,28],
			  [22,30],
			  [23,25],
			  [24,26],
			  [25,27],
			  [25,29],
			  [25,31],
			  [26,27],
			  [26,30],
			  [26,31]]



#TODO
#Replace node, and frame lists with numpy arrays

def from_material(mat_matrix,vox_pitch):
	size_x = len(mat_matrix)
	size_y = len(mat_matrix[0])
	size_z = int(len(mat_matrix[0][0]))
	node_frame_map = np.zeros((size_x,size_y,size_z,len(node_locs)),dtype=int)
	
	mat_dims = (np.array([size_x,size_y,size_z])-2)*uc_dims*vox_pitch

	nodes = []
	frames = []
	cur_node_id = 0
	#assumes a 1 voxel boundary of empty space
	for i in range(1,size_x-1):
		for j in range(1,size_y-1):
			for k in range(1,size_z-1):
				node_ids = []
				pos = [i-1,j-1,k-1]
				node_ids = [0]*len(node_locs)
				if(mat_matrix[i][j][k] == 1):
					tids = [0,1,2,3]
					if(mat_matrix[i-1][j][k] == 0):
						for p in range(4):
							nodes.append([(i+node_locs[tids[p]][0]-1)*vox_pitch,
										  (j+node_locs[tids[p]][1]-1)*vox_pitch, 
										  (k+node_locs[tids[p]][2]-1)*vox_pitch])
							node_ids[tids[p]] = cur_node_id
							cur_node_id = cur_node_id+1
					else:
						refids = [28,29,30,31]
						for p in range(4):
							node_ids[tids[p]] = node_frame_map[i-1][j][k][refids[p]]

					tids = [4,12,13,20]
					if(mat_matrix[i][j-1][k] == 0):
						for p in range(4):
							nodes.append([(i+node_locs[tids[p]][0]-1)*vox_pitch,
										  (j+node_locs[tids[p]][1]-1)*vox_pitch, 
										  (k+node_locs[tids[p]][2]-1)*vox_pitch])
							node_ids[tids[p]] = cur_node_id
							cur_node_id = cur_node_id+1
					else:
						refids = [11,18,19,27]
						for p in range(4):
							node_ids[tids[p]] = node_frame_map[i][j-1][k][refids[p]]

					tids = [7,14,16,23]
					if(mat_matrix[i][j][k-1] == 0):
						for p in range(4):
							nodes.append([(i+node_locs[tids[p]][0]-1)*vox_pitch,
										  (j+node_locs[tids[p]][1]-1)*vox_pitch, 
										  (k+node_locs[tids[p]][2]-1)*vox_pitch])
							node_ids[tids[p]] = cur_node_id
							cur_node_id = cur_node_id+1
					else:
						refids = [8,15,17,24]
						for p in range(4):
							node_ids[tids[p]] = node_frame_map[i][j][k-1][refids[p]]


					for q,node in enumerate(node_locs):
						if not np.in1d(0.0,node):
							nodes.append([(i+node_locs[q][0]-1)*vox_pitch,
										  (j+node_locs[q][1]-1)*vox_pitch, 
										  (k+node_locs[q][2]-1)*vox_pitch])
							node_ids[q] = int(cur_node_id)
							cur_node_id = cur_node_id+1

					#print(node_ids)
					node_frame_map[i][j][k][:] = node_ids

					### Frame Population
					#Once The node IDs for a voxel have been found, we populate
					#A list with the frame elements that compose the octahedron
					#contained within a voxel

					rel_nodes = np.array((np.array(node_locs)-0.5)*2.0,dtype=int)

					for q in range(len(frame_locs)):
						tmpid1 = frame_locs[q][0]
						tmpid2 = frame_locs[q][1]
						if -1 in rel_nodes[tmpid1] and np.array_equal(rel_nodes[tmpid1],rel_nodes[tmpid2]):
							#this assumes they are flat along a side
							if(mat_matrix[i+rel_nodes[tmpid1][0]][j+rel_nodes[tmpid1][1]][k+rel_nodes[tmpid1][2]] == 0):
								frames.append([node_ids[frame_locs[q][0]],
									   		   node_ids[frame_locs[q][1]]])
						else:
							frames.append([node_ids[frame_locs[q][0]],
									   	   node_ids[frame_locs[q][1]]])
	return nodes,frames, node_frame_map,uc_dims

def frame_length(vox_pitch):
	return vox_pitch/(2*np.sqrt(2))

def implicit(coords):
	#returns first order nodal approximation for the
	#p-schwarz surface
	#f(x,y,z) = cos(x) + cos(y) + cos(z) 
	#f(x,y,z) == 0 is the surface
	return np.cos((coords[0]-0.5)*(2*np.pi))+np.cos((coords[1]-0.5)*(2*np.pi))+np.cos((coords[2]-0.5)*(2*np.pi))

def boundary_curves(res,thresh):
	#returns all six boundary curves for the cubic unit
	#cell
	#must specify resolution
	#tval format = [i,j,0.0,1.0]

	sprops = [[2,0,1],
			  [3,0,1],
			  [0,2,1],
			  [0,3,1],
			  [0,1,2],
			  [0,1,3]]
	
	bound_curves = []

	
	for prop in sprops:
		bound_curve = []
		for i in np.arange(0.0,1.0,1.0/res):
			for j in np.arange(0.0,1.0,1.0/res):
				tval = [i,j,0.0,1.0]
				if np.abs(implicit((tval[prop[0]],tval[prop[1]],tval[prop[2]]))) < thresh:
					bound_curve.append([tval[prop[0]],tval[prop[1]],tval[prop[2]]])
		bound_curves.append(bound_curve)

	return bound_curves








