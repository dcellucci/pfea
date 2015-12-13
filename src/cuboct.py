import numpy as np


#Geometric Properties

#The location of nodes within a cubic unit cell
#Units are relative to the voxel pitch

uc_dims = [1.0,1.0,1.0]

node_locs = [[0.0,0.5,0.5],
			 [0.5,0.0,0.5],  
			 [0.5,0.5,0.0],
			 [1.0,0.5,0.5],
			 [0.5,1.0,0.5],
			 [0.5,0.5,1.0]]

#References Node_locs, maps frame number to
#indices in node_locs corresponding to end-points
#The order of both the pairs and the IDs corresponds to 
#the Frame3dd convention of assigning endpoints
#see http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html
#Section 7.3

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

#TODO
#Replace node, and frame lists with numpy arrays

def from_material(mat_matrix,vox_pitch):
	size_x = len(mat_matrix)
	size_y = len(mat_matrix[0])
	size_z = len(mat_matrix[0][0])
	node_frame_map = np.zeros((size_x,size_y,size_z,6))

	mat_dims = (np.array([size_x,size_y,size_z])-2)*uc_dims*vox_pitch

	nodes = []
	frames = []
	cur_node_id = 0
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
	for i in range(1,size_x-1):
		for j in range(1,size_y-1):
			for k in range(1,size_z-1):
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
	return nodes,frames,node_frame_map,uc_dims

def frame_length(vox_pitch):
	return 1.0*vox_pitch/np.sqrt(2.0)

def remove_frame(loc,node_frame_map,frames):
	# loc is a list containing a (1,3) tuple with the x,y,z location of the voxel
	# and an int with the index of the desired frame to remove.
	id1 = node_frame_map[loc[0][0]][loc[0][1]][loc[0][2]][frame_locs[loc[1]][0]]
	id2 = node_frame_map[loc[0][0]][loc[0][1]][loc[0][2]][frame_locs[loc[1]][1]]

	c_frame = np.copy(np.array(frames))
	for i,frame in enumerate(frames):
		if frame[0] == id1 and frame[1] == id2:
			c_frame = np.delete(c_frame,i,0)

	return c_frame

def remove_node(loc, node_frame_map,frames,nodes):
	# loc is a list containing a the x,y,z location of the voxel
	# and the index of the desired node to remove.
	# removes the node, and all nodes connected to the node.
	nid = int(node_frame_map[loc[0]][loc[1]][loc[2]][loc[3]])
	remove_index = []

	for i,frame in enumerate(frames):
		if nid in frame:
			remove_index.append(i)
		if frame[0] > nid:
			frame[0] = frame[0]-1
		if frame[1] > nid:
			frame[1] = frame[1]-1

	for i,row in enumerate(node_frame_map):
		for j,col in enumerate(node_frame_map):
			for k,dep in enumerate(node_frame_map):
				for l,item in enumerate(node_frame_map):
					if(node_frame_map[i][j][k][l] > nid):
						node_frame_map[i][j][k][l] = node_frame_map[i][j][k][l]-1
	print(remove_index)
	return np.delete(frames,remove_index,0), np.delete(nodes,nid,0),node_frame_map

