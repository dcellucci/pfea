import numpy as np
import latticegen

#Geometric Properties

#The location of nodes within a cubic unit cell
#Units are relative to the voxel pitch
uc_dims = np.array([1.0,1.0,1.0])

node_locs = [[0.0 ,0.25,0.5 ],
			 [0.0 ,0.5 ,0.25],
			 [0.0 ,0.5 ,0.75],
			 [0.0 ,0.75,0.5 ],
			 [0.25,0.0 ,0.5 ],
			 [0.25,0.5 ,0.0 ],
			 [0.25,0.5 ,1.0 ],
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
			 [0.75,0.5 ,0.0 ],
			 [0.75,0.5 ,1.0 ],
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

frame_locs = [[0 ,1 ],
			  [0 ,2 ],
			  [0 ,4 ],
			  [1 ,3 ],
			  [1 ,5 ],
			  [2 ,3 ],
			  [2 ,6 ],
			  [3 ,7 ],
			  [4 ,8 ],
			  [4 ,9 ],
			  [5 ,10],
			  [5 ,12],
			  [6 ,11],
			  [6 ,13],
			  [7 ,14],
			  [7 ,15],
			  [8 ,10],
			  [8 ,16],
			  [9 ,11],
			  [9 ,16],
			  [10,17],
			  [11,18],
			  [12,14],
			  [12,17],
			  [13,15],
			  [13,18],
			  [14,19],
			  [15,19],
			  [16,20],
			  [17,21],
			  [18,22],
			  [19,23],
			  [20,21],
			  [20,22],
			  [21,23],
			  [22,23]]


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
						refids = [20,21,22,23]
						for p in range(4):
							node_ids[tids[p]] = node_frame_map[i-1][j][k][refids[p]]

					tids = [4,8,9,16]
					if(mat_matrix[i][j-1][k] == 0):
						for p in range(4):
							nodes.append([(i+node_locs[tids[p]][0]-1)*vox_pitch,
										  (j+node_locs[tids[p]][1]-1)*vox_pitch, 
										  (k+node_locs[tids[p]][2]-1)*vox_pitch])
							node_ids[tids[p]] = cur_node_id
							cur_node_id = cur_node_id+1
					else:
						refids = [7,14,15,19]
						for p in range(4):
							node_ids[tids[p]] = node_frame_map[i][j-1][k][refids[p]]

					tids = [5,10,12,17]
					if(mat_matrix[i][j][k-1] == 0):
						for p in range(4):
							nodes.append([(i+node_locs[tids[p]][0]-1)*vox_pitch,
										  (j+node_locs[tids[p]][1]-1)*vox_pitch, 
										  (k+node_locs[tids[p]][2]-1)*vox_pitch])
							node_ids[tids[p]] = cur_node_id
							cur_node_id = cur_node_id+1
					else:
						refids = [6,11,13,18]
						for p in range(4):
							node_ids[tids[p]] = node_frame_map[i][j][k-1][refids[p]]

					outerids = [6,11,13,18,7,14,15,19,20,21,22,23]

					for q in range(len(outerids)):
						nodes.append([(i+node_locs[outerids[q]][0]-1)*vox_pitch,
									  (j+node_locs[outerids[q]][1]-1)*vox_pitch, 
									  (k+node_locs[outerids[q]][2]-1)*vox_pitch])
						node_ids[outerids[q]] = cur_node_id
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

def gen_111(hex_radius,hex_height,vox_pitch):
	offset = np.array([0.25,0.25,0.25])
	debug,mat_matrix,invol = latticegen.cubic_to_111(hex_radius,hex_height,np.array(node_locs),np.array(frame_locs),offset)
	nodes,frames,node_frame_map,dims = from_material(mat_matrix,vox_pitch)

	return latticegen.crop_framework(nodes,frames,node_frame_map,invol)

def frame_length(vox_pitch):
	return vox_pitch/(2*np.sqrt(2))

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

