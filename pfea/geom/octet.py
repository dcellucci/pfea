import numpy as np
import util.latticegen

#Geometric Properties

#The location of nodes within a cubic unit cell
#Units are relative to the voxel pitch

uc_dims = [1.0,1.0,1.0]

node_locs = [[0.0,0.0,0.0],
			 [0.0,0.0,1.0],
			 [0.0,0.5,0.5],
			 [0.0,1.0,0.0],
			 [0.0,1.0,1.0],
			 [0.5,0.0,0.5],
			 [0.5,0.5,0.0],
			 [0.5,0.5,1.0],
			 [0.5,1.0,0.5],
			 [1.0,0.0,0.0],
			 [1.0,0.0,1.0],
			 [1.0,0.5,0.5],
			 [1.0,1.0,0.0],
			 [1.0,1.0,1.0]]

#References Node_locs, maps frame number to
#indices in node_locs corresponding to end-points
#The order of both the pairs and the IDs corresponds to
#the Frame3dd convention of assigning endpoints
#see http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html
#Section 7.3

frame_locs = [[0,2],
			  [0,5],
			  [0,6],
			  [1,2],
			  [1,5],
			  [1,7],
			  [2,3],
			  [2,4],
			  [2,5],
			  [2,6],
			  [2,7],
			  [2,8],
			  [3,6],
			  [3,8],
			  [4,7],
			  [4,8],
			  [5,6],
			  [5,7],
			  [5,9],
			  [5,10],
			  [5,11],
			  [6,8],
			  [6,9],
			  [6,11],
			  [6,12],
			  [7,8],
			  [7,10],
			  [7,11],
			  [7,13],
			  [8,11],
			  [8,12],
			  [8,13],
			  [9,11],
			  [10,11],
			  [11,12],
			  [11,13]]

#TODO
#Replace node, and frame lists with numpy arrays

def from_material(mat_matrix,vox_pitch):
	size_x = len(mat_matrix)
	size_y = len(mat_matrix[0])
	size_z = len(mat_matrix[0][0])
	node_frame_map = np.zeros((size_x,size_y,size_z,len(node_locs)))

	mat_dims = (np.array([size_x,size_y,size_z])-2)*uc_dims*vox_pitch

	nodes = []
	frames = []
	cur_node_id = 0

	rel_coords = np.array(2.0*(np.array(node_locs)-0.5),dtype=int)

	nei_coords = [[-1, 0, 0],[ 0,-1, 0],[ 0, 0,-1],
				  [-1,-1, 0],[-1, 0,-1],[ 0,-1,-1],[-1,-1,-1],
			      [-1, 0, 1],[ 0,-1, 1],[-1,-1, 1]]
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
				node_ids = [0]*len(node_locs)
				if(mat_matrix[i][j][k] == 1):
					for q,node in enumerate(node_locs):
						abs_coord = [(i+node[0]-1)*vox_pitch,
									 (j+node[1]-1)*vox_pitch,
									 (k+node[2]-1)*vox_pitch]
						elsewhere = False
						if -1 in rel_coords[q]:
							for nei_coord in nei_coords:
								if mat_matrix[i+nei_coord[0]][j+nei_coord[1]][k+nei_coord[2]] == 1 and not elsewhere:
									for neid in node_frame_map[i+nei_coord[0]][j+nei_coord[1]][k+nei_coord[2]]:
										node_dist = np.linalg.norm(np.array(abs_coord)-np.array(nodes[int(neid)]))
										if node_dist < 0.001*vox_pitch and not elsewhere:
											elsewhere = True
											node_ids[q] = int(neid)
						if not elsewhere:
							nodes.append(abs_coord)
							node_ids[q] = cur_node_id
							cur_node_id = cur_node_id+1

					node_frame_map[i][j][k][0:len(node_locs)] = node_ids
					
					### Frame Population
					#Once The node IDs for a voxel have been found, we populate
					#A list with the frame elements that compose the octahedron
					#contained within a voxel
					for q in range(len(frame_locs)):
						tmpid1 = frame_locs[q][0]
						tmpid2 = frame_locs[q][1]
						shared = False
						if any(rel_coords[tmpid1]==-1):
							for p in range(0,3):
								if rel_coords[tmpid1][p]==-1 and rel_coords[tmpid1][p]==rel_coords[tmpid2][p]:
									shared = True
						#this assumes they are flat along a side
						if shared:
							if(mat_matrix[i+rel_coords[tmpid1][0]][j+rel_coords[tmpid1][1]][k+rel_coords[tmpid1][2]] == 0):
								frames.append([node_ids[frame_locs[q][0]],
									   		   node_ids[frame_locs[q][1]]])
						else:
							frames.append([node_ids[frame_locs[q][0]],
									   	   node_ids[frame_locs[q][1]]])
					#Constraints are added based on simple requirements right now
					#The bottom-most nodes are constrained to neither translate nor
					#rotate
	return nodes,frames,node_frame_map,uc_dims

def gen_111(hex_radius,hex_height,vox_pitch):
	offset = np.array([0.25,0.25,0.25])
	debug,mat_matrix,invol = latticegen.cubic_to_111(hex_radius,hex_height,np.array(node_locs),np.array(frame_locs),offset)
	nodes,frames,node_frame_map,dims = from_material(mat_matrix,vox_pitch)

	return latticegen.crop_framework(nodes,frames,node_frame_map,invol)

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
