import numpy as np
import latticegen

#Geometric Properties

#The location of nodes within a cubic unit cell
#Units are relative to the voxel pitch
uc_dims = np.array([1.0,1.0,1.0])
node_locs = [[0.25,0.25,0.25],
			 [0.25,0.75,0.75],  
			 [0.75,0.25,0.75],
			 [0.75,0.75,0.25]]

#References Node_locs, maps frame number to
#indices in node_locs corresponding to end-points
#The order of both the pairs and the IDs corresponds to 
#the Frame3dd convention of assigning endpoints
#see http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html
#Section 7.3

frame_locs = [[0,1],
			  [0,2],
			  [1,3],
			  [2,3]]

#These are frames shared between nodes on a unit cell. The convention is
#[relx, rely, relz] # of unit cells away from the current cell.
#Which node goes first
#IDs of the two nodes in the two unit cells, local first

shared_frames = [[[-1, 0,-1], 1, [0,2]],
				 [[-1, 0, 0], 1, [0,3]],
				 [[-1, 0, 0], 1, [1,2]],
				 [[-1, 0, 1], 1, [1,3]],
				 [[ 0,-1,-1], 1, [0,1]],
				 [[ 0,-1, 0], 0, [0,3]],
				 [[ 0,-1, 0], 1, [2,1]],
				 [[ 0,-1, 1], 1, [2,3]]]

#Gives the shared frames for a periodic lattice.
#Initially, I'll assume 1x1x1.
#First part is the delta direction in unit cell magnitude
#second part is the start, end node
per_shared_frames = [[[0,1],[-0.5, 0.0,-0.5]],
					 [[0,2],[ 0.0,-0.5,-0.5]],
					 [[0,3],[-0.5, 0.5, 0.0]],
					 [[0,3],[ 0.5,-0.5, 0.0]],
					 [[1,3],[ 0.0,-0.5, 0.5]],
					 [[1,2],[ 0.5, 0.5, 0.0]],
					 [[1,2],[-0.5,-0.5, 0.0]],
					 [[2,3],[-0.5, 0.0, 0.5]]]
#TODO
#Replace node, and frame lists with numpy arrays

def from_material(mat_matrix,vox_pitch):
	size_x = len(mat_matrix)
	size_y = len(mat_matrix[0])
	size_z = len(mat_matrix[0][0])#/uc_dims[2]

	mat_dims = (np.array([size_x,size_y,size_z])-2)*uc_dims*vox_pitch
	node_frame_map = np.zeros((size_x,size_y,size_z,4))

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
				node_ids = []
				pos = [i,j,k]
				if(mat_matrix[i][j][k] == 1):
					for node in node_locs:
						nodes.append([(node[m]+pos[m]-1)*vox_pitch for m in range(3)])
						node_ids.append(cur_node_id)
						cur_node_id = cur_node_id+1
					
					node_frame_map[i][j][k][0:4] = node_ids

					### Frame Population
					#Once The node IDs for a voxel have been found, we populate
					#A list with the frame elements that compose the octahedron
					#contained within a voxel
					for q in range(0,4):
						frames.append([node_ids[frame_locs[q][0]],
									   node_ids[frame_locs[q][1]]])

					for shared_frame in shared_frames:
						rel_pos = [pos[m]+shared_frame[0][m] for m in range(3)]
						if(mat_matrix[rel_pos[0]][rel_pos[1]][rel_pos[2]] == 1):
							nlid = node_frame_map[rel_pos[0]][rel_pos[1]][rel_pos[2]][shared_frame[2][1]]
							tframe = [node_ids[shared_frame[2][0]]]*2
							if shared_frame[1] == 0:
								tframe[1] = nlid
							else:
								tframe[0] = nlid
							frames.append(tframe)

	return nodes,frames,node_frame_map,mat_dims

#We want to make a (111) oriented version of
#DSchwarz, but there's no clean/nice unit cell for this
#So this version just takes a cubic block, rotates it, and
#Extracts out a large hexagonal extrusion from this.
def gen_111(hex_radius,hex_height,vox_pitch):
	offset = np.array([0.25,0.25,0.25])
	debug,mat_matrix,invol = latticegen.cubic_to_111(hex_radius,hex_height,np.array(node_locs),np.array(frame_locs),offset)
	nodes,frames,node_frame_map,dims = from_material(mat_matrix,vox_pitch)

	return latticegen.crop_framework(nodes,frames,node_frame_map,invol)

def gen_111_invol(hex_radius,hex_height,vox_pitch):
	offset = np.array([0.25,0.25,0.25])
	debug,mat_matrix,invol = latticegen.cubic_to_111(hex_radius,hex_height,np.array(node_locs),np.array(frame_locs),offset)
	return invol

def frame_length(vox_pitch):
	return 0.707*vox_pitch

def implicit(coords):
	#returns first order nodal approximation for the
	#p-schwarz surface
	#f(x,y,z) = cos(x) + cos(y) + cos(z) 
	#f(x,y,z) == 0 is the surface
	coords = np.array(coords)
	coords = (coords - 0.5)*(2*np.pi)
	return np.sin(coords[0])*np.sin(coords[1])*np.sin(coords[2]) + np.sin(coords[0])*np.cos(coords[1])*np.cos(coords[2]) + np.cos(coords[0])*np.sin(coords[1])*np.cos(coords[2]) + np.cos(coords[0])*np.cos(coords[1])*np.sin(coords[2]) 

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
