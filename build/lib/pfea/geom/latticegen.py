import numpy as np

#import matplotlib.pyplot as plt
#import matplotlib
#from mpl_toolkits.mplot3d import Axes3D

######  #######  #####   #####  ######  ### ######  ####### ### ####### #     # 
#     # #       #     # #     # #     #  #  #     #    #     #  #     # ##    # 
#     # #       #       #       #     #  #  #     #    #     #  #     # # #   # 
#     # #####    #####  #       ######   #  ######     #     #  #     # #  #  # 
#     # #             # #       #   #    #  #          #     #  #     # #   # # 
#     # #       #     # #     # #    #   #  #          #     #  #     # #    ## 
######  #######  #####   #####  #     # ### #          #    ### ####### #     #

# (111) Lattice generation takes a hexagon size and height and outputs
# a material matrix in cubic coordinates that, when transformed to the
# (111) coordinate system, fills this hexagonal volume.

# The (111) coordinate system is defined by the cubic coordinate system
# vectors listed in the columns of the transform matrix shown below. 

tform111 = np.array([[ 1.0/np.sqrt(2) , -1.0/np.sqrt(6), 1/np.sqrt(3)],
					 [-1.0/np.sqrt(2) , -1.0/np.sqrt(6), 1/np.sqrt(3)],
					 [ 			  0.0 ,np.sqrt(2.0/3.0), 1/np.sqrt(3)]])

tform111 = np.linalg.inv(tform111)

# The hexagonal volume is defined as being a certain number of unit cells
# XY plane view
#        /\y
#      __||__
#     /\ || /\
#    /  \||/  \
#   /____\/O___\__>x
#   \ r  /\    /
#    \  /  \  /
#     \/____\/

# A unit cell in the hexagon is a triangular unit, whose side length
# is related to the cube dimension by a factor of sqrt(2). 
# Since the unit cell has side length 1:

hex_triangle_width = np.sqrt(2)

# In the z-direction, the units are in cube-diagonal lengths, or

hex_base_height = np.sqrt(3)

# Two utilities simplify checking whether a point is in this
# hexagonal volume, a 30 and 60 degree rotation about the [0,0,1] axis

# Transform matrix that rotates around the [0,0,1] axis by 30 degrees
init_hex_rot_tform = 0.5*np.array([[np.sqrt(3),        -1 ,0],
						           [        1 , np.sqrt(3),0],
						           [        0 ,         0 ,2]])

# Transform matrix that rotates around the [0,0,1] axis by 60 degrees
hex_rot_tform = 0.5*np.array([[        1 ,-np.sqrt(3),0],
	 						       [np.sqrt(3),         1 ,0],
	 						       [        0 ,         0 ,2]])

def cubic_to_111(hex_radius,hex_height,cubic_nodes,cubic_frames,offset):
	# offset: sometimes the base 111 plane isnt at the origin but another
	#         point within the unit cell. this 3x1 numpy array compensates
	#         for that

	#This part probably needs work. Finding maximum mat_matrix extents
	#for a given hex_radius and hex_height

	extents = int(np.max([3.0*hex_radius,3.0*hex_height+1]))

	#extents = extents+(3-extents%3)
	mat_matrix = np.zeros((extents+2,extents+2,extents+2))
	invol 	   = np.zeros((extents+2,extents+2,extents+2,len(cubic_nodes)))
	#
	#creation of the hex volume wireframe
	#

	#first node
	hexnode1 = np.array([hex_radius*hex_triangle_width,0,0])

	hex_nodes = np.array([hexnode1])

	#Applying 60 degree rotation to get the other base nodes
	for i in range(5):
		hexnode1 = np.dot(hex_rot_tform,hexnode1)
		hex_nodes = np.append(hex_nodes,[np.copy(hexnode1)],axis=0)

	# Translating base nodes in the (001) direction to get top nodes
	hex_nodes = np.append(hex_nodes,hex_nodes+np.array([0,0,hex_base_height*hex_height]),axis=0)

	# Frames are defined as relationships between nodes
	# A visual aid to just help with seeing the volume as a wireframe
	hex_frames = [[ 0, 1],
				  [ 0, 6],
				  [ 1, 2],
				  [ 1, 7],
				  [ 2, 3],
				  [ 2, 8],
				  [ 3, 4],
				  [ 3, 9],
				  [ 4, 5],
				  [ 4,10],
				  [ 5, 0],
				  [ 5,11],
				  [ 6, 7],
				  [ 7, 8],
				  [ 8, 9],
				  [ 9,10],
				  [10,11],
				  [11, 6]]


	tot_cube_nodes = np.dot(tform111,np.copy(cubic_nodes-offset).T).T

	tot_cube_frames = np.copy(cubic_frames)

	temptformframe = np.copy(cubic_frames)
	print(extents,-extents/3,extents-extents/3)
	for delx in range(-extents/3,extents-extents/3):
		for dely in range(-extents/3,extents-extents/3):
			for delz in range(-extents/3,extents-extents/3):
				tformcube = cubic_nodes+np.array([delx,dely,delz])-offset
				tformcube = np.transpose(np.dot(tform111,np.transpose(tformcube)))
				volcheck, ptsinvol = in_hex_volume(np.copy(tformcube),hex_radius*hex_triangle_width,hex_height*hex_base_height)
				if volcheck:
					tot_cube_nodes = np.append(tot_cube_nodes,tformcube,axis=0)
					temptformframe = temptformframe + np.array([len(cubic_nodes),len(cubic_nodes)])
					tot_cube_frames = np.append(tot_cube_frames,temptformframe,axis=0)
					mat_matrix[delx+extents/3+1][dely+extents/3+1][delz+extents/3+1] = 1
					invol[delx+extents/3+1][dely+extents/3+1][delz+extents/3+1] = ptsinvol


	return [hex_nodes,hex_frames,tot_cube_nodes,tot_cube_frames],mat_matrix,invol

#
# Utility for checking if a given point (or set of points) is 
# contained within a hexagonal volume with a given radius and height
# 
def in_hex_volume(points,abs_hex_radius,abs_hex_height):
	inhexvolume = True
	factor = np.sqrt(3)/2.0
	correction = 0.1
	hexcheck1 = np.transpose(np.dot(init_hex_rot_tform,np.transpose(np.copy(points))))
	hexcheck2 = np.transpose(np.dot(hex_rot_tform,np.transpose(np.copy(hexcheck1))))
	hexcheck3 = np.transpose(np.dot(hex_rot_tform,np.transpose(np.copy(hexcheck2))))

	inhexvolume = False
	pointsinvolume = []
	for i in range(len(points)):
		pointinvolume = 0
		if hexcheck1[i][2] < abs_hex_height+correction and hexcheck1[i][2] > -correction:
			if hexcheck1[i][0] < factor*abs_hex_radius+correction and hexcheck1[i][0] > -factor*abs_hex_radius-correction:
				if hexcheck2[i][0] < factor*abs_hex_radius+correction and hexcheck2[i][0] > -factor*abs_hex_radius-correction:  
					if hexcheck3[i][0] < factor*abs_hex_radius+correction and hexcheck3[i][0] > -factor*abs_hex_radius-correction: 
						inhexvolume = True
						pointinvolume = 1
		pointsinvolume.append(pointinvolume)

	return inhexvolume, pointsinvolume

def crop_framework(nodes,frames,node_frame_map,invol):
	size_x = len(invol)
	size_y = len(invol[0])
	size_z = len(invol[0][0])

	nodemapping = np.zeros(len(nodes))-1
	remap_index = 0
	newnodes = []
	newframes = []
	for i in range(1,size_x-1):
		for j in range(1,size_y-1):
			for k in range(1,size_z-1):
				for n,inval in enumerate(invol[i][j][k]):
					if inval > 0:
						nodemapping[int(node_frame_map[i][j][k][n])] = remap_index
						newnodes.append(nodes[int(node_frame_map[i][j][k][n])])
						remap_index = remap_index+1

	for frame in frames:
		if nodemapping[frame[0]] != -1 and nodemapping[frame[1]] != -1:
				newframes.append([nodemapping[frame[0]],nodemapping[frame[1]]])
					
	nodes = np.array(newnodes)
	frames = np.array(newframes)

	nodemapping = np.zeros(len(nodes))-1
	remap_index = 0
	newnodes = []
	newframes = []
	for i,node in enumerate(nodes):
		if not node_is_unconnected(i,frames):
			nodemapping[i] = remap_index
			newnodes.append(node)
			remap_index = remap_index + 1

	for frame in frames:
		if nodemapping[frame[0]] != -1 and nodemapping[frame[1]] != -1:
				newframes.append([nodemapping[frame[0]],nodemapping[frame[1]]])

	nodes = np.array(newnodes)
	frames = np.array(newframes)



	
	tform = np.linalg.inv(np.array([[1.0/6*(3+np.sqrt(3)),1.0/6*(-3+np.sqrt(3)),1.0/np.sqrt(3)],
				  [1.0/6*(-3+np.sqrt(3)),1.0/6*(3+np.sqrt(3)),1.0/np.sqrt(3)],
				  [-1.0/np.sqrt(3),-1.0/np.sqrt(3),1.0/np.sqrt(3)]]))
	
	# matrix that orients it into the (111) plane
	#need to align to the hexagon
	nodes = (np.dot(tform,nodes.T)).T

	compvec = np.array([np.mean(nodes.T[0]),np.mean(nodes.T[1]),np.min(nodes.T[2])])
	
	nodes = nodes-compvec

	return nodes,frames

def node_is_unconnected(node_id, frames):
	node_alone = True
	if any(frames.T[0] == node_id) or any(frames.T[1] == node_id):
		node_alone = False
	return node_alone

def get_hex_area(hex_radius,vox_pitch):
	return 6.0*0.5*hex_radius*hex_triangle_width*vox_pitch*hex_radius*hex_triangle_width*vox_pitch*np.sqrt(3.0)/2.0

def get_layer_height(vox_pitch):
	return vox_pitch*hex_base_height/3.0

'''
debug = False
if(debug):
	# The cube defined in the cubic coordinate system.
	cube_nodes =  [[0.25,0.25,0.25], #0
				   [0.25,0.75,0.75], #1
				   [0.75,0.25,0.75], #2
				   [0.75,0.75,0.25]] #3

	cube_frames = np.array([[0,1],
						    [0,2],
						    [1,3],
						    [2,3]])
	offset = np.array([0.25,0.25,0.25])

	[hex_nodes, hex_frames, nodes, frames],material_matrix,invol = cubic_to_111(1,1,cube_nodes,cube_frames,offset)

	#print(frames,np.shape(nodes))
	#Set up figure plotting
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	#nodes = np.array(nodes)
	xs = np.array(nodes.T[0])
	ys = np.array(nodes.T[1])
	zs = np.array(nodes.T[2])

	hxs = np.array(hex_nodes.T[0])
	hys = np.array(hex_nodes.T[1])
	hzs = np.array(hex_nodes.T[2])

	#This maintains proper aspect ratio for the 3d plot
	max_range = np.array([hxs.max()-hxs.min(), hys.max()-hys.min(), hzs.max()-hzs.min()]).max()
	Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(hxs.max()+hxs.min())
	Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(hys.max()+hys.min())
	Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(hzs.max()+hzs.min())
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
	   ax.plot([xb], [yb], [zb], 'w')

	#plot all of the frames
	for i,frame in enumerate(frames):
			nid1 = int(frame[0])
			nid2 = int(frame[1])
			start = [xs[nid1],ys[nid1],zs[nid1]]
			end   = [xs[nid2],ys[nid2],zs[nid2]]
			ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)

	#plot all of the frames
	for i,frame in enumerate(hex_frames):
			nid1 = int(frame[0])
			nid2 = int(frame[1])
			start = [hxs[nid1],hys[nid1],hzs[nid1]]
			end   = [hxs[nid2],hys[nid2],hzs[nid2]]
			ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='b', alpha=1.0)


	#plot the nodes
	ax.scatter(xs,ys,zs)
	ax.scatter(hxs,hys,hzs,color='r')

	#show it
	plt.show()
'''