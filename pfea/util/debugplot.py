# Imports
import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#
# Making sure all three axes are equal scale
#
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

#
# Debugplot
#
def debugplot(nodes,framesets,res_displace,Q,loads,scale_factor):
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

	#print(matplotlib.projections.get_projection_names())
	for i,node in enumerate(nodes):
		xs.append(node[0])
		ys.append(node[1])
		zs.append(node[2])
		rxs.append(node[0]+res_displace[i][0]*scale_factor)
		rys.append(node[1]+res_displace[i][1]*scale_factor)
		rzs.append(node[2]+res_displace[i][2]*scale_factor)

	frame_args = framesets[0][2]
	st_nrg = Q.T[0,:]#0.5*frame_args["Le"]/frame_args["E"]*(Q.T[0,:]**2/frame_args["Ax"]+Q.T[4,:]**2/frame_args["Iy"]+Q.T[5,:]**2/frame_args["Iz"])
	st_nrg = st_nrg.T
	qmax = np.max(abs(st_nrg))
	#print st_nrg
	print qmax
	print np.mean(abs(st_nrg))

	factors = np.zeros((len(st_nrg),2))

	for i, strain in enumerate(st_nrg):
		if strain > 0:
			if abs(strain)/(pi**3*frame_args["E"]*frame_args["d1"]**4/(4*frame_args["Le"]**2)) > 1.0:
				print("**DANGER AT FRAME ID {0}".format(i))
			#print "element {0} safety factor".format(abs(strain)/(pi**3*frame_args["E"]*frame_args["d1"]**4/(4*frame_args["Le"]**2)))
			factors[i][0] = abs(strain)/(pi**3*frame_args["E"]*frame_args["d1"]**4/(4*frame_args["Le"]**2))
		if strain < 0:
			if abs(strain)/(37.5e6*frame_args["d1"]**2) > 1.0:
				print("**DANGER AT FRAME ID {0}".format(i))
			#print "element {0} safety factor".format(abs(strain)/(37.5e6*frame_args["d1"]**2))
			factors[i][1] = abs(strain)/(37.5e6*frame_args["d1"]**2)

	max_ten = np.max(factors.T[1])
	max_com = np.max(factors.T[0])
	for i,frame in enumerate(framesets[0][0]):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
		rend   = [rxs[nid2],rys[nid2],rzs[nid2]]

		#ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.5)
		if(st_nrg[i] < 0 ):
			ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='r', alpha=(1.0*factors[i][1]/max_ten)**3)
		else:
			ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=(1.0*factors[i][0]/max_com)**3)

	'''
	for dframe in dframes:
		nid1 = int(dframe[0])
		nid2 = int(dframe[1])
		dstart = [dxs[nid1],rys[nid1],rzs[nid1]]
		dend   = [dxs[nid2],rys[nid2],rzs[nid2]]
		ax.plot([dstart[0],dend[0]],[dstart[1],dend[1]],[dstart[2],dend[2]],color='b', alpha=0.1)
	'''

	#plot loads
	loadcoords = []
	for load in loads:
		loadcoords.append(nodes[load["node"]]+res_displace[load["node"]]*scale_factor)
	loadcoords = np.array(loadcoords)
	ax.scatter(loadcoords.T[0],loadcoords.T[1],loadcoords.T[2], color='g')

	axisEqual3D(ax)
	ax.scatter(xs,ys,zs, color='g',alpha=0)
	#ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
	plt.show()
	#print(frames)
