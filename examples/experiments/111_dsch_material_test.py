import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import frame3dd
import subprocess
import pfea
import cProfile
import latticegen

#Geometry Imports
import dschwarz
import pschwarz
import cuboct
import cuboct_buckle
import cubic
import kelvin
import octet

from math import *


######  #######  #####   #####  ######  ### ######  ####### ### ####### #     #
#     # #       #     # #     # #     #  #  #     #    #     #  #     # ##    #
#     # #       #       #       #     #  #  #     #    #     #  #     # # #   #
#     # #####    #####  #       ######   #  ######     #     #  #     # #  #  #
#     # #             # #       #   #    #  #          #     #  #     # #   # #
#     # #       #     # #     # #    #   #  #          #     #  #     # #    ##
######  #######  #####   #####  #     # ### #          #    ### ####### #     #

# Performs a standard modulus sweep for a lattice.
# We should see a monotonically increasing modulus, as we
# increased the resolution of the lattice


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

output =[] 
for param in range(2,6):
	vox_pitch = 0.02 #m
	tar_den = 0.1

	hex_radius = param
	hex_height = param

	hex_area = latticegen.get_hex_area(hex_radius,vox_pitch)
	hex_layer_height = latticegen.get_layer_height(vox_pitch)

	#subdiving beams?
	subdiv_beam = True
	num_bsub = 8

	#altering the strut dimensions to match a desired density?
	renormalize = True

	notskipping = True
	twod = True

			
		

	#STRUT PROPERTIES
	#Physical Properties
	#Using Newtons, meters, and kilograms as the units

	frame_props = {"nu"  : 0.19, #poisson's ratio
				   "d1"	 : 0.002, #m
				   "d2"	 : 0.002, #m
				   "th"  : 0,
				   "E"   :  460e9, #N/m^2,
				   "rho" :  2520, #kg/m^3
				   "beam_divisions" : 0,
				   "cross_section"  : 'rectangular',
				   "roll": 0}

	#Node Map Population
	#Referencing the geometry-specific file.

	#nodes,frames= dschwarz.gen_111(hex_radius,hex_height,vox_pitch)
	#frame_props["Le"] = dschwarz.frame_length(vox_pitch)
	nodes,frames= pschwarz.gen_111(hex_radius,hex_height,vox_pitch)
	frame_props["Le"] = pschwarz.frame_length(vox_pitch)
	#nodes,frames= octet.gen_111(hex_radius,hex_height,vox_pitch)
	#frame_props["Le"] = octet.frame_length(vox_pitch)
	#nodes,frames= kelvin.gen_111(hex_radius,hex_height,vox_pitch)
	#frame_props["Le"] = kelvin.frame_length(vox_pitch)


	nodes = np.array(nodes)

	topframes = []

	#Denote the topmost 
	maxval = np.max(nodes.T[2])
	minval = np.min(nodes.T[2])

	print("Maxval: {0} | Minval: {1}".format(maxval,minval))

	newframes = []
	pitchfactor = 1.05
	zdim = maxval-minval-hex_layer_height
	for i, frame in enumerate(frames):
		if np.abs(nodes[frame[0]][2] - maxval) <= pitchfactor*1.0*hex_layer_height and np.abs(nodes[frame[1]][2] - maxval) <= pitchfactor*1.0*hex_layer_height:
			topframes.append([frame[0],frame[1]])
		else:
			newframes.append([frame[0],frame[1]])


	frames = newframes

	art_vol = hex_area*zdim

	if renormalize:
		#tar_den = 0.01

		frame_props["d1"] = np.sqrt(tar_den*art_vol/(len(frames)*frame_props["Le"]))
		frame_props["d2"] = frame_props["d1"]
		#print(frame_props["d1"]*frame_props["d2"]*frame_props["Le"],rel_den)

	rel_den = len(frames)*frame_props["d1"]*frame_props["d2"]*frame_props["Le"]/art_vol

	print(frame_props["d1"], " ", rel_den)



	loadval = 10*tar_den #N

	#Constraint and load population
	constraints = []
	loads = []
	#Constraints are added based on simple requirements right now
	strains = np.zeros(6)

	for i,node in enumerate(nodes):
		#print(node)
		if np.abs(node[2]-maxval)< 0.005:
			for dof in range(6):
				if dof != 2:
					constraints.append({'node':i,'DOF':dof, 'value':strains[dof]})
				if dof == 2:
					#constraints.append({'node':i,'DOF':dof, 'value':-strain_disp})
					loads.append(      {'node':i,'DOF':dof, 'value':-loadval})
		elif np.abs(node[2]-minval) < 0.005:# pitchfactor*vox_pitch:#0.005:
			for dof in range(6):
				constraints.append({'node':i,'DOF':dof, 'value':0})
		else:
			if np.abs(node[2]-maxval)-vox_pitch > 0.005:
				jitter = np.random.rand(3)-0.5
				jitter = jitter/np.linalg.norm(jitter)*frame_props["d1"]*0.0
				#jitter[2] = 0
				#nodes[i][:] = node+jitter

	if subdiv_beam:
		frame_props["Le"] = frame_props["Le"]*1.0/(num_bsub+1)
		newframes = list(frames)
		tdex = len(nodes)
		for i,frame in enumerate(frames):
			st_nid = frame[0]
			en_nid = frame[1]
			stnode = nodes[st_nid]
			nodevec = nodes[en_nid]-nodes[st_nid]
			#print(nodevec)
			#add nodes in the middle
			idvals = [frame[0]]
			for sb in range(num_bsub):
				nodes =  np.vstack((nodes,[nodes[st_nid]+1.0*(sb+1.0)/(num_bsub+1.0)*nodevec]))
				idvals.append(tdex)
				tdex = tdex+1
			idvals.append(frame[1])
			newframes[i][:] = [idvals[0],idvals[1]]
			for bs in range(1,num_bsub+1):
				newframes.append([idvals[bs],idvals[bs+1]])

		frames = newframes



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
									 'cross_section'  : frame_props["cross_section"],
									 'roll': frame_props["roll"],
									 'loads':{'element':0},
									 'prestresses':{'element':0},
									 'Le': frame_props["Le"],
									 'beam_divisions': 1,
									 'shear':True}),
			   (np.array(topframes),{'E'   : frame_props["E"]*1000,
									 'rho' : frame_props["rho"],
									 'nu'  : frame_props["nu"],
									 'd1'  : frame_props["d1"]*4,
									 'd2'  : frame_props["d2"]*4,
									 'th'  : frame_props["th"],
									 #'beam_divisions' : frame_props["beam_divisions"],
									 'cross_section'  : frame_props["cross_section"],
									 'roll': frame_props["roll"],
									 'loads':{'element':0},
									 'prestresses':{'element':0},
									 'Le': frame_props["Le"],
									 'beam_divisions': 1,
									 'shear':True})]


	#Format node positions
	out_nodes = np.array(nodes)

	#Global Arguments
	global_args = {'frame3dd_filename': "test",'length_scaling':1,"using_Frame3dd":False,"debug_plot":False,"gravity" : [0,0,0]}

	#global_args["node_radius"] = np.zeros(len(nodes))+0.05*frame_props["Le"]

	if global_args["using_Frame3dd"]:
		frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
		subprocess.call("frame3dd {0}.csv {0}.out".format(global_args["frame3dd_filename"]), shell=True)
		res_nodes, C = frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
		res_displace = frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])
	else:
		if notskipping:
			res_displace,C,Q = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)
		#cProfile.run('res_displace,C,Q = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)')
		#def_displace = pfea.analyze_System(out_nodes, global_args, def_frames, constraints,loads)
		#else:
			#print("skipping")

	tot_force = 0
	tot_disp = 0
	num_forced = 0

	xdim = np.max(nodes.T[0])-np.min(nodes.T[0])
	ydim = np.max(nodes.T[1])-np.min(nodes.T[1])
	#zdim = np.max(nodes.T[2])-np.min(nodes.T[2])

	#print(xdim,ydim,zdim,zdim/xdim)

	top_disp = []
	jitvals = []
	if notskipping:
		'''
		for constraint in constraints:
			if constraint["value"] != 0:
				tot_force = tot_force + C[constraint["node"]*6+constraint["DOF"]]
				num_forced+=1
		'''
		for load in loads:
			top_disp.append(res_displace[load["node"],load["DOF"]])
		#struct_den = len(frames)*(0.5*frame_props["d1"])**2*np.pi*frame_props["Le"]/(vox_pitch**3*subdiv**3)
		#print("Min/Max Z-displacement: {0} and {1}".format(np.max(top_disp),np.min(top_disp)))
		#print(subdiv,tot_force/(xdim*ydim*strain_disp/zdim),tar_den)
		print("Number of loaded nodes")
		print(len(loads))
		jitvals.append(len(loads)*loadval/(hex_area*np.mean(top_disp)/zdim))
		#print(subdiv,len(loads)*loadval/(xdim*ydim*np.mean(top_disp)/zdim),tar_den)
		#print("{0} of 10".format(jit+1))

	jitvals = np.array(jitvals)

	if notskipping:
		print([param,rel_den,num_bsub,np.mean(jitvals)])
		output.append([param,rel_den,num_bsub,np.mean(jitvals)])

print(output)

if global_args["debug_plot"]:
	### Right now the debug plot only does x-y-z displacements, no twisting
	xs = []
	ys = []
	zs = []

	rxs = []
	rys = []
	rzs = []

	fig = plt.figure()
	if twod:	
		ax = fig.add_subplot(111)
	else:
		ax = fig.add_subplot(111, projection='3d')
	ax.set_aspect('equal')
	frame_coords = []
	factor = 10

	for i,node in enumerate(nodes):
		xs.append(node[0])
		ys.append(node[1])
		zs.append(node[2])
		if notskipping:
			rxs.append(node[0]+res_displace[i][0]*factor)
			rys.append(node[1]+res_displace[i][1]*factor)
			rzs.append(node[2]+res_displace[i][2]*factor)

	frame_args = out_frames[0][1]

	#if notskipping:
		#st_nrg = 0.5*frame_args["Le"]/frame_args["E"]*(Q[:,0]**2/frame_args["Ax"]+Q[:,4]**2/frame_args["Iy"]+Q[:,5]**2/frame_args["Iz"])
		#print(Q[:,0],st_nrg)
		#qmax = np.max(st_nrg)
	#print(qmax)

	for i,frame in enumerate(frames):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		if notskipping:
			rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
			rend   = [rxs[nid2],rys[nid2],rzs[nid2]]
			if(twod):
				ax.plot([rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=(0.25))#*st_nrg[i]/qmax)**2)
			else:
				ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=(1.0))#*st_nrg[i]/qmax)**2)

		#ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)

	for i,frame in enumerate(topframes):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		if notskipping:
			rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
			rend   = [rxs[nid2],rys[nid2],rzs[nid2]]
			if twod:
				ax.plot([rstart[1],rend[1]],[rstart[2],rend[2]],color='k')#*st_nrg[i]/qmax)**2)
			else:
				ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='k', alpha=1.0)#(1.0*st_nrg[i]/qmax)**2)
	'''
	xs = np.array(xs)
	ys = np.array(ys)
	zs = np.array(zs)
	max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
	Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xs.max()+xs.min())
	Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ys.max()+ys.min())
	Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zs.max()+zs.min())
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
	   ax.plot([xb], [yb], [zb], 'w')
	'''
	lxs = []
	lys = []
	lzs = []
	cxs = []
	cys = []
	czs = []

	for load in loads:
		lid = int(load["node"])
		lxs.append(nodes[lid][0])
		lys.append(nodes[lid][1])
		lzs.append(nodes[lid][2])

	for constraint in constraints:
		cid = int(constraint["node"])
		cxs.append(nodes[cid][0])
		cys.append(nodes[cid][1])
		czs.append(nodes[cid][2])
	if twod:
		ax.scatter(ys,zs, color='r',alpha=0.1)
	else:
		ax.scatter(xs,ys,zs, color='r',alpha=0.1)
	#ax.scatter(lxs,lys,lzs,color='m',alpha=1.0,s=40)
	if not twod:
		ax.scatter(cxs,cys,czs,color='k',alpha=1.0,s=40)
	#ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
	plt.show()
	#print(frames)
