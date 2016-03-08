import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import frame3dd
import subprocess
import pfea
import cProfile

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


#Temporary Material Matrix - NxNxN grid
# at the moment:
# 1's correspond to material being there
# 0's correspond to no material



modvals = []
#Subdiv is the amount of subdivision
#We split a cube into Subdiv^3 subdivisions

for subdiv in range(8,9):
	
	jitvals = []
	
	for jit in [0.0]:#[0.0001,0.000316,0.001,0.00316,0.01,0.0316,0.1]:
		
		tar_den = 0.01

		mat_matrix = [[[0,0,0],
					   [0,0,0],
					   [0,0,0]],
					  [[0,0,0],
					   [0,1,0],
					   [0,0,0]],
					  [[0,0,0],
					   [0,0,0],
					   [0,0,0]]]
		#subdiv = 8
		zheight = subdiv+1
		pitchfactor = 1.05

		#subdiving beams?
		subdiv_beam = False
		num_bsub = 6

		#performing a solid-body rotation on the article?
		rotate = False

		#altering the strut dimensions to match a desired density?
		renormalize = True

		notskipping = True
		twod = True

		#Subdivide the material matrix
		dims = np.shape(mat_matrix)
		new_mat_matrix = np.zeros(((dims[0]-2)*subdiv+2,(dims[1]-2)*subdiv+2,(dims[2]-2)*zheight+2))

		for i in range(1,dims[0]-1):
			for j in range(1,dims[1]-1):
				for k in range(1,dims[2]-1):
					if mat_matrix[i][j][k] == 1:
						dex = ((i-1)*subdiv+1,(j-1)*subdiv+1,(k-1)*zheight+1)
						for l in range(0,subdiv):
							for m in range(0,subdiv):
								for n in range(0,zheight):
									new_mat_matrix[l+dex[0]][m+dex[1]][n+dex[2]] = 1

		#subdiv=8
		#zheight= #units of subdiv
		#print(new_mat_matrix)

		mat_matrix = new_mat_matrix

		# Material Properties
		#for pitchfactor in [0.1,0.5,0.75,1.0,1.5,2.0,3.0]
		#Physical Voxel Properties
		vox_pitch = 0.00854#/subdiv #m


		node_radius = 0

		#STRUT PROPERTIES
		#Physical Properties
		#Assuming a square, Ti6Al4V strut that is 0.5 mm on a side
		#Using Newtons, meters, and kilograms as the units

		frame_props = {"nu"  : 0.35, #poisson's ratio
					   "d1"	 : 0.000762, #m
					   "d2"	 : 0.000762, #m
					   "th"  : 0,
					   "E"   :  11e9, #N/m^2,
					   "rho" :  1650, #kg/m^3
					   "beam_divisions" : 0,
					   "cross_section"  : 'rectangular',
					   "roll": 0}

		#Alternate frame properties to ensure that circular vs. rectangular struts
		#Dont matter (much)
		'''
		frame_props = {"nu"  : 0.33, #poisson's ratio
					   "d1"	 : 0.000564/subdiv, #m
					   "d2"	 : 0.000564/subdiv, #m
					   "th"	 : 0.000564/2.0/subdiv, #m
					   "E"   :  116000000000, #N/m^2
					   "G"   :   42000000000,  #N/m^2
					   "rho" :  4420, #kg/m^3
					   "beam_divisions" : 0,
					   "cross_section"  : 'circular',
					   "roll": 0} #0.56*vox_pitch}#
		'''

		#Node Map Population
		#Referencing the geometry-specific file.

		nodes,frames,node_frame_map,dims = dschwarz.from_material(mat_matrix,vox_pitch)
		frame_props["Le"] = dschwarz.frame_length(vox_pitch)
		#nodes,frames,node_frame_map,dims = pschwarz.from_material(mat_matrix,vox_pitch)
		#frame_props["Le"] = pschwarz.frame_length(vox_pitch)
		#nodes,frames,node_frame_map,dims = cuboct.from_material(mat_matrix,vox_pitch)
		#frame_props["Le"] = cuboct.frame_length(vox_pitch)
		#nodes,frames,node_frame_map,dims = octet.from_material(mat_matrix,vox_pitch)
		#frame_props["Le"] = octet.frame_length(vox_pitch)
		#nodes,frames,node_frame_map,dims = cuboct_buckle.from_material(mat_matrix,vox_pitch)
		#frame_props["Le"] = cuboct_buckle.frame_length(vox_pitch)
		#nodes,frames,node_frame_map,dims = cubic.from_material(mat_matrix,vox_pitch)
		#frame_props["Le"] = cubic.frame_length(vox_pitch)
		#nodes,frames,node_frame_map,dims = kelvin.from_material(mat_matrix,vox_pitch)
		#frame_props["Le"] = kelvin.frame_length(vox_pitch)

		nodes = np.array(nodes)
		
		#rotation of the structure so it's properly oriented
		#Using Rodrigues' rotation formula
		topframes = []

		if rotate:
			k = [1,0,0]
			K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
			angle = np.arctan(np.sqrt(2))
			R = np.identity(3)+np.sin(angle)*K+(1-np.cos(angle))*np.dot(K,K)

			newnodes = np.zeros(np.shape(nodes))
			for i,node in enumerate(nodes):
				newnodes[i,:] = np.dot(R,node)

			#Rezero the z-location of the structure
			meanval = [np.mean(newnodes.T[0]),np.mean(newnodes.T[1]),np.mean(newnodes.T[2])]

			newnodes.T[0] = newnodes.T[0]-meanval[0]
			newnodes.T[1] = newnodes.T[1]-meanval[1]
			newnodes.T[2] = newnodes.T[2]-meanval[2]
			#maxval = np.max(newnodes.T[2])

			tmp = dims[2]
			dims[2] = dims[1]
			dims[1] = tmp

			nodes = newnodes

			#cropping
			newnodes = np.zeros(np.shape(nodes))
			nodemap = np.zeros(len(nodes))
			newdex = 0

			for i,node in enumerate(nodes):
				if np.sqrt(node[0]**2+node[1]**2) >= 0.5*subdiv*vox_pitch or np.abs(node[2]) - zheight/2.0*vox_pitch >= 0.02:
					nodemap[i] = -1
				else:
					newnodes[newdex][0:3]= node
					nodemap[i] = newdex
					newdex = newdex+1
			#print(nodemap,newnodes[0:newdex])
			maxval = np.max(newnodes.T[2])
			newframes = []
			for i, frame in enumerate(frames):
				if not(nodemap[frame[0]] == -1 or nodemap[frame[1]] == -1):
					if np.abs(newnodes[nodemap[frame[0]]][2] - maxval) < vox_pitch and np.abs(newnodes[nodemap[frame[1]]][2] - maxval) < vox_pitch:
						topframes.append([nodemap[frame[0]],nodemap[frame[1]]])
					else:
						newframes.append([nodemap[frame[0]],nodemap[frame[1]]])

			#print(topframes)
			minval = [np.min(newnodes.T[0]),np.min(newnodes.T[1]),np.min(newnodes.T[2])]

			newnodes.T[0] = newnodes.T[0]-minval[0]
			newnodes.T[1] = newnodes.T[1]-minval[1]
			newnodes.T[2] = newnodes.T[2]-minval[2]

			nodes = newnodes[0:newdex]
			frames = newframes
		else:
			maxval = np.max(nodes.T[2])
			newframes = []
			zdim = maxval-vox_pitch
			for i, frame in enumerate(frames):
				if np.abs(nodes[frame[0]][2] - maxval) <= pitchfactor*vox_pitch*dims[2] and np.abs(nodes[frame[1]][2] - maxval) <= pitchfactor*vox_pitch*dims[2]:
					topframes.append([frame[0],frame[1]])
				else:
					newframes.append([frame[0],frame[1]])

		frames = newframes

		maxval = np.max(nodes.T[2])
		minval = np.min(nodes.T[2])
		zdim = maxval-minval-vox_pitch

		#renormalize beam cross section to keep the same relative density
		xdim = np.max(nodes.T[0])-np.min(nodes.T[0])
		ydim = np.max(nodes.T[1])-np.min(nodes.T[1])
		art_vol = xdim*ydim*zdim

		if renormalize:
			#tar_den = 0.01

			frame_props["d1"] = np.sqrt(tar_den*art_vol/(len(frames)*frame_props["Le"]))
			frame_props["d2"] = frame_props["d1"]
			#print(frame_props["d1"]*frame_props["d2"]*frame_props["Le"],rel_den)

		rel_den = len(frames)*frame_props["d1"]*frame_props["d2"]*frame_props["Le"]/art_vol

		print(frame_props["d1"], " ", rel_den)

		minval = np.min(nodes.T[2])
		maxval = np.max(nodes.T[2])


		strain = 0.001 #percent
		strain_disp = (maxval-minval)*strain
		loadval = 50*tar_den #N

		#Constraint and load population
		constraints = []
		loads = []
		#Constraints are added based on simple requirements right now
		strains = np.zeros(6)
		#strains[2] = -strain_disp
		minval = np.min(nodes.T[2])
		maxval = np.max(nodes.T[2])


		
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

		#print(frames)

		#print(len(loads)*loadval)
		#print(len(constraints),len(nodes)*6)

		'''
		for x in range(1,subdiv+1):
			for y in range(1,subdiv+1):
				#The bottom-most nodes are constrained to neither translate nor
				#rotate
				constraints.append({'node':node_frame_map[x][y][1][0],'DOF':0, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][0],'DOF':1, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][0],'DOF':2, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][0],'DOF':3, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][0],'DOF':4, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][0],'DOF':5, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][3],'DOF':0, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][3],'DOF':1, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][3],'DOF':2, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][3],'DOF':3, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][3],'DOF':4, 'value':0})
				constraints.append({'node':node_frame_map[x][y][1][3],'DOF':5, 'value':0})
				#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':3, 'value':0})
				#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':4, 'value':0})
				#constraints.append({'node':node_frame_map[x][y][1][2],'DOF':5, 'value':0})

				#The top most nodes are assigned a z-axis load, as well as being
				#constrained to translate in only the z-direction.

				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':0, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':1, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':2, 'value':-strain_disp})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':3, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':4, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':5, 'value':0})

				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':0, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':1, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':2, 'value':-strain_disp})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':3, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':4, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':5, 'value':0})

				loads.append(      {'node':node_frame_map[x][y][subdiv][1],'DOF':2, 'value':-5.0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':0, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':1, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':3, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':4, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][1],'DOF':5, 'value':0})

				loads.append(      {'node':node_frame_map[x][y][subdiv][2],'DOF':2, 'value':-5.0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':0, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':1, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':3, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':4, 'value':0})
				constraints.append({'node':node_frame_map[x][y][subdiv][2],'DOF':5, 'value':0})

		'''
		#frames = cuboct.remove_frame([(int(size_x/2.0)+1,int(size_y/2.0)+1,int(size_z/2.0)+1),2],node_frame_map,frames)
		#dframes = cuboct.remove_frame([(int(size_x/2.0)+1,int(size_y/2.0)+1,int(size_z/2.0)+1),5],node_frame_map,dframes)

		#print(len(loads))


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
										 #'beam_divisions' : frame_props["beam_divisions"],
										 'cross_section'  : frame_props["cross_section"],
										 'roll': frame_props["roll"],
										 'loads':{'element':0},
										 'prestresses':{'element':0},
										 'Le': frame_props["Le"],
										 'beam_divisions': 2}),
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
										 'beam_divisions': 1})]


		#Format node positions
		out_nodes = np.array(nodes)

		#Global Arguments
		global_args = {'frame3dd_filename': "test",'length_scaling':1,"using_Frame3dd":False,"debug_plot":True,"gravity" : [0,0,0]}
		global_args["node_radius"] = np.zeros(len(nodes))+0.05*frame_props["Le"]
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
			jitvals.append(len(loads)*loadval/(xdim*ydim*np.mean(top_disp)/zdim))
			#print(subdiv,len(loads)*loadval/(xdim*ydim*np.mean(top_disp)/zdim),tar_den)
			#print("{0} of 10".format(jit+1))
	jitvals = np.array(jitvals)
	if notskipping:
		print([subdiv,tar_den,np.mean(jitvals),np.mean(top_disp)])
		modvals.append([subdiv,tar_den,np.mean(jitvals),np.std(jitvals)])

print(modvals)

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
