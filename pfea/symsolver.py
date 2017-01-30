#Various attempts at GPU Optimization

#import pycuda.gpuarray as gpuarray
#import pycuda.autoinit
#import skcuda.linalg as sklinalg

import numpy as np
from util import pfeautil
from math import *
import scipy as sp
import sympy 

#    #       #     #    #    ####### ######  ### #     # 
#   #        ##   ##   # #      #    #     #  #   #   #  
#  #         # # # #  #   #     #    #     #  #    # #   
###          #  #  # #     #    #    ######   #     #    
#  #         #     # #######    #    #   #    #    # #   
#   #        #     # #     #    #    #    #   #   #   #  
#    #       #     # #     #    #    #     # ### #     # 

# Functions related to the construction of the stiffness matrix K
                                                        
def assemble_K(nodes,beam_sets,Q,args):
	'''
	# Nodes is all the nodes
	# Q is the internal pre-load forces
	# Beam_sets is a list of tuples (beams, properties)
	# 	beams is an n-by-2 array consisting of node indices
	# 	properties is a dictionary listing the corresponding
	#	parameters for the beams in the tuple
	# Args is a dictionary consisting of the simulation parameters
	# Key:
	#	Shear 	: are we considering shear here
	#	dof 	: the numbers of degrees of freedom
	#	nE 		: the number of beam elements
	'''
	#Initialize K to zeros
	#K = np.zeros((args["dof"],args["dof"]))
	tot_dof = args["dof"]
	K = co.spmatrix([],[],[],(tot_dof,tot_dof))
	data = np.zeros((tot_dof*6*6*9,))
	row = np.zeros((tot_dof*6*6*9,), dtype=np.int)
	col = np.zeros((tot_dof*6*6*9,), dtype=np.int)
	data2 = np.array([])
	q_index = 0
	dat_dex = 0
	size = 18
	for beamset,bargs in beam_sets:
		#every beam set lists the physical properties
		#associated with that beam
		
		#transfer those properties over
		beam_props = {"Ax"		:bargs["Ax"],
					  "Asy"		: bargs["Asy"],
					  "Asz"		: bargs["Asz"],
					  "G"		: bargs["G"],
					  "E"		: bargs["E"],
					  "J"		: bargs["J"],
					  "Iy"		: bargs["Iy"],
					  "Iz"		: bargs["Iz"],
					  "p"		: bargs["roll"],
					  "Le"		: bargs["Le"],
					  "shear"	: bargs["shear"]}
		for beam in beamset:
			#Positions of the endpoint nodes for this beam
			xn1 = nodes[beam[0]]
			xn2 = nodes[beam[1]]
			beam_props["xn1"] = xn1
			beam_props["xn2"] = xn2
			
			#beam_props["Le"]  = sqrt((xn2[0]-xn1[0])**2+(xn2[1]-xn1[1])**2+(xn2[2]-xn1[2])**2) 
			#Things that are not faster than the above:
			#   sp.spatial.distance.euclidean(xn2,xn1)
			
			beam_props["T"]	  = -Q[q_index,0]
			q_index = q_index+1

			ke = elastic_K(beam_props)
			kg = geometric_K(beam_props)
			bi0 = 6*int(beam[0])
			bi1 = 6*int(beam[1])
			offsets = [[bi0,bi0],[bi1,bi0],[bi0,bi1],[bi1,bi1]]

			'''
			locs = [[0,10],[10,20],[10,20],[20,30]]

			tdata, trow, tcol = pop_k(beam_props)
			order = [[trow,tcol],[trow,tcol],[tcol,trow],[trow,tcol]]

			for i in range(4):
				data[size*dat_dex:size*(dat_dex+1)] = tdata[locs[i][0]:locs[i][1]]
				row[size*dat_dex:size*(dat_dex+1)]  = order[i][0]+offsets[i][0]
				col[size*dat_dex:size*(dat_dex+1)]  = order[i][1]+offsets[i][1]
				dat_dex=dat_dex+1
			'''
			for i in range(4):
				ktmp = sp.sparse.coo_matrix(ke[i]+kg[i])
				trow = ktmp.row+offsets[i][0]
				tcol = ktmp.col+offsets[i][1]
				lendat = len(ktmp.data)
				#data2 = np.append(data2,ktmp.data)
				#row = np.append(row,trow)#ktmp.row+offsets[i][0])
				#col = np.append(col,tcol)#ktmp.col+offsets[i][1])
				data[dat_dex:dat_dex+lendat] = ktmp.data[:]
				row[dat_dex:dat_dex+lendat] = trow[:]
				col[dat_dex:dat_dex+lendat] = tcol[:]
				dat_dex+=lendat
	
	#print(data[dat_dex*size:(dat_dex+1)*size])
	data = data[:dat_dex]
	row = row[:dat_dex]
	col = col[:dat_dex]
	K = co.spmatrix(data,col,row,(tot_dof,tot_dof))
	return K
	
#ELASTIC_K - space frame elastic stiffness matrix in global coordinates

def elastic_K(beam_props):
	# beam_props is a dictionary with the following values
	# xn1   : position vector for start node
	# xn2	: position vector for end node
	# Le    : Effective beam length (taking into account node diameter)
	# Asy   : Effective area for shear effects, y direction
	# Asz   : Effective area for shear effects, z direction
	# G		: Shear modulus
	# E 	: Elastic modulus
	# J 	: Polar moment of inertia
	# Iy 	: Bending moment of inertia, y direction
	# Iz 	: bending moment of inertia, z direction
	# p 	: The roll angle (radians)
	# T 	: internal element end force
	# shear : Do we consider shear effects

	#Start by importing the beam properties
	xn1 	= beam_props["xn1"]
	xn2 	= beam_props["xn2"]
	Le  	= beam_props["Le"]
	Ax		= beam_props["Ax"]
	Asy 	= beam_props["Asy"]
	Asz 	= beam_props["Asz"]
	G   	= beam_props["G"]
	E   	= beam_props["E"]
	J 		= beam_props["J"]
	Iy 		= beam_props["Iy"]
	Iz 		= beam_props["Iz"]
	p 		= beam_props["p"]
	shear 	= beam_props["shear"]

	#initialize the output
	k = np.zeros((12,12))
	#k = co.matrix(0.0,(12,12))
	#define the transform between local and global coordinate frames
	t = pfeautil.coord_trans(xn1,xn2,Le,p)

	#calculate Shear deformation effects
	Ksy = 0
	Ksz = 0

	#begin populating that elastic stiffness matrix
	if shear:
		Ksy = 12.0*E*Iz / (G*Asy*Le*Le)
		Ksz = 12.0*E*Iy / (G*Asz*Le*Le)
	else:
		Ksy = Ksz = 0.0
	
	k[0,0]  = k[6,6]   = 1.0*E*Ax / Le
	k[1,1]  = k[7,7]   = 12.*E*Iz / ( Le*Le*Le*(1.+Ksy) )
	k[2,2]  = k[8,8]   = 12.*E*Iy / ( Le*Le*Le*(1.+Ksz) )
	k[3,3]  = k[9,9]   = 1.0*G*J / Le
	k[4,4]  = k[10,10] = (4.+Ksz)*E*Iy / ( Le*(1.+Ksz) )
	k[5,5]  = k[11,11] = (4.+Ksy)*E*Iz / ( Le*(1.+Ksy) )

	k[4,2]  = k[2,4]   = -6.*E*Iy / ( Le*Le*(1.+Ksz) )
	k[5,1]  = k[1,5]   =  6.*E*Iz / ( Le*Le*(1.+Ksy) )
	k[6,0]  = k[0,6]   = -k[0,0]

	k[11,7] = k[7,11]  =  k[7,5] = k[5,7] = -k[5,1]
	k[10,8] = k[8,10]  =  k[8,4] = k[4,8] = -k[4,2]
	k[9,3]  = k[3,9]   = -k[3,3]
	k[10,2] = k[2,10]  =  k[4,2]
	k[11,1] = k[1,11]  =  k[5,1]

	k[7,1]  = k[1,7]   = -k[1,1]
	k[8,2]  = k[2,8]   = -k[2,2]
	k[10,4] = k[4,10]  = (2.-Ksz)*E*Iy / ( Le*(1.+Ksz) )
	k[11,5] = k[5,11]  = (2.-Ksy)*E*Iz / ( Le*(1.+Ksy) )


	#now we transform k to the global coordinates
	k = pfeautil.atma(t,k)

	# Check and enforce symmetry of the elastic stiffness matrix for the element
	k = 0.5*(k+k.T)
	'''
	for i in range(12):
		for j in range(i+1,12):
			if(k[i][j]!=k[j][i]):
				if(abs(1.0*k[i][j]/k[j][i]-1.0) > 1.0e-6 and (abs(1.0*k[i][j]/k[i][i]) > 1e-6 or abs(1.0*k[j][i]/k[i][i]) > 1e-6)):
					print("Ke Not Symmetric")
				k[i][j] = k[j][i] = 0.5 * ( k[i][j] + k[j][i] )
	'''
	return [k[:6,:6],k[6:,:6],k[:6,6:],k[6:,6:]]

#GEOMETRIC_K - space frame geometric stiffness matrix, global coordnates

def geometric_K(beam_props):
	# beam_props is a dictionary with the following values
	# xn1   : position vector for start node
	# xn2	: position vector for end node
	# Le    : Effective beam length (taking into account node diameter)
	# Asy   : Effective area for shear effects, y direction
	# Asz   : Effective area for shear effects, z direction
	# G		: Shear modulus
	# E 	: Elastic modulus
	# J 	: Polar moment of inertia
	# Iy 	: Bending moment of inertia, y direction
	# Iz 	: bending moment of inertia, z direction
	# p 	: The roll angle (radians)
	# T 	: internal element end force
	# shear : whether shear effects are considered. 
	
	xn1 	= beam_props["xn1"]
	xn2 	= beam_props["xn2"]
	L   	= beam_props["Le"]
	Le  	= beam_props["Le"]
	Ax		= beam_props["Ax"]
	Asy 	= beam_props["Asy"]
	Asz 	= beam_props["Asz"]
	G   	= beam_props["G"]
	E   	= beam_props["E"]
	J 		= beam_props["J"]
	Iy 		= beam_props["Iy"]
	Iz 		= beam_props["Iz"]
	p 		= beam_props["p"]
	T 		= beam_props["T"]
	shear 	= beam_props["shear"]

	#initialize the geometric stiffness matrix
	kg = np.zeros((12,12))
	t = pfeautil.coord_trans(xn1,xn2,Le,p)

	if shear:
		Ksy = 12.0*E*Iz / (G*Asy*Le*Le);
		Ksz = 12.0*E*Iy / (G*Asz*Le*Le);
		Dsy = (1+Ksy)*(1+Ksy);
		Dsz = (1+Ksz)*(1+Ksz);
	else:
		Ksy = Ksz = 0.0;
		Dsy = Dsz = 1.0;

	#print(T)
	kg[0][0]  = kg[6][6]   =  0.0 # T/L
	 
	kg[1][1]  = kg[7][7]   =  T/L*(1.2+2.0*Ksy+Ksy*Ksy)/Dsy
	kg[2][2]  = kg[8][8]   =  T/L*(1.2+2.0*Ksz+Ksz*Ksz)/Dsz
	kg[3][3]  = kg[9][9]   =  T/L*J/Ax
	kg[4][4]  = kg[10][10] =  T*L*(2.0/15.0+Ksz/6.0+Ksz*Ksz/12.0)/Dsz
	kg[5][5]  = kg[11][11] =  T*L*(2.0/15.0+Ksy/6.0+Ksy*Ksy/12.0)/Dsy
	 
	kg[0][6]  = kg[6][0]   =  0.0 # -T/L
	
	kg[4][2]  = kg[2][4]   =  kg[10][2] = kg[2][10] = -T/10.0/Dsz
	kg[8][4]  = kg[4][8]   =  kg[10][8] = kg[8][10] =  T/10.0/Dsz
	kg[5][1]  = kg[1][5]   =  kg[11][1] = kg[1][11] =  T/10.0/Dsy
	kg[7][5]  = kg[5][7]   =  kg[11][7] = kg[7][11] = -T/10.0/Dsy
	
	kg[3][9]  = kg[9][3]   = -kg[3][3]
	
	kg[7][1]  = kg[1][7]   = -T/L*(1.2+2.0*Ksy+Ksy*Ksy)/Dsy
	kg[8][2]  = kg[2][8]   = -T/L*(1.2+2.0*Ksz+Ksz*Ksz)/Dsz

	kg[10][4] = kg[4][10]  = -T*L*(1.0/30.0+Ksz/6.0+Ksz*Ksz/12.0)/Dsz
	kg[11][5] = kg[5][11]  = -T*L*(1.0/30.0+Ksy/6.0+Ksy*Ksy/12.0)/Dsy

	#now we transform kg to the global coordinates
	kg = pfeautil.atma(t,kg)

	# Check and enforce symmetry of the elastic stiffness matrix for the element
	kg = 0.5*(kg+kg.T)
	
	return [kg[:6,:6],kg[6:,:6],kg[:6,6:],kg[6:,6:]]

def pop_k(beam_props):
	# beam_props is a dictionary with the following values
	# xn1   : position vector for start node
	# xn2	: position vector for end node
	# Le    : Effective beam length (taking into account node diameter)
	# Asy   : Effective area for shear effects, y direction
	# Asz   : Effective area for shear effects, z direction
	# G		: Shear modulus
	# E 	: Elastic modulus
	# J 	: Polar moment of inertia
	# Iy 	: Bending moment of inertia, y direction
	# Iz 	: bending moment of inertia, z direction
	# p 	: The roll angle (radians)
	# T 	: internal element end force
	# shear : whether shear effects are considered. 
	
	xn1 	= beam_props["xn1"]
	xn2 	= beam_props["xn2"]
	L   	= beam_props["Le"]
	Le  	= beam_props["Le"]
	Ax		= beam_props["Ax"]
	Asy 	= beam_props["Asy"]
	Asz 	= beam_props["Asz"]
	G   	= beam_props["G"]
	E   	= beam_props["E"]
	J 		= beam_props["J"]
	Iy 		= beam_props["Iy"]
	Iz 		= beam_props["Iz"]
	p 		= beam_props["p"]
	T 		= beam_props["T"]
	shear 	= beam_props["shear"]

	t = pfeautil.coord_trans(xn1,xn2,Le,p)

	if shear:
		Ksy = 12.0*E*Iz / (G*Asy*Le*Le)
		Ksz = 12.0*E*Iy / (G*Asz*Le*Le)
		Dsy = (1+Ksy)*(1+Ksy)
		Dsz = (1+Ksz)*(1+Ksz)
	else:
		Ksy = Ksz = 0.0
		Dsy = Dsz = 1.0

	data = np.zeros(30)
	rows = np.array([0,1,2,3,4,5,5,4,2,1])
	cols = np.array([0,1,2,3,4,5,1,2,4,5])
	
	

	return data,rows,cols

def provide_K(nodes, global_args, beam_sets):
	
	nE = sum(map(lambda x: np.shape(x[0])[0], beam_sets))    
	Q = co.matrix(0.0,(nE,12))
	
	try: 
		length_scaling = global_args['length_scaling']
	except(KeyError): 
		length_scaling = 1.
	
	for beamset, args in beam_sets:
		E = 1.0*args['E']/length_scaling/length_scaling
		nu = args['nu']
		d1 = args['d1']*length_scaling
		roll = args['roll']
		if args['cross_section']=='circular':
			Ro = .5*d1
			args['th'] = args['th']*length_scaling
			assert(0<args['th']<=Ro)
			Ri  = Ro-args['th']
			Ax  = pi*(Ro**2-Ri**2)
			Asy = Ax/(0.54414 + 2.97294*(Ri/Ro) - 1.51899*(Ri/Ro)**2 )
			Asz = Asy
			Jxx = .5*pi*(Ro**4-Ri**4)
			Iyy = .25*pi*(Ro**4-Ri**4)
			Izz = Iyy	
		elif args['cross_section']=='rectangular':
			d2 = args['d2']*length_scaling
			Ax = d1*d2
			Asy = Ax*(5+5*nu)/(6+5*nu)
			Asz = Asy
			Iyy = d1**3*d2/12.
			Izz = d1*d2**3/12.
			tmp = .33333 - 0.2244/(max(d1,d2)/min(d1,d2) + 0.1607); 
			Jxx = tmp*max(d1,d2)*min(d1,d2)**3 
		args['E']   = E
		args['d1']  = d1
		args['Ax']  = Ax 
		args['Asy'] = Asy
		args['Asz'] = Asz
		args['J']   = Jxx
		args['Iy']  = Iyy
		args['Iz']  = Izz		
		args['G']   = E/2./(1+nu)
		args['rho'] = args['rho']/length_scaling/length_scaling/length_scaling
		args['Le']  = args['Le']*length_scaling
	
	tot_dof = len(nodes)*6
	
	global_args["dof"] = tot_dof

	return assemble_K(nodes,beam_sets,Q,global_args)




