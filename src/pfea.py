#Various attempts at GPU Optimization

#import pycuda.gpuarray as gpuarray
#import pycuda.autoinit
#import skcuda.linalg as sklinalg




import numpy as np
from pfeautil import *
from math import *
import scipy as sp

#The sparse solver
from cvxopt import matrix,sparse,cholmod



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
	K = np.zeros((args["dof"],args["dof"]))

	q_index = 0
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
					  "shear"	: True}

		for beam in beamset:
			#Positions of the endpoint nodes for this beam
			xn1 = nodes[beam[0]]
			xn2 = nodes[beam[1]]
			beam_props["xn1"] = xn1
			beam_props["xn2"] = xn2
			
			#beam_props["Le"]  = sqrt((xn2[0]-xn1[0])**2+(xn2[1]-xn1[1])**2+(xn2[2]-xn1[2])**2) 
			#Things that are not faster than the above:
			#   sp.spatial.distance.euclidean(xn2,xn1)
			
			beam_props["T"]	  = -Q[q_index][0]
			q_index = q_index+1
			ke = elastic_K(beam_props)
			kg = geometric_K(beam_props)

			ktot = ke+kg

			K[6*beam[0]:6*beam[0]+6,6*beam[0]:6*beam[0]+6] += ktot[0:6,0:6]
			K[6*beam[1]:6*beam[1]+6,6*beam[0]:6*beam[0]+6] += ktot[6:12,0:6]
			K[6*beam[0]:6*beam[0]+6,6*beam[1]:6*beam[1]+6] += ktot[0:6,6:12]
			K[6*beam[1]:6*beam[1]+6,6*beam[1]:6*beam[1]+6] += ktot[6:12,6:12]

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

	#define the transform between local and global coordinate frames
	t = coord_trans(xn1,xn2,Le,p)

	#calculate Shear deformation effects
	Ksy = 0
	Ksz = 0

	#begin populating that elastic stiffness matrix
	if shear:
		Ksy = 12.0*E*Iz / (G*Asy*Le*Le)
		Ksz = 12.0*E*Iy / (G*Asz*Le*Le)
	else:
		Ksy = Ksz = 0.0
	
	k[0][0]  = k[6][6]   = 1.0*E*Ax / Le
	k[1][1]  = k[7][7]   = 12.*E*Iz / ( Le*Le*Le*(1.+Ksy) )
	k[2][2]  = k[8][8]   = 12.*E*Iy / ( Le*Le*Le*(1.+Ksz) )
	k[3][3]  = k[9][9]   = 1.0*G*J / Le
	k[4][4]  = k[10][10] = (4.+Ksz)*E*Iy / ( Le*(1.+Ksz) )
	k[5][5]  = k[11][11] = (4.+Ksy)*E*Iz / ( Le*(1.+Ksy) )

	k[4][2]  = k[2][4]   = -6.*E*Iy / ( Le*Le*(1.+Ksz) )
	k[5][1]  = k[1][5]   =  6.*E*Iz / ( Le*Le*(1.+Ksy) )
	k[6][0]  = k[0][6]   = -k[0][0]

	k[11][7] = k[7][11]  =  k[7][5] = k[5][7] = -k[5][1]
	k[10][8] = k[8][10]  =  k[8][4] = k[4][8] = -k[4][2]
	k[9][3]  = k[3][9]   = -k[3][3]
	k[10][2] = k[2][10]   =  k[4][2]
	k[11][1] = k[1][11]   =  k[5][1]

	k[7][1]  = k[1][7]   = -k[1][1]
	k[8][2]  = k[2][8]   = -k[2][2]
	k[10][4] = k[4][10]  = (2.-Ksz)*E*Iy / ( Le*(1.+Ksz) )
	k[11][5] = k[5][11]  = (2.-Ksy)*E*Iz / ( Le*(1.+Ksy) )


	#now we transform k to the global coordinates
	k = atma(t,k)

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
	return k

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
	t = coord_trans(xn1,xn2,Le,p)

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
	kg = atma(t,kg)

	# Check and enforce symmetry of the elastic stiffness matrix for the element
	kg = 0.5*(kg+kg.T)
	
	return kg

 #####  ####### #       #     # ####### ######  
#     # #     # #       #     # #       #     # 
#       #     # #       #     # #       #     # 
 #####  #     # #       #     # #####   ######  
      # #     # #        #   #  #       #   #   
#     # #     # #         # #   #       #    #  
 #####  ####### #######    #    ####### #     # 

def solve_system(K,nodemap,D,forces,con_dof):
	# we want to solve the matrix equation
	# |Kqq Kqr||xq| = |fq  |
	# |Krq Krr||xr|   |fr+c|
	# Where K, xr, fq, and fr are known. 

	#First step is to organize K, x, and f into that form
	#(we will use nodemap)

	spl_dex = len(K)-con_dof

	for nmap in nodemap:
		swap_Matrix_Rows(K,nmap[0],nmap[1])
		swap_Matrix_Cols(K,nmap[0],nmap[1])
		swap_Vector_Vals(forces,nmap[0],nmap[1])
		swap_Vector_Vals(D,nmap[0],nmap[1])

	#splitting the reorganized matrix up into the partially solved equations
	K1 = np.hsplit(K,np.array([spl_dex])) 
	[Kqq,Krq] = np.vsplit(K1[0],np.array([spl_dex]))
	[Kqr,Krr] = np.vsplit(K1[1],np.array([spl_dex]))

	#Knew = np.hstack((np.vstack((Kqq,Krq)),np.vstack((Kqr,Krr))))

	#for row in K-Knew:
	#	print(row)

	[xq,xr] = np.split(D,[spl_dex])
	[fq,fr] = np.split(forces,[spl_dex])
	

	#Now we want to solve the equation
	# Kqq xq + Kqr xr = fq

	Kqr_xr = np.dot(Kqr,xr)
	
	try:
		# Scipy solver- good for dense matrices (small DoF)
		#tC = sp.linalg.cho_factor(Kqq)
		#xq = sp.linalg.cho_solve(tC,fq-Kqr_xr)

		# Sparse Solver- using the CVXOPT cholesky solver 
		spKqq = sparse(matrix(Kqq))
		b = matrix(fq-Kqr_xr)
		cholmod.linsolve(spKqq,b)
		xq = np.array(b).T[0]

	except:
		print("Warning: Cholesky did not work")
		#If Cholesky dies (perhaps the matrix is not pos-def and symmetric)
		#We switch over to the sad scipy solver. very slow.
		xq = sp.linalg.solve(Kqq,fq-Kqr_xr)
	
	Krq_xq = np.dot(Krq,xq)
	Krr_xr = np.dot(Krr,xr)

	cr = Krq_xq+Krr_xr-fr

	D = np.append(xq,xr)
	C = np.append(np.zeros(spl_dex),cr)
	
	for nmap in nodemap:
		swap_Matrix_Rows(K,nmap[0],nmap[1])
		swap_Matrix_Cols(K,nmap[0],nmap[1])
		swap_Vector_Vals(forces,nmap[0],nmap[1])
		swap_Vector_Vals(D,nmap[0],nmap[1])
		swap_Vector_Vals(C,nmap[0],nmap[1])

	return D,C
	#for de in D:
	#	print(de)
	#for ce in C:
	#	print(ce)

####### ####### ######   #####  #######  #####  
#       #     # #     # #     # #       #     # 
#       #     # #     # #       #       #       
#####   #     # ######  #       #####    #####  
#       #     # #   #   #       #             # 
#       #     # #    #  #     # #       #     # 
#       ####### #     #  #####  #######  #####  
                                                

def assemble_loads(loads,constraints,tot_dof,length_scaling):
	# creates force vector b
	# Nodal Loads
	# virtual loads from prescribed displacements
	forces = np.zeros(tot_dof)
	dP = np.zeros(tot_dof)

	for load in loads:
		forces[6*load['node']+load['DOF']] = load['value']

	for constraint in constraints:
		dP[6*constraint['node']+constraint['DOF']] = constraint['value']*length_scaling

	return forces,dP

def element_end_forces(nodes,Q,beam_sets,D):

	for beamset,bargs in beam_sets:
		#every beam set lists the physical properties
		#associated with that beam
		
		#transfer those properties over
		beam_props = {"Ax"		: bargs["Ax"],
					  "Asy"		: bargs["Asy"],
					  "Asz"		: bargs["Asz"],
					  "G"		: bargs["G"],
					  "E"		: bargs["E"],
					  "J"		: bargs["J"],
					  "Iy"		: bargs["Iy"],
					  "Iz"		: bargs["Iz"],
					  "p"		: bargs["roll"],
					  "Le"		: bargs["Le"],
					  "shear"	: True}

		s = np.zeros(12)

		for m,beam in enumerate(beamset):
			dn1 = D[beam[0]*6:beam[0]*6+6]
			dn2 = D[beam[1]*6:beam[1]*6+6]
			beam_props["dn1"] = dn1
			beam_props["dn2"] = dn2
			xn1 = nodes[beam[0]]
			xn2 = nodes[beam[1]]
			beam_props["xn1"] = xn1
			beam_props["xn2"] = xn2

			#beam_props["Le"]  = sqrt((xn2[0]-xn1[0])**2+(xn2[1]-xn1[1])**2+(xn2[2]-xn1[2])**2) 
			#Things that are not faster than the above:
			#   sp.spatial.distance.euclidean(xn2,xn1)
			#beam_props["eqF_mech"] = Null
			frame_element_force(s,beam_props)

			Q[m,:] = s

		return Q

def frame_element_force(s,beam_props):
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
	xn1		= beam_props["xn1"]
	xn2		= beam_props["xn2"]
	dn1 	= beam_props["dn1"]
	dn2 	= beam_props["dn2"]
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
	shear 	= beam_props["shear"]

	f = np.zeros(12)

	t = coord_trans(xn1,xn2,Le,p)


	if shear:
		Ksy = 12.*E*Iz / (G*Asy*Le*Le)
		Ksz = 12.*E*Iy / (G*Asz*Le*Le)
		Dsy = (1+Ksy)*(1+Ksy)
		Dsz = (1+Ksz)*(1+Ksz)
	else:
		Ksy = Ksz = 0.0
		Dsy = Dsz = 1.0

	del1  = np.dot(dn2[0:3]-dn1[0:3],t[0:3]) # (d7-d1)*t1 + (d8-d2)*t2 + (d9-d3)*t3 	II
	del2  = np.dot(dn2[0:3]-dn1[0:3],t[3:6]) # (d7-d1)*t4 + (d8-d2)*t5 + (d9-d3)*t6 	III 
	del3  = np.dot(dn2[0:3]-dn1[0:3],t[6:9]) # (d7-d1)*t7 + (d8-d2)*t8 + (d9-d3)*t9 	III

	# del4  = np.dot(dn2[3:6]+dn1[3:6],t[0:3]) # (d4+d10)*t1 + (d5+d11)*t2 + (d6+d12)*t3  
	del5  = np.dot(dn2[3:6]+dn1[3:6],t[3:6]) # (d4+d10)*t4 + (d5+d11)*t5 + (d6+d12)*t6  I
	del6  = np.dot(dn2[3:6]+dn1[3:6],t[6:9]) # (d4+d10)*t7 + (d5+d11)*t8 + (d6+d12)*t9 	I
	
	del7  = np.dot(dn2[3:6]-dn1[3:6],t[0:3]) # (d10-d4)*t1 + (d11-d5)*t2 + (d12-d6)*t3 
	# del8  = np.dot(dn2[3:6]-dn1[3:6],t[3:6]) # (d10-d4)*t4 + (d11-d5)*t5 + (d12-d6)*t6 
	# del9  = np.dot(dn2[3:6]-dn1[3:6],t[6:9]) # (d10-d4)*t7 + (d11-d5)*t8 + (d12-d6)*t9 	

	# del10 = np.dot(dn1[3:6],t[0:3])			 # d4 *t1 + d5 *t2 + d6 *t3
	del11 = np.dot(dn1[3:6],t[3:6])			 # d4 *t4 + d5 *t5 + d6 *t6 
	del12 = np.dot(dn1[3:6],t[6:9])			 # d4 *t7 + d5 *t8 + d6 *t9

	# del13 = np.dot(dn2[3:6],t[0:3])			 # d10*t1 + d11*t2 + d12*t3
	del14 = np.dot(dn2[3:6],t[3:6])			 # d10*t4 + d11*t5 + d12*t6
	del15 = np.dot(dn2[3:6],t[6:9])			 # d10*t7 + d11*t8 + d12*t9

	axial_strain = del1 / Le

	#Axial force component
	s[0]  =  -(Ax*E/Le)*del1
	
	T = -s[0]
	#if geom:
	#T = -s[1]

	#Shear forces
	# positive Vy in local y direction
	s[1]  = -1.0*del2*(12.*E*Iz/(Le*Le*Le*(1.+Ksy)) + T/L*(1.2+2.0*Ksy+Ksy*Ksy)/Dsy) + \
		         del6*( 6.*E*Iz/(Le*Le*(1.+Ksy)) + T/10.0/Dsy)
	# positive Vz in local z direction
	s[2]  = -1.0*del3*(12.*E*Iy/(Le*Le*Le*(1.+Ksz)) + T/L*(1.2+2.0*Ksz+Ksz*Ksz)/Dsz) - \
		         del5*( 6.*E*Iy/(Le*Le*(1.+Ksz)) + T/10.0/Dsz) 
	#Torsion Forces
	# positive Tx r.h.r. about local x axis
	s[3]  = -1.0*del7*(G*J/Le)

	#Bending Forces
	#positive My -> positive x-z curvature
	s[4]  =  1.0*del3*( 6.*E*Iy/(Le*Le*(1.+Ksz)) + T/10.0/Dsz) + \
		         del11*((4.+Ksz)*E*Iy/(Le*(1.+Ksz)) + T*L*(2.0/15.0+Ksz/6.0+Ksz*Ksz/12.0)/Dsz ) + \
		         del14*((2.-Ksz)*E*Iy/(Le*(1.+Ksz)) - T*L*(1.0/30.0+Ksz/6.0+Ksz*Ksz/12.0)/Dsz )
	#positive Mz -> positive x-y curvature	         
	s[5]  = -1.0*del2*( 6.*E*Iz/(Le*Le*(1.+Ksy)) + T/10.0/Dsy) + \
			     del12*((4.+Ksy)*E*Iz/(Le*(1.+Ksy)) + T*L*(2.0/15.0+Ksy/6.0+Ksy*Ksy/12.0)/Dsy ) + \
				 del15*((2.-Ksy)*E*Iz/(Le*(1.+Ksy)) - T*L*(1.0/30.0+Ksy/6.0+Ksy*Ksy/12.0)/Dsy ) 

	s[6]  = -s[0];
	s[7]  = -s[1]; 
	s[8]  = -s[2]; 
	s[9]  = -s[3]; 

	s[10] =  1.0*del3*( 6.*E*Iy/(Le*Le*(1.+Ksz)) + T/10.0/Dsz ) + \
				 del14*((4.+Ksz)*E*Iy/(Le*(1.+Ksz)) + T*L*(2.0/15.0+Ksz/6.0+Ksz*Ksz/12.0)/Dsz ) + \
				 del11*((2.-Ksz)*E*Iy/(Le*(1.+Ksz)) - T*L*(1.0/30.0+Ksz/6.0+Ksz*Ksz/12.0)/Dsz )
	s[11] = -1.0*del2*( 6.*E*Iz/(Le*Le*(1.+Ksy)) + T/10.0/Dsy ) + \
				 del15*((4.+Ksy)*E*Iz/(Le*(1.+Ksy)) + T*L*(2.0/15.0+Ksy/6.0+Ksy*Ksy/12.0)/Dsy ) + \
				 del12*((2.-Ksy)*E*Iz/(Le*(1.+Ksy)) - T*L*(1.0/30.0+Ksy/6.0+Ksy*Ksy/12.0)/Dsy ) 


'''
EQUILIBRIUM_ERROR -  
compute: {dF_q} =   {F_q} - [K_qq]{D_q} - [K_qr]{D_r} 
 return: dF and ||dF||/||F||
'''
def equilibrium_error(K,nodemap,F,D,tot_dof,con_dof):
	dF = np.zeros(tot_dof)
	
	#First, reorganize K, F, D so that 
	#
	for nmap in nodemap:
		swap_Matrix_Rows(K,nmap[0],nmap[1])
		swap_Matrix_Cols(K,nmap[0],nmap[1])
		swap_Vector_Vals(F,nmap[0],nmap[1])
		swap_Vector_Vals(D,nmap[0],nmap[1])

	q = tot_dof-con_dof
	Fq = F[0:q].copy()
	dF[0:q] = Fq - np.dot(K[0:q,0:q],D[0:q]) - np.dot(K[0:q,q:tot_dof],D[q:tot_dof])
	
	for nmap in nodemap:
		swap_Matrix_Rows(K,nmap[0],nmap[1])
		swap_Matrix_Cols(K,nmap[0],nmap[1])
		swap_Vector_Vals(F,nmap[0],nmap[1])
		swap_Vector_Vals(dF,nmap[0],nmap[1])
		swap_Vector_Vals(D,nmap[0],nmap[1])

	norm = np.linalg.norm(Fq)
	#if norm==0:
	#	norm = 1
	return dF,np.linalg.norm(dF)/norm



   #    #     #    #    #       #     # ####### ####### 
  # #   ##    #   # #   #        #   #       #  #       
 #   #  # #   #  #   #  #         # #       #   #       
#     # #  #  # #     # #          #       #    #####   
####### #   # # ####### #          #      #     #       
#     # #    ## #     # #          #     #      #       
#     # #     # #     # #######    #    ####### ####### 
                                                        
 #####  #     #  #####  ####### ####### #     # 
#     #  #   #  #     #    #    #       ##   ## 
#         # #   #          #    #       # # # # 
 #####     #     #####     #    #####   #  #  # 
      #    #          #    #    #       #     # 
#     #    #    #     #    #    #       #     # 
 #####     #     #####     #    ####### #     # 

''' 
	ALGORITHM DETAILS:
	(Source: http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html)
    1. Assemble the elastic structural stiffness matrix 
       for the un-stressed structure, K(D=0).
    2. Compute the node displacements due to temperature 
       loads, Ft, using a linear elastic analysis, 
            Kqq (0) Dtq = Ftq.
    3. If geometric stiffness effects are to be considered,
       3a. Compute frame element end forces from the 
           displacements due to temperature loads, Q(Dt).
       3b. Assemble the structural stiffness matrix again,
           making use of the axial frame element forces
           arising from the temperature loads, K(Dt). 
    4. Compute the node displacements due to mechanical loads,
       Fm, only, including prescribed joint displacements, 
           Kqq (Dt) Dmq = Fmq - Kqr (Dt) Dr.
       (For a linear-elastic analysis, neglecting geometric
        stiffness, K = K(0).)
    5. Add the node displacements due to mechanical loads 
       to the node displacements due to temperature loads, 
           D1 = Dt + Dm,
       and combine the temperature and mechanical loads, 
           F = Ft + Fm.
    6. Compute frame element end forces from the 
       displacements due to the combined temperature and
       mechanical loads, Q(D1).
    7. If geometric stiffness effects are to be considered,
       carry out a second-order analysis via quasi Newton-
       Raphson iterations in order to converge upon the 
       displacements that satisfy equilibrium (starting with i=1).
       7a. Assemble the structural stiffness matrix again, 
           now making use of the axial frame element forces 
           arising from the combined temperature and mechanical 
           loads, K(D(i)),
       7b. Compute the equilibrium error at displacements 
               D(i): dFq(i) = Fq - Kqq(D(i)) Dq(i) - Kqr(D(i)) Dr
       7c. Compute the RMS relative equilibrium error criterion 
               || dFq(i) || / || Fq ||
       7d. Solve for the incremental displacements, 
               dD(i): Kqq(D(i)) dDq(i) = dFq(i)
       7e. Increment the displacements: 
               Dq(i+1) = Dq(i) + dDq(i)
       7f. Compute frame element end forces from the displacements
           due to the combined temperature and mechanical loads, Q(D(i)).
       7g. The Newton-Raphson iterations have converged when the
           root-mean-square relative equilibrium error (the root-
           mean-squre of dFq(i) divided by the root-mean-square of 
           Fq computed in step 3., above) is less than the specified 
           tolerance.
       7h. The convergence tolerance is taken as the convergence 
           tolerance for the dynamic modal analysis. 
           The default value is 10-9. 
    8. Compute the reaction forces for the converged solution: 
           Rr = Krq(D(i)) Dq(i) + Krr(D(i)) Dr - Fr

    Notation:
        subscript q indicates the set of displacement coordinates
        subscript r indicates the set of reaction coordinates
        superscript (i) indicates the iteration number
        Dq(i) is the vector of unknown displacements at iteration i
        Dr is the vector of known prescribed displacements (at reaction coordinates)
        K(D(i)) is the stiffness matrix, dependent upon displacements at iteration i
        Fq is the vector of known applied loads (at displacement coordinates)
        Fr is the vector of known applied loads (at reaction coordinates)
        Rr is the vector of unknown reaction forces for the converged solution
        dFq(i) is the equilibrium error vector at iteration i
        dDq(i) is the incremental displacement vector at iteration i
        Dq(i+1) is the vector of uknown displacements at iteration i+1
        Q(D(i)) is the set of frame element end forces at iteration i 
'''
def analyze_System(nodes, global_args, beam_sets, constraints,loads):
	'''
		nodes | is a numpy array of floats of shape (-1,3) specifying spatial location of nodes
  global_args | contains information about the whole problem:
  			     > 		 node_radius | is a rigid node radius
                 > 			 n_modes | is the number of dynamic modes to compute
                 > 	  length_scaling | is a spatial scale for numerical purposes. 
                                     | All lengths are multiplied, other units adjusted appropriately.
                 > frame3dd_filename | is the name of the CSV input file for frame3dd that is written.
    beam_sets | is a list of tuples beams,args.  
    	beams | is a numpy array of ints of shape (-1,2) referencing into nodes
    	 args | is a dictionary containing properties of the corresponding beam_set.  It must contain
		         >   'E': young's modulus
		         >  'nu': poisson ratio
		         > 'rho': density 
		         >  'dx': x dimension of cross section
		         >  'dy': y dimension of cross section
  constraints | are both lists of dictionaries with keys ['node','DOF','value']
    and loads | 
    	         >
	'''

	try: 
		n_modes = global_args['n_modes'] 
	except(KeyError): 
		n_modes = 0
	try: 
		length_scaling = global_args['length_scaling']
	except(KeyError): 
		length_scaling = 1.
	try: 
		node_radius = global_args['node_radius']*length_scaling 
	except(KeyError):
		node_radius=np.zeros(np.shape(nodes)[0])

	nE = sum(map(lambda x: np.shape(x[0])[0], beam_sets))    
	nodes = nodes*length_scaling

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
	con_dof = len(constraints) #constrained Degrees of Freedom
	
	global_args["dof"] = tot_dof
	#calculate the node mapping that allows for straightforward
	#organization of the stiffness matrix
	D,node_map = gen_Node_map(nodes,constraints)

	#Part 1 of the Above algorithm
	K = np.zeros((tot_dof,tot_dof))
	Q = np.zeros((nE,12))
	K = assemble_K(nodes,beam_sets,Q,global_args)

	#Part 2
	#This is where we'll solve for the displacements that occur
	#due to temperature loads

	#Part 3
	#We first calculate the frame element end forces due to 
	#the displacements from the temperature loads
	#Then we recalculate K due to these node displacements

	#Part 4
	#Calculate the node displacements due to mechanical loads
	#as well as prescribed node displacements
	F,dP = assemble_loads(loads,constraints,tot_dof,length_scaling)
	dD = dP
	C  = np.zeros(tot_dof)

	dD,C = solve_system(K,node_map,dD,F,con_dof)

	#Part 5 
	#Add together displacements due to temperature and
	#mechanical, as well as forces
	D = D + dD
	
	#Part 6
	element_end_forces(nodes,Q, beam_sets, D)

	error = 1.0
	
	dF,error = equilibrium_error(K,node_map,F,D,tot_dof,con_dof)
	if error==inf:
		error = 0.0
		print("Something's wrong: did you remember to set loads? Skipping Quasi-Newton")
	
	#Part 7
	#Quasi newton-raphson
	it = 0
	while error > 1.0e-9 and it < 10:
		it = it + 1
		
		K = assemble_K(nodes,beam_sets,Q,global_args)

		dF,error = equilibrium_error(K,node_map,F,D,tot_dof,con_dof)

		dD,C = solve_system(K,node_map,dD,dF,con_dof)

		D = D + dD
		Q = element_end_forces(nodes,Q,beam_sets,D)
		print("NR Iteration {0}".format(it))
		print("RMS relative equilibrium error = {0}".format(error))

	#Part 8 Find the reaction forces

	#Epilogue: Output the final node displacements.
	fin_node_disp = np.zeros((len(nodes),3))

	for n in range(len(nodes)):
		fin_node_disp[n:,] = np.array([D[n*6],D[n*6+1],D[n*6+2]])	
	

	return fin_node_disp,C,Q







