
import numpy as np
from pfeautil import *
from math import *
import scipy as sp

#The sparse solver
import cvxopt as co
from cvxopt import cholmod

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
	
	We need to solve for Forces and Moments on the nodes

	
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
	node_map = gen_Node_map(nodes,constraints)

	D = co.matrix(0.0,(1,tot_dof))

	#Part 1 of the Above algorithm
	K = co.spmatrix([],[],[],(tot_dof,tot_dof))
	Q = co.matrix(0.0,(nE,12))
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
	F,dP = assemble_loads(loads,constraints,nodes,beam_sets,global_args,tot_dof,length_scaling)
	dD = dP
	C  = np.zeros(tot_dof)
	dD,C = solve_system(K,node_map,dD,F,con_dof)

	
	#Part 5 
	#Add together displacements due to temperature and
	#mechanical, as well as forces
	D = D.T + dD
	
	#Part 6
	element_end_forces(nodes,Q, beam_sets, D)
	
	error = 1.0
	dF,error = equilibrium_error(K,node_map,F,D,tot_dof,con_dof)
	
	
	#Part 7
	#Quasi newton-raphson
	it = 0
	
	if error == inf:
		error = 0.0
		print("Something's wrong: did you remember to set loads? Skipping Quasi-Newton")
		it = 11
	
	#print(error)
	
	lasterror = 1.0
	error = 0.5
	
	while np.abs(error-lasterror) > 0.01*error and error > 1e-9 and it < 10:
		it = it + 1
		
		K = assemble_K(nodes,beam_sets,Q,global_args)
		lasterror = error
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







