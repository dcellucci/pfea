import pfea
import numpy as np
from scipy.optimize import curve_fit
#See Appendix A of "Stiffness and strength of tridimensional periodic lattices" By Vigliotti and Pasini



import matplotlib
import matplotlib.pyplot as plt

Be = np.zeros((9,6))
Be[0][0] = 1.0
Be[4][1] = 1.0
Be[8][2] = 1.0
Be[1][3] = 0.5
Be[2][5] = 0.5
Be[3][3] = 0.5
Be[5][4] = 0.5
Be[6][5] = 0.5
Be[7][4] = 0.5

#Specific to Kelvin-Type Lattices
#32 nodes total, 20 unique
Bo = np.zeros((192,120))
idenind = [0 ,1 ,2 ,3 ,4 ,5 , 
		   6 ,7 ,7 ,8 ,9 ,4 , 
		   10,11,12,12,13,13,
		   10,11,14,15,16,17,
		   17,18,19,14,0 ,1 ,
		   2 ,3] 

for j,ind in enumerate(idenind):
	for i in range(6):
		Bo[j*6+i][ind*6+i] = 1.0

Ba = np.zeros((192,9))
itind = [[ 8,2],
		 [11,1],
		 [15,2],
		 [17,2],
		 [18,1],
		 [19,1],
		 [24,2],
		 [27,1],
		 [28,0],
		 [29,0],
		 [30,0],
		 [31,0]]

for coord in itind:
	for i in range(3):
		Ba[coord[0]*6+i][coord[1]*3+i] = 1.0


#32 nodes in total
nodes = [[0.0 ,0.25,0.5 ], #0   0
		 [0.0 ,0.5 ,0.25], #1   1
		 [0.0 ,0.5 ,0.75], #2   2
		 [0.0 ,0.75,0.5 ], #3   3
		 [0.25,0.0 ,0.5 ], #4   4
		 [0.25,0.25,0.25], #5   5  - Psch
		 [0.25,0.25,0.75], #6   6  - Psch
		 [0.25,0.5 ,0.0 ], #7   7
		 [0.25,0.5 ,1.0 ], #7   8
		 [0.25,0.75,0.25], #8   9  - Psch
		 [0.25,0.75,0.75], #9   10 - Psch
		 [0.25,1.0 ,0.5 ], #4   11
		 [0.5 ,0.0 ,0.25], #10  12
		 [0.5 ,0.0 ,0.75], #11  13
		 [0.5 ,0.25,0.0 ], #12  14
		 [0.5 ,0.25,1.0 ], #12  15
		 [0.5 ,0.75,0.0 ], #13  16
		 [0.5 ,0.75,1.0 ], #13  17
		 [0.5 ,1.0 ,0.25], #10  18
		 [0.5 ,1.0 ,0.75], #11  19
		 [0.75,0.0 ,0.5 ], #14  20
		 [0.75,0.25,0.25], #15  21 - Psch
		 [0.75,0.25,0.75], #16  22 - Psch
		 [0.75,0.5 ,0.0 ], #17  23
		 [0.75,0.5 ,1.0 ], #17  24
		 [0.75,0.75,0.25], #18  25 - Psch
		 [0.75,0.75,0.75], #19  26 - Psch
		 [0.75,1.0 ,0.5 ], #14  27
		 [1.0 ,0.25,0.5 ], #0   28
		 [1.0 ,0.5 ,0.25], #1   29
		 [1.0 ,0.5 ,0.75], #2   30
		 [1.0 ,0.75,0.5 ]] #3   31

#24 Frames in the kelvin lattice (references nodes array)
kel_frames = [[0 ,1 ],
			  [0 ,2 ],
			  [0 ,4 ],
			  [1 ,3 ],
			  [1 ,7 ],
			  [2 ,3 ],
			  [2 ,8 ],
			  [3 ,11],
			  [4 ,12],
			  [4 ,13],
			  [7 ,14],
			  [7 ,16],
			  [12,14],
			  [12,20],
			  [13,15],
			  [13,20],
			  [14,23],
			  [16,18],
			  [16,23],
			  [17,19],
			  [20,28],
			  [23,29],
			  [24,30],
			  [27,31]]

#48 Frames in P-Schwarz (references node array)
psch_frames =[[0 ,5 ],
			  [0 ,6 ],
			  [1 ,5 ],
			  [1 ,9 ],
			  [2 ,6 ],
			  [2 ,10],
			  [3 ,9 ],
			  [3 ,10],
			  [4 ,5 ],
			  [4 ,6 ],
			  [5 ,7 ],
			  [5 ,12],
			  [5 ,14],
			  [6 ,8 ],
			  [6 ,13],
			  [6 ,15],
			  [7 ,9 ],
			  [8 ,10],
			  [9 ,11],
			  [9 ,16],
			  [9 ,18],
			  [10,11],
			  [10,17],
			  [10,19],
			  [12,21],
			  [13,22],
			  [14,21],
			  [15,22],
			  [16,25],
			  [17,26],
			  [18,25],
			  [19,26],
			  [20,21],
			  [20,22],
			  [21,23],
			  [21,28],
			  [21,29],
			  [22,24],
			  [22,28],
			  [22,30],
			  [23,25],
			  [24,26],
			  [25,27],
			  [25,29],
			  [25,31],
			  [26,27],
			  [26,30],
			  [26,31]]

#Basic parameters
uc_dims = 0.1 #m, 
Emat = 1 #Pa, the material modulus
numat = 0.3 #Material poisson's ratio
#rel_den = 0.1 #Relative density
lb = uc_dims/(2*np.sqrt(2)) #m, frame length

#Absolute dimensions
Be = Be*uc_dims
nodes = uc_dims*np.array(nodes)

#Allows easy switching between lattice topologies
frame_names = ["Kelvin","PSch"]
frame_list = [kel_frames,psch_frames]
frame_num = [24,48]

rel_dens = [0.005,0.01,0.05,0.1]
debugprint = False

frame_index = 0
for framedex in [frame_index]: #see the frame_* variables for reference
	frames = frame_list[framedex]
	
	d1s = []
	
	alphas = []
	betas = []
	gammas = []

	for rel_den in rel_dens:
		# rel_den = num_beams*A*lb/uc_dims**3 
		#d1 = np.sqrt(rel_den/frame_num[framedex]*uc_dims**3/lb)
		# rel_den = num_beams*pi*(d1/2)^2*lb/uc_dims^3
		
		#Solid, circular cross-section frame elements
		d1 = 2.0*np.sqrt(rel_den/(frame_num[framedex]*np.pi*lb)*uc_dims**3)

		d1s.append(d1)
		frame_props = {"nu"  : numat, #poisson's ratio
							   "d1"	 : d1, #m
							   "d2"	 : d1, #m
							   "th"  : d1/2.0,
							   "E"   :  Emat, #N/m^2,
							   "rho" :  1650, #kg/m^3
							   "Le"  : lb,
							   "beam_divisions" : 0,
							   "cross_section"  : 'circular',
							   "roll": 0}



		out_frames = [(np.array(frames),0,{'E'   : frame_props["E"],
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
										 'shear': False})]

		global_args = {"dof" : len(nodes)*6}

		#Stiffness matrix for the frame
		K_uc = pfea.solver.provide_K(nodes,global_args,out_frames)
		 
		#Conversion from sparse matrix to dense
		tk_uc = np.zeros((192,192))

		for i,col in enumerate(K_uc.I):
			tk_uc[K_uc.J[i]][col] = K_uc[col*192+K_uc.J[i]]

		K_uc = tk_uc

		#Eq. 11 of Vigliotti & Pasini
		Do = -1*np.dot(np.linalg.pinv(np.dot(Bo.T,np.dot(K_uc,Bo))),np.dot(Bo.T,np.dot(K_uc,Ba)))

		#Eq. 12
		Da = np.dot(Bo,Do)+Ba

		#Eq. 13 
		Kda = np.dot(Da.T,np.dot(K_uc,Da))

		#Eq. 20
		Ke = 1.0/(uc_dims**3)*np.dot(Be.T,np.dot(Kda,Be))

		w,v = np.linalg.eig(Ke)
		sw = np.sort(w)[::-1]
		#print(sw)
		evals = np.array([sw[0],sw[1],sw[3]])
		tform = np.array([[ 1, 2, 0],
						  [ 1,-1, 0],
						  [ 0, 0, 1]])

		eig1 = Emat/(1-2*numat)
		eig2 = Emat/(numat+1)
		eig3 = Emat/(2*(numat+1))

		base = np.dot(np.linalg.inv(tform),evals)/(Emat)#(Emat*np.pi*(d1/2.0)**2)*lb**2

		compliance = np.linalg.inv(Ke)*Emat*rel_den

		alphas.append(base[0])
		betas.append(base[1])
		gammas.append(base[2])
		if(debugprint):
			print('\n'.join([' '.join(['{:>7.3f}'.format(item) for item in row]) 
			  for row in compliance]))

			print(frame_names[framedex])

			print("Alpha: {0}".format(base[0])) #*frame_props["Le"]**2/frame_props["d1"]**2))#np.dot(np.linalg.inv(tform),evals))
			print("Beta: {0}".format(base[1])) #*frame_props["Le"]**2/frame_props["d1"]**2))
			print("Gamma: {0}".format(base[2])) #*frame_props["Le"]**2/frame_props["d1"]**2))

			print("Hydrostatic: {0}".format(evals[0]/Emat/rel_den))#np.dot(np.linalg.inv(tform),evals))
			print("Deviatoric: {0}".format(evals[1]/Emat/rel_den))
			print("Shear: {0}".format(evals[2]/Emat/rel_den))

			print("\n")

def func(d1, a, b):
	return np.pi*(0.5*d1)**2/lb**2*a+0.25*np.pi*(0.5*d1)**4/lb**4*b

alphas = np.array(alphas)
betas = np.array(betas)
gammas = np.array(gammas)

print((alphas-betas)/(Emat/(numat+1)))

plt.title(frame_names[frame_index])
plt.plot(rel_dens,(alphas-betas)/(Emat/(numat+1)))
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.003,0.3)
plt.show()

print("       Axial | Bending")
popt, pcov = curve_fit(func, d1s, np.real(alphas))
print("a/E_s| {0:6.3f}|{1:6.3f}".format(popt[0],popt[1]))
popt, pcov = curve_fit(func, d1s, np.real(betas))
print("b/E_s| {0:6.3f}|{1:6.3f}".format(popt[0],popt[1]))
popt, pcov = curve_fit(func, d1s, np.real(gammas))
print("g/E_s| {0:6.3f}|{1:6.3f}".format(popt[0],popt[1]))
#print np.sqrt(np.diag(pcov))

