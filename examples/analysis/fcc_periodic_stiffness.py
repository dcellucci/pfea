# See the README for more info.
import pfea
import numpy as np
from scipy.optimize import curve_fit

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

#Populating a block matrix full of 6x6 identity matrices
Bo = np.zeros((84,24))
idenind = [0,0,1,0,0,2,3,3,2,0,0,1,0,0]
for j,ind in enumerate(idenind):
	for i in range(6):
		Bo[j*6+i][ind*6+i] = 1.0

Ba = np.zeros((84,9))
itind = [[ 1,2],
		 [ 3,1],
		 [ 4,1],
		 [ 4,2],
		 [ 7,2],
		 [ 8,1],
		 [ 9,0],
		 [10,0],
		 [10,2],
		 [11,0],
		 [12,0],
		 [12,1],
		 [13,0],
		 [13,1],
		 [13,2]]

for coord in itind:
	for i in range(3):
		Ba[coord[0]*6+i][coord[1]*3+i] = 1.0


#14 nodes in total
nodes = [[0.0,0.0,0.0],
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
#18 bars

fcc_frames = 	 [[ 0, 2],
				  [ 0, 5],
				  [ 0, 6],
				  [ 1, 2],
				  [ 1, 5],
				  #[ 1, 7],
				  [ 2, 3],
				  [ 2, 4],
				  [ 3, 6],
				  #[ 3, 8],
				  #[ 4, 7],
				  #[ 4, 8],
				  [ 5, 9],
				  [ 5,10],
				  [ 6, 9],
				  [ 6,12]#,
				  #[ 7,10],
				  #[ 7,13],
				  #[ 8,12],
				  #[ 8,13],
				  #[ 9,11],
				  #[10,11],
				  #[11,12],
				  #[11,13]
				  ]
fcc_cube_frame = [[0,1],
				  [0,3],
				  [0,9]]

octet_frames = 	 [[ 0, 2],
				  [ 0, 5],
				  [ 0, 6],
				  [ 1, 2],
				  [ 1, 5],
				  #[ 1, 7],
				  [ 2, 3],
				  [ 2, 4],
				  [ 2, 5],
				  [ 2, 6],
				  [ 2, 7],
				  [ 2, 8],
				  [ 3, 6],
				  #[ 3, 8],
				  #[ 4, 7],
				  #[ 4, 8],
				  [ 5, 6],
				  [ 5, 7],
				  [ 5, 9],
				  [ 5,10],
				  [ 5,11],
				  [ 6, 8],
				  [ 6, 9],
				  [ 6,11],
				  [ 6,12],
				  [ 7, 8],
				  #[ 7,10],
				  [ 7,11],
				  #[ 7,13],
				  [ 8,11]#,
				  #[ 8,12],
				  #[ 8,13],
				  #[ 9,11],
				  #[10,11],
				  #[11,12],
				  #[11,13]
				  ]

d_sch_frames = 	 [[ 0, 2],
				  [ 0, 5],
				  [ 2, 4],
				  [ 2, 6],
				  [ 2, 8],
				  [ 3, 6],
				  #[ 3, 8],
				  #[ 4, 7],
				  [ 5, 6],
				  [ 5,10],
				  [ 5,11],
				  [ 6, 9],
				  [ 7, 8],
				  #[ 7,10],
				  [ 7,11]#,
				  #[ 8,13],
				  #[ 9,11],
				  #[11,13]
				  ]

cuboct_frames = [[2,5],
				 [2,6],
				 [2,7],
				 [2,8],
				 ]

uc_dims = 0.1 #m
Emat = 1
numat = 0.3
rel_den = 0.1

Be = Be*uc_dims
nodes = uc_dims*np.array(nodes)

frame_names = ["FCC","Octet","D-Schwarz","CubOct"]
frame_list = [fcc_frames,octet_frames,d_sch_frames,cuboct_frames]
frame_num = [12.0,24.0,12.0,12.0]

debugprint = False

rel_dens = [0.005,0.01,0.05,0.1]

frame_index = 1

for framedex in [frame_index]:
	frames = frame_list[framedex]
	lb = uc_dims/np.sqrt(2)

	d1s = []
	
	alphas = []
	betas = []
	gammas = []

	for rel_den in rel_dens:

		if(framedex == 0):
			# rel_den = (12*A*lb/sqrt(2)+3*A*lb)/lb^3
			# rel_den = (6/sq(2)+3/2)A/lb^2
			# rel_den = A*3/2(2sq(2)+1)/lb^2
			# A = rel_den*lb^2/(3/2*(2sq(2)+1))

			d1 = 2.0*np.sqrt(rel_den*lb**2/(np.pi*3.0/2.0*(2*np.sqrt(2)+1)))

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
											 'shear': False}),
						(np.array(fcc_cube_frame),0,{'E'   : frame_props["E"],
											 'rho' : frame_props["rho"],
											 'nu'  : frame_props["nu"],
											 'd1'  : frame_props["d1"],
											 'd2'  : frame_props["d2"],
											 'th'  : frame_props["th"],
											 'cross_section'  : frame_props["cross_section"],
											 'roll': frame_props["roll"],
											 'loads':{'element':0},
											 'prestresses':{'element':0},
											 'Le': uc_dims,
											 'beam_divisions': 1,
											 'shear': False})]
		else:
			# rel_den = num_beams*A*lb/uc_dims**3 
			#d1 = np.sqrt(rel_den/frame_num[framedex]*uc_dims**3/lb)
			# rel_den = num_beams*pi*(d1/2)^2*lb/uc_dims^3
			d1 = 2.0*np.sqrt(rel_den/frame_num[framedex]/np.pi*uc_dims**3/lb)

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
		d1s.append(d1)
		global_args = {"dof" : len(nodes)*6}

		
		K_uc = pfea.solver.provide_K(nodes,global_args,out_frames)
		 
		tk_uc = np.zeros((84,84))

		for i,col in enumerate(K_uc.I):
			tk_uc[K_uc.J[i]][col] = K_uc[col*84+K_uc.J[i]]

		K_uc = tk_uc


		Do = -1*np.dot(np.linalg.pinv(np.dot(Bo.T,np.dot(K_uc,Bo))),np.dot(Bo.T,np.dot(K_uc,Ba)))

		Da = np.dot(Bo,Do)+Ba

		Kda = np.dot(Da.T,np.dot(K_uc,Da))



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

		base = np.dot(np.linalg.inv(tform),evals)/(Emat)#np.pi*(d1/2.0)**2)*lb**2


		compliance = np.linalg.inv(Ke)*Emat*rel_den
		
		alphas.append(base[0])
		betas.append(base[1])
		gammas.append(base[2])
		
		if(debugprint):
			print('\n'.join([' '.join(['{:>6.3f}'.format(item) for item in row]) 
		      for row in compliance]))

			print(frame_names[framedex])
			print("compare: {0}".format(np.sqrt(2)*(d1/2.0)**2*np.pi/lb**2+12*np.sqrt(2)*0.25*(d1/2.0)**4*np.pi/lb**4))
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

