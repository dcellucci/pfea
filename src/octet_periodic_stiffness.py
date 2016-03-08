import pfea
import numpy as np


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

frames = [[ 0, 2],
		  [ 0, 5],
		  [ 0, 6],
		  [ 1, 2],
		  [ 1, 5],
		  [ 1, 7],
		  [ 2, 3],
		  [ 2, 4],
		  [ 2, 5],
		  [ 2, 6],
		  [ 2, 7],
		  [ 2, 8],
		  [ 3, 6],
		  [ 3, 8],
		  [ 4, 7],
		  [ 4, 8],
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
		  [ 7,10],
		  [ 7,11],
		  [ 7,13],
		  [ 8,11],
		  [ 8,12],
		  [ 8,13],
		  [ 9,11],
		  [10,11],
		  [11,12],
		  [11,13]
		  ]

uc_dims = 0.01 #m
Emat = 11e9
numat = 0.35
rel_den = 0.01


d1 = np.sqrt(np.sqrt(2)/24.0*rel_den)*uc_dims

frame_props = {"nu"  : numat, #poisson's ratio
					   "d1"	 : d1, #m
					   "d2"	 : d1, #m
					   "th"  : 0,
					   "E"   :  Emat, #N/m^2,
					   "rho" :  1650, #kg/m^3
					   "Le"  : uc_dims/np.sqrt(2),
					   "beam_divisions" : 0,
					   "cross_section"  : 'rectangular',
					   "roll": 0}



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
								 'beam_divisions': 1})]

global_args = {"dof" : len(nodes)*6}

nodes = uc_dims*np.array(nodes)

K_uc = pfea.provide_K(nodes,global_args,out_frames)
 
tk_uc = np.zeros((84,84))

for i,col in enumerate(K_uc.I):
	tk_uc[K_uc.J[i]][col] = K_uc[col*84+K_uc.J[i]]

K_uc = tk_uc

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

Do = -1*np.dot(np.linalg.pinv(np.dot(Bo.T,np.dot(K_uc,Bo))),np.dot(Bo.T,np.dot(K_uc,Ba)))

Da = np.dot(Bo,Do)+Ba

Kda = np.dot(Da.T,np.dot(K_uc,Da))

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

Be = Be*uc_dims

Ke = 1.0/(uc_dims**3)*np.dot(Be.T,np.dot(Kda,Be))

w,v = np.linalg.eig(Ke)

sw = np.sort(w)[::-1]

evals = np.array([sw[0],sw[1],sw[3]])
tform = np.array([[ 1, 2, 0],
				  [ 1,-1, 0],
				  [ 0, 0, 1]])
eig1 = Emat/(1-2*numat)
eig2 = Emat/(numat+1)
eig3 = Emat/(2*(numat+1))
print("Hydrostatic: {0}".format(evals[0]/eig1))#np.dot(np.linalg.inv(tform),evals))
print("Deviatoric: {0}".format(evals[1]/eig2))
print("Shear: {0}".format(evals[2]/eig3))


