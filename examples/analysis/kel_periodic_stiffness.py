import pfea
import numpy as np

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

Bo = np.zeros((144,72))
idenind = [0,1,2,3,
		   4,5,5,4,
		   6,7,8,8,9,9,6,7,
		   10,11,11,10,
		   0,1,2,3]

for j,ind in enumerate(idenind):
	for i in range(6):
		Bo[j*6+i][ind*6+i] = 1.0

Ba = np.zeros((144,9))
itind = [[ 6,2],
		 [ 7,1],
		 [11,2],
		 [13,2],
		 [14,1],
		 [15,1],
		 [18,2],
		 [19,1],
		 [20,0],
		 [21,0],
		 [22,0],
		 [23,0]]

for coord in itind:
	for i in range(3):
		Ba[coord[0]*6+i][coord[1]*3+i] = 1.0


#14 nodes in total
nodes = [[0.0 ,0.25,0.5 ],
		 [0.0 ,0.5 ,0.25],
		 [0.0 ,0.5 ,0.75],
		 [0.0 ,0.75,0.5 ],
		 [0.25,0.0 ,0.5 ],
		 [0.25,0.5 ,0.0 ],
		 [0.25,0.5 ,1.0 ],
		 [0.25,1.0 ,0.5 ],
		 [0.5 ,0.0 ,0.25],
		 [0.5 ,0.0 ,0.75],
		 [0.5 ,0.25,0.0 ],
		 [0.5 ,0.25,1.0 ],
		 [0.5 ,0.75,0.0 ],
		 [0.5 ,0.75,1.0 ],
		 [0.5 ,1.0 ,0.25],
		 [0.5 ,1.0 ,0.75],
		 [0.75,0.0 ,0.5 ],
		 [0.75,0.5 ,0.0 ],
		 [0.75,0.5 ,1.0 ],
		 [0.75,1.0 ,0.5 ],
		 [1.0 ,0.25,0.5 ],
		 [1.0 ,0.5 ,0.25],
		 [1.0 ,0.5 ,0.75],
		 [1.0 ,0.75,0.5 ]]

#18 bars

frames = 	[[0 ,1 ],
			  [0 ,2 ],
			  [0 ,4 ],
			  [1 ,3 ],
			  [1 ,5 ],
			  [2 ,3 ],
			  [2 ,6 ],
			  [3 ,7 ],
			  [4 ,8 ],
			  [4 ,9 ],
			  [5 ,10],
			  [5 ,12],
			  [8 ,10],
			  [8 ,16],
			  [9 ,11],
			  [9 ,16],
			  [10,17],
			  [12,14],
			  [12,17],
			  [13,15],
			  [16,20],
			  [17,21],
			  [18,22],
			  [19,23]]

uc_dims = 0.01 #m
Emat = 11e9
numat = 0.3
rel_den = 0.01

Be = Be*uc_dims
nodes = uc_dims*np.array(nodes)

lb = uc_dims/(2*np.sqrt(2))


# rel_den = num_beams*A*lb/uc_dims**3 
#d1 = np.sqrt(rel_den/frame_num[framedex]*uc_dims**3/lb)
# rel_den = num_beams*pi*(d1/2)^2*lb/uc_dims^3
d1 = 2.0*np.sqrt(rel_den/(24.0*np.pi*lb)*uc_dims**3)

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
								 'shear': False})]

global_args = {"dof" : len(nodes)*6}


K_uc = pfea.provide_K(nodes,global_args,out_frames)
 
tk_uc = np.zeros((144,144))

for i,col in enumerate(K_uc.I):
	tk_uc[K_uc.J[i]][col] = K_uc[col*144+K_uc.J[i]]

K_uc = tk_uc


Do = -1*np.dot(np.linalg.pinv(np.dot(Bo.T,np.dot(K_uc,Bo))),np.dot(Bo.T,np.dot(K_uc,Ba)))

Da = np.dot(Bo,Do)+Ba

Kda = np.dot(Da.T,np.dot(K_uc,Da))



Ke = 1.0/(uc_dims**3)*np.dot(Be.T,np.dot(Kda,Be))

w,v = np.linalg.eig(Ke)
sw = np.sort(w)[::-1]
print(sw)
evals = np.array([sw[0],sw[1],sw[3]])
tform = np.array([[ 1, 2, 0],
				  [ 1,-1, 0],
				  [ 0, 0, 1]])

eig1 = Emat/(1-2*numat)
eig2 = Emat/(numat+1)
eig3 = Emat/(2*(numat+1))

base = np.dot(np.linalg.inv(tform),evals)/(Emat*np.pi*(d1/2.0)**2)*lb**2

compliance = np.linalg.inv(Ke)*Emat*rel_den*rel_den

print('\n'.join([' '.join(['{:>7.3f}'.format(item) for item in row]) 
  for row in compliance]))



#print(frame_names[framedex])

print("Alpha: {0}".format(base[0])) #*frame_props["Le"]**2/frame_props["d1"]**2))#np.dot(np.linalg.inv(tform),evals))
print("Beta: {0}".format(base[1])) #*frame_props["Le"]**2/frame_props["d1"]**2))
print("Gamma: {0}".format(base[2])) #*frame_props["Le"]**2/frame_props["d1"]**2))

print("Hydrostatic: {0}".format(evals[0]/Emat/rel_den))#np.dot(np.linalg.inv(tform),evals))
print("Deviatoric: {0}".format(evals[1]/Emat/rel_den))
print("Shear: {0}".format(evals[2]/Emat/rel_den))

print("\n")


