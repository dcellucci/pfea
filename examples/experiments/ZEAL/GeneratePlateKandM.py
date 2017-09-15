import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pfea.frame3dd
import subprocess
import pfea
import pfea.util.pfeautil
import cProfile
import pfea.geom.cuboct
from math import *
import pfea.solver
import os
import cvxopt as co
import scipy.sparse.linalg as spLinAlg
import csv


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


#Physical Voxel Properties
#Current voxel pitch is 3in
vox_pitch = 0.0762 #m
                                                                                
#Setting up a 2 by 2 by 4
size_x = 7;
size_y = 7;
size_z = 1;

#Temporary Material Matrix - NxNxN cubic grid (corresponding to cubic-octahedra)
# at the moment:
# 1's correspond to material being there
# 0's correspond to no material
mat_matrix = []
for i in range(0,size_x+2):
    tempcol = []
    for j in range(0,size_y+2):
        tempdep = [1]*(size_z+1)
  	tempdep.append(0)
  	tempdep[0] = 0
  	if(i*j*(i-(size_x+1))*(j-(size_y+1)) == 0):
            tempdep = [0]*(size_z+2)
  	tempcol.append(tempdep)
    mat_matrix.append(tempcol)

# Material Properties
        
node_radius = 0

#STRUT PROPERTIES
#Physical Properties
#Assuming a Ultem1000 with material properties taken from:
#https://www.plasticsintl.com/datasheets/ULTEM_GF30.pdf
#Using Newtons, meters, and kilograms as the units
frame_props = {"nu"  : 0.36, #poisson's ratio +
               "d1"	 : 0.00127, #m
               "d2"	 : 0.00127, #m
               "E"   : 3580000000, #N/m^2 +
               "G"   : 3510000000,  #N/m^2 +
               "rho" :  1270, #kg/m^3
               "beam_divisions" : 0,
               "cross_section"  : 'rectangular',
               "roll": 0,
               "Le":vox_pitch/sqrt(2.0)}

#Node Map Population
#Referencing the geometry-specific kelvin.py file. 
#Future versions might have different files?
    
node_frame_map = np.zeros((size_x,size_y,size_z,6))
nodes,frames,node_frame_map,dims = pfea.geom.cuboct.from_material(mat_matrix,vox_pitch)
frame_props["Le"] = pfea.geom.cuboct.frame_length(vox_pitch)

with open('nodes7X7.csv','wb') as nodeFile:
    wr = csv.writer(nodeFile, quoting=csv.QUOTE_ALL)
    wr.writerows(nodes)
with open('edges7X7.csv','wb') as edgeFile:
    wr2 = csv.writer(edgeFile, quoting=csv.QUOTE_ALL)
    wr2.writerows(frames)

force = 12
weight = 72.2*9.8*1e-3
CM = [0.0762,0.0762,2*0.0762]
Cf = 0.02

#Constraint and load population
constraints = []
loads = []

#####  ### #     #    ####### #     # ####### ######  #     # ####### 
#       #  ##   ##    #     # #     #    #    #     # #     #    #    
#       #  # # # #    #     # #     #    #    #     # #     #    #    
#####   #  #  #  #    #     # #     #    #    ######  #     #    #
    #   #  #     #    #     # #     #    #    #       #     #    #    
    #   #  #     #    #     # #     #    #    #       #     #    #        
#####  ### #     #    #######  #####     #    #        #####     #

#Group frames with their characteristic properties.
out_frames = [(np.array(frames),{'E'   : frame_props["E"],
                'rho' : frame_props["rho"],
                'nu'  : frame_props["nu"],
                'd1'  : frame_props["d1"],
                'd2'  : frame_props["d2"],
                'beam_divisions' : frame_props["beam_divisions"],
                'cross_section'  : frame_props["cross_section"],
                'roll': frame_props["roll"],
                'loads':{'element':0},
                'prestresses':{'element':0},
                'Le': frame_props["Le"]})]
    
#Format node positions
out_nodes = np.array(nodes)
    
#Global Arguments 
global_args = {'frame3dd_filename': os.path.join('experiments','Results','test'),"lump": False, 'length_scaling':1,"using_Frame3dd":False,"debug_plot":True, "gravity" : [0,0,0],"save_matrices":True}

'''if global_args["using_Frame3dd"]:
        frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
        subprocess.call("frame3dd -i {0}.csv -o {0}.out -q".format(global_args["frame3dd_filename"]), shell=True)
        res_nodes, res_reactions = frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
        res_displace = frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])
else:
	res_displace,C,Q = pfea.solver.analyze_System(out_nodes, global_args, out_frames, constraints,loads)

#print res_displace
#print res_displace
temp = np.where(out_nodes[:,0]==min(out_nodes[:,0]))
#temp = np.where(res_displace[:,0] < min(res_displace[:,0])+0.01*min(res_displace[:,0]))
#print temp
#print res_displace[temp,0]
botRes_Disp = res_displace[temp,0]
#print botRes_Disp[0,:]
minIndex = np.where(botRes_Disp[0,:]<min(botRes_Disp[0,:])+abs(min(botRes_Disp[0,:]))*0.001)
bottomNodes = temp[0]
bottomDisp = np.zeros((len(bottomNodes),3))
#print out_nodes[bottomNodes,:]#+res_displace[bottomNodes[minIndex],0:3]
#print (out_nodes[bottomNodes,:]/0.0381+1)/2
reactForce = np.zeros((len(bottomDisp),3))
for i in minIndex:
    bottomDisp[i,:] = out_nodes[bottomNodes[i],:]+res_displace[bottomNodes[i],0:3]
    reactForce[i,:] = [weight/len(bottomDisp),Cf*weight/len(bottomDisp),0]
#print bottomDisp
#print reactForce
reactions = np.zeros((8,6))
reactions[:,0:3] = reactForce
reactions[:,3:7] = np.dot(np.cross(bottomDisp,reactForce),np.array([[1,0,0],[0,1,0],[0,0,1]]))
#print np.dot(np.cross(bottomDisp,reactForce),np.array([[1,0,0],[0,1,0],[0,0,1]]))
#print reactions

np.savetxt('VoxelWorm.csv', out_frames[0][0], delimiter=',')'''

#pfea.util.pfeautil.plotLattice(nodes,frames,res_displace,1)
pfea.solver.write_K(out_nodes,out_frames,global_args,'A7X7.txt')
pfea.solver.write_M(out_nodes,out_frames,global_args,'M7X7.txt')
#pfea.util.pfeautil.writeCSV(nodes,res_displace,'Force12NCompression.csv')

#M = pfea.solver.provide_M(out_nodes,out_frames,global_args)
#K = pfea.solver.provide_K(out_nodes,out_frames,global_args)
