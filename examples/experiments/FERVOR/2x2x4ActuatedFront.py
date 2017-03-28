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

def applyForce(force,reactions):
    #Constraint and load population
    constraints = []
    loads = []
    
    '''Vertical mount'''
    #Displacement Constraints
    constraints.append({'node':node_frame_map[1][1][1][5],'DOF':0, 'value':0})
    constraints.append({'node':node_frame_map[1][1][1][5],'DOF':1, 'value':0})
    constraints.append({'node':node_frame_map[1][1][1][5],'DOF':2, 'value':0})
    constraints.append({'node':node_frame_map[1][2][1][5],'DOF':0, 'value':0})
    constraints.append({'node':node_frame_map[1][2][1][5],'DOF':1, 'value':0})
    constraints.append({'node':node_frame_map[1][2][1][5],'DOF':2, 'value':0})

    #Rotational Constraints
    constraints.append({'node':node_frame_map[1][1][1][5],'DOF':3, 'value':0})
    constraints.append({'node':node_frame_map[1][1][1][5],'DOF':4, 'value':0})
    constraints.append({'node':node_frame_map[1][1][1][5],'DOF':5, 'value':0})
    constraints.append({'node':node_frame_map[1][2][1][5],'DOF':3, 'value':0})
    constraints.append({'node':node_frame_map[1][2][1][5],'DOF':4, 'value':0})
    constraints.append({'node':node_frame_map[1][2][1][5],'DOF':5, 'value':0})
    constraints.append({'node':node_frame_map[2][1][1][5],'DOF':3, 'value':0})
    constraints.append({'node':node_frame_map[2][1][1][5],'DOF':4, 'value':0})
    constraints.append({'node':node_frame_map[2][1][1][5],'DOF':5, 'value':0})
    constraints.append({'node':node_frame_map[2][2][1][5],'DOF':3, 'value':0})
    constraints.append({'node':node_frame_map[2][2][1][5],'DOF':4, 'value':0})
    constraints.append({'node':node_frame_map[2][2][1][5],'DOF':5, 'value':0})
    
    #Loads
    #loads.append({'node':node_frame_map[1][1][2][5],'DOF':1, 'value':force})
    #loads.append({'node':node_frame_map[1][2][2][5],'DOF':0, 'value':force})
    loads.append({'node':node_frame_map[2][1][1][5],'DOF':0, 'value':force})
    loads.append({'node':node_frame_map[2][2][1][5],'DOF':0, 'value':force})

    index = np.array([[1,1,1],[1,1,2],[1,1,3],[1,1,4],[1,2,1],[1,2,2],[1,2,3],[1,2,4]])
    for i in np.arange(0,8):
        loads.append({'node':node_frame_map[index[i][0]][index[i][1]][index[i][2]][0],'DOF':0, 'value':reactions[i][0]})
        loads.append({'node':node_frame_map[index[i][0]][index[i][1]][index[i][2]][0],'DOF':1, 'value':reactions[i][1]})
        loads.append({'node':node_frame_map[index[i][0]][index[i][1]][index[i][2]][0],'DOF':2, 'value':reactions[i][2]})
        
    return constraints,loads

def solveIteration(frames,frame_props,nodes,constraints,loads):
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
    
    if global_args["using_Frame3dd"]:
            frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
            subprocess.call("frame3dd -i {0}.csv -o {0}.out -q".format(global_args["frame3dd_filename"]), shell=True)
            res_nodes, res_reactions = frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
            res_displace = frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])
    else:
            res_displace,C,Q = pfea.solver.analyze_System(out_nodes, global_args, out_frames, constraints,loads)

    return res_displace,out_nodes,global_args,out_frames,C

def calcReactionForces(weight,res_displace,out_nodes,CM,Cf,bottomIndex,prevNodes,prevReact):
    #Sort left and right side of the center of mass
    leftCMSide = np.where(out_nodes[bottomIndex,2]+res_displace[bottomIndex,2]-CM[2]>0)
    rightCMSide = np.where(out_nodes[bottomIndex,2]+res_displace[bottomIndex,2]-CM[2]<0)

    #Create list of index touching feet
    temp = bottomIndex[0]
    botLeftRes_disp = res_displace[temp[leftCMSide[1]],:]
    botRightRes_disp = res_displace[temp[rightCMSide[1]],:]

    minLeftIndex = np.where(botLeftRes_disp[:,0]<min(botLeftRes_disp[:,0])+abs(min(botLeftRes_disp[:,0]))*0.001)
    minRightIndex = np.where(botRightRes_disp[:,0]<min(botRightRes_disp[:,0])+abs(min(botRightRes_disp[:,0]))*0.001)
    minIndex = np.concatenate((leftCMSide[1][minLeftIndex],rightCMSide[1][minRightIndex]),axis=0)

    #Create array of bottom displacement
    botRes_Disp = res_displace[bottomIndex,0]

    #Intialize arrays
    bottomNodes = bottomIndex[0]
    bottomDisp = np.zeros((8,3))
    reactions = np.zeros((8,3))

    A = np.zeros((4,len(minIndex)))
    B = np.array([weight,0,0,0])
    identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
    cofFrict = np.zeros((len(minIndex),1));
    count = 0

    for i in minIndex:
        bottomDisp[i,:] = out_nodes[bottomNodes[i],:]+res_displace[bottomNodes[i],0:3]
        A[0,count] = 1
        #If the previous node value is greater than the current one then that means the foot is moving in the "forward" direction or towards the origin in the z direction. This corresponds
        #with the "smooth" side of the feet and the smooth friction coefficent is used in the direction opposite of the motion which is in the positive direction. When the opposite is true
        # then the force goes in the opposite direction and the "rough" coefficient of friction is used.
        #    ^  Current                 Previous
        #   x|   XX                     XX
        #   z-> XXX                     XXX ->CfSmooth*ReactionForce
        if bottomDisp[i,2]<prevNodes[bottomNodes[i],2]:
            cofFrict[count] = Cf[1]
        else:
            cofFrict[count] = -Cf[0]
        A[1:4,count] = np.cross(np.array([1,0,0]),CM-bottomDisp[i,:])+np.cross(np.array([0,0,cofFrict[count]]),bottomDisp[i,:])
        #B[1:4] = B[1:4] - np.dot(np.cross(np.array([-weight,0,0]),CM-bottomDisp[i,:]),identity)
        count = count+1

    #print A
    #print B
    #print out_nodes[bottomNodes[minIndex],:]
    rows = np.array([[0, 0],[2, 2]], dtype=np.intp)
    columns = np.array([[0, 2],[0, 2]], dtype=np.intp)
    reactForce = np.dot(np.linalg.pinv(A[[0,2,3],:]),B[[0,2,3]])
    #print np.dot(np.linalg.pinv(Atest),Btest)
    #print reactForce

    count = 0;
    for i in minIndex:
        reactions[i,0] = reactForce[count]
        reactions[i,2] = cofFrict[count]*reactForce[count]
        count = count +1
    
    minDisp = res_displace[bottomIndex[0][:],0]

    return reactions,minDisp

def writeForceCSV(reactions,force,goal,index):
    np.savetxt('/home/nick/Documents/FERVORSimulationResults/CorrectFriction/2x2x4Front/Force2x2x4ActDist'+str(goal)+'Index'+str(count)+'.csv', np.concatenate((reactions,force*np.ones((reactions.size/3,1))), axis=1), delimiter=',')

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
                                                                                
#Setting up a 2 by 2 by 2
size_x = 2;
size_y = 2;
size_z = 4;

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
#Assuming a Ultem1000 20% Glass Filled tube with material properties taken from:
#https://www.plasticsintl.com/datasheets/ULTEM_GF30.pdf
#Using Newtons, meters, and kilograms as the units
frame_props = {"nu"  : 0.35, #poisson's ratio +
               "d1"	 : 0.00127, #m
               "d2"	 : 0.00127, #m
               "E"   : 6894757000, #N/m^2 +
               "G"   : 9205282000,  #N/m^2 +
               "rho" :  1420, #kg/m^3
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

force = -10.0
weight = 72.2*9.8*1e-3
u = 0.0762
CubeAct = 58.2
MAS = 7
CM = [u,u,(CubeAct+2*MAS*3)*u/(2*MAS+CubeAct)]
print CM
CfRough = 0.009063249338741
CfSmooth = 0.006980062850706
Cf = np.array([CfRough,CfSmooth])
actuationRange = np.concatenate((np.arange(-0.01,0,0.001),np.arange(0.001,0.011,0.001)), axis=0)
actuationRange = np.concatenate((actuationRange,np.flipud(actuationRange)),axis=0)
count = 0
prevNodes = np.zeros((68,6))
prevGoal = -0.01

for goal in actuationRange:
    converge = 1.0
    force = force/2
    if np.sign(goal) != np.sign(prevGoal):
        force = -force
    print force
    print goal

    bottomDisp = np.zeros((8,3))
    reactForce = np.zeros((8,3))
    reactions = np.zeros((8,6))
    prevReactions = np.zeros((8,6))
    for i in range(0,len(reactions)):
        reactions[i,0] = weight/8

    while converge>0.01:
        constraints,loads = applyForce(force,reactions)
        res_displace,out_nodes,global_args,out_frames,C = solveIteration(frames,frame_props,nodes,constraints,loads)

        bottomIndex = np.where(out_nodes[:,0]==min(out_nodes[:,0]))
        reactions,prevMin = calcReactionForces(weight,res_displace,out_nodes,CM,Cf,bottomIndex,prevNodes,reactions)

        reactionTruth = True
        while np.all(reactionTruth) or abs(sum(reactions[:,0])-weight)>0.01*weight:
            constraints,loads = applyForce(force,reactions)
            res_displace,out_nodes,global_args,out_frames,C = solveIteration(frames,frame_props,nodes,constraints,loads)
            
            reactions,minDisp = calcReactionForces(weight,res_displace,out_nodes,CM,Cf,bottomIndex,prevNodes,prevReactions)
            #print reactions
            
            reactionConverge = abs((prevMin-minDisp)/prevMin)
            reactionTruth = np.greater(reactionConverge,0.05*np.ones_like(reactionConverge))
            prevMin = minDisp
            prevReactions = reactions
            #print abs(sum(reactions[:,0])-weight)
            #print "Reaction Converge: " + str(reactionConverge)

        converge = (goal-res_displace[42][0])/goal
        print"Converge: " + str(converge)
        force = force+force*converge
        print force
        converge = abs(converge)
        #pfea.util.pfeautil.plotLattice(nodes,frames,res_displace,1)

    writeForceCSV(reactions,force,goal,count)
    prevNodes = np.zeros((68,6))
    prevNodes[:,0:3] = out_nodes
    prevNodes = prevNodes+res_displace
    #pfea.util.pfeautil.plotLattice(nodes,frames,res_displace,1)
    pfea.util.pfeautil.writeCSV(nodes,res_displace,'/home/nick/Documents/FERVORSimulationResults/CorrectFriction/2x2x4Front/Front2x2x4ActDist'+str(goal)+'Index'+str(count)+'.csv')
    count = count+1
    prevGoal = goal
    
#print(res_displace[25][0])
#print out_frames
pfea.solver.write_K(out_nodes,out_frames,global_args,'/home/nick/Documents/FERVORSimulationResults/CorrectFriction/2x2x4Front/K2x2x4.txt')
pfea.solver.write_M(out_nodes,out_frames,global_args,'/home/nick/Documents/FERVORSimulationResults/CorrectFriction/2x2x4Front/M2x2x4.txt')
np.savetxt('2x2x4FrontFrames.csv', out_frames[0][0], delimiter=',')

M = pfea.solver.provide_M(out_nodes,out_frames,global_args)
K = pfea.solver.provide_K(out_nodes,out_frames,global_args)
