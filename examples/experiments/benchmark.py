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
from timeit import default_timer as timer

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

maxDeminsion = 10
#create numpy array for times
times = np.zeros((pow(maxDeminsion-2,3),4))

#Physical Voxel Properties
#Current voxel pitch is 3in
vox_pitch = 0.0762 #m

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

for size_x in range(2,maxDeminsion):
    for size_y in range(2,maxDeminsion):
        for size_z in range(2,maxDeminsion):

            start = timer()

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
    
            #Node Map Population
            #Referencing the geometry-specific cuboct.py file. 
            #Future versions might have different files?

            node_frame_map = np.zeros((size_x,size_y,size_z,6))
            nodes,frames,node_frame_map,dims = pfea.geom.cuboct.from_material(mat_matrix,vox_pitch)

            num_nodes = size_x*size_y*2 + size_x + size_y

            node_load = (size_z-1)*vox_pitch*0.01

            #Constraint and load population
            constraints = []
            loads = []

            for x in range(1,size_x+1):
                    for y in range(1,size_y+1):
                            #The bottom-most nodes are constrained to neither translate nor
                            #rotate
                            constraints.append({'node':node_frame_map[x][y][1][0],'DOF':0, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][0],'DOF':1, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][0],'DOF':2, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][0],'DOF':3, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][0],'DOF':4, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][0],'DOF':5, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][1],'DOF':0, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][1],'DOF':1, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][1],'DOF':2, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][1],'DOF':3, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][1],'DOF':4, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][1][1],'DOF':5, 'value':0})


                            constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':0, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':1, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':3, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':4, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':5, 'value':0})

                            constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':0, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':1, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':3, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':4, 'value':0})
                            constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':5, 'value':0})

                            #The top most nodes are assigned a z-axis load, as well as being
                            #constrained to translate in only the z-direction.
                            #loads.append(      {'type':'disp','node':node_frame_map[x][y][size_z][0],'DOF':2, 'value':node_load})
                            #loads.append(      {'type':'disp','node':node_frame_map[x][y][size_z][1],'DOF':2, 'value':node_load})

                            if x == size_x:
                                    constraints.append({'node':node_frame_map[x][y][1][3],'DOF':0, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][1][3],'DOF':1, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][1][3],'DOF':2, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][1][3],'DOF':3, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][1][3],'DOF':4, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][1][3],'DOF':5, 'value':0})

                                    constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':0, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':1, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':3, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':4, 'value':0})
                                    constraints.append({'node':node_frame_map[x][y][size_z][3],'DOF':5, 'value':0})
                                    
                                    loads.append(      {'type':'disp','node':node_frame_map[x][y][size_z][3],'DOF':2, 'value':node_load})


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
            global_args = {'frame3dd_filename': 'test',"lump": False, 'length_scaling':1,"using_Frame3dd":True,"debug_plot":True, "gravity" : [0,0,0],"save_matrices":True}

            if global_args["using_Frame3dd"]:
                    pfea.frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
                    subprocess.call("frame3dd -i {0}.csv -o {0}.out -q".format(global_args["frame3dd_filename"]), shell=True)
                    res_nodes, res_reactions = pfea.frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
                    res_displace = pfea.frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])
            else:
                    res_displace,C,Q = pfea.solver.analyze_System(out_nodes, global_args, out_frames, constraints,loads)

            end = timer()
            times[[size_z-2+(size_y-2)*(maxDeminsion-2)+(size_x-2)*pow(maxDeminsion-2,2)],0] = size_x
            times[[size_z-2+(size_y-2)*(maxDeminsion-2)+(size_x-2)*pow(maxDeminsion-2,2)],1] = size_y
            times[[size_z-2+(size_y-2)*(maxDeminsion-2)+(size_x-2)*pow(maxDeminsion-2,2)],2] = size_z
            times[[size_z-2+(size_y-2)*(maxDeminsion-2)+(size_x-2)*pow(maxDeminsion-2,2)],3] = end-start
np.savetxt('timingTest.csv', times, delimiter=',')
