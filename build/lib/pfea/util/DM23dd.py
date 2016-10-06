####################################################################
####################################################################
## The 'font' is Banner from http://www.network-science.de/ascii/ ##
####################################################################
####################################################################


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import frame3dd
import subprocess
import pfea
import cProfile
import cuboct
from math import *


## Python Script for Converting DM files to .3dd Files

      #  #####  ####### #     #    ### #     # ######  ####### ######  ####### 
      # #     # #     # ##    #     #  ##   ## #     # #     # #     #    #    
      # #       #     # # #   #     #  # # # # #     # #     # #     #    #    
      #  #####  #     # #  #  #     #  #  #  # ######  #     # ######     #    
#     #       # #     # #   # #     #  #     # #       #     # #   #      #    
#     # #     # #     # #    ##     #  #     # #       #     # #    #     #    
 #####   #####  ####### #     #    ### #     # #       ####### #     #    #    

                                                                              
import json
from pprint import pprint

def mat_matrix_from_json(filename):
	with open(filename) as data_file:    
	    data = json.load(data_file)

	size_x = data["assembly"]["cellsMax"]["x"]-data["assembly"]["cellsMin"]["x"]+3
	size_y = data["assembly"]["cellsMax"]["y"]-data["assembly"]["cellsMin"]["y"]+3
	size_z = data["assembly"]["cellsMax"]["z"]-data["assembly"]["cellsMin"]["z"]+3

	mat_matrix = np.zeros((size_x,size_y,size_z))

	#print(data["assembly"]["sparseCells"])

	for x,row in enumerate(data["assembly"]["sparseCells"]):
		for y,col in enumerate(row):
			for z,dep in enumerate(col):
				if dep != None:
					mat_matrix[x+1][y+1][z+1] = 1

	return mat_matrix



