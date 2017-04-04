from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from math import *

def debugplot(nodes,framesets,res_displace,Q,loads,constraints,scale_factor=1,axial_forces=True):
    #
    # Basic setup things
    #
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('PFEA Debug Plot')

    g = gl.GLGridItem()
    w.addItem(g)

    centroid = [0.5*(np.max(nodes.T[i])+np.min(nodes.T[i])) for i in range(3)]

    #
    # node population
    #

    loadids = [a["node"] for a in loads]
    constids = [a["node"] for a in constraints]

    colors = []
    for i in range(len(nodes)):
        if(i in loadids):
            colors.append((0.0,1.0,0.0,1.0))
        elif(i in constids):
            colors.append((1.0,1.0,0.0,1.0))
        else:
            colors.append((1.0,1.0,1.0,0.5))

    displace_nodes = nodes+res_displace*scale_factor
    #sp1 = gl.GLScatterPlotItem(pos=nodes,color=np.array(colors),size=0.1,pxMode=False)#, size=size, color=color, pxMode=False)
    #sp1.translate(-centroid[0],-centroid[1],-centroid[2])

    #w.addItem(sp1)

    sp1d = gl.GLScatterPlotItem(pos=displace_nodes,color=(1.0,0.0,0.0,0.5),size=0.1,pxMode=False)#, size=size, color=color, pxMode=False)
    sp1d.translate(-centroid[0],-centroid[1],-centroid[2])
    w.addItem(sp1d)
    #
    # Calculating strain energy (axial)
    #

    frame_args = framesets[0][2]
    if(axial_forces):
        st_nrg = Q.T[0,:]
    else:
        st_nrg = 0.5*frame_args["Le"]/frame_args["E"]*(Q.T[0,:]**2/frame_args["Ax"]+Q.T[4,:]**2/frame_args["Iy"]+Q.T[5,:]**2/frame_args["Iz"])

    st_nrg = st_nrg.T
    qmax = np.max(abs(st_nrg))
    #print st_nrg

    print qmax
    print np.mean(abs(st_nrg))

    factors = np.zeros((len(st_nrg),2))

    for i, strain in enumerate(st_nrg):
        if strain > 0:
            if abs(strain)/(pi**3*frame_args["E"]*frame_args["d1"]**4/(4*frame_args["Le"]**2)) > 1.0:
                print("**DANGER AT FRAME ID {0}".format(i))
            #print "element {0} safety factor".format(abs(strain)/(pi**3*frame_args["E"]*frame_args["d1"]**4/(4*frame_args["Le"]**2)))
            factors[i][0] = abs(strain)/(pi**3*frame_args["E"]*frame_args["d1"]**4/(4*frame_args["Le"]**2))
        if strain < 0:
            if abs(strain)/(37.5e6*frame_args["d1"]**2) > 1.0:
                print("**DANGER AT FRAME ID {0}".format(i))
            #print "element {0} safety factor".format(abs(strain)/(37.5e6*frame_args["d1"]**2))
            factors[i][1] = abs(strain)/(37.5e6*frame_args["d1"]**2)

    max_ten = np.max(factors.T[1])
    max_com = np.max(factors.T[0])

    plotframes = []
    colors = []

    for i,frame in enumerate(framesets[0][0]):
        nid1 = int(frame[0])
        nid2 = int(frame[1])
        plotframes.append(displace_nodes[nid1])
        plotframes.append(displace_nodes[nid2])

        if(st_nrg[i] < 0 ):
            colors.append((1.0,0.0,0.0,0.5))#(1.0*factors[i][1]/max_ten)**3))
            colors.append((1.0,0.0,0.0,0.5))#(1.0*factors[i][1]/max_ten)**3))
        else:
            colors.append((0.0,0.0,1.0,0.5))#(1.0*factors[i][0]/max_com)**3))
            colors.append((0.0,0.0,1.0,0.5))#(1.0*factors[i][0]/max_com)**3))
    sp2 = gl.GLLinePlotItem(pos=np.array(plotframes),color=np.array(colors),mode='lines',antialias=True)
    sp2.translate(-centroid[0],-centroid[1],-centroid[2])
    w.addItem(sp2)

    ##Had to add this so it wouldn't immediately get garbage collected
    app.exec_()
