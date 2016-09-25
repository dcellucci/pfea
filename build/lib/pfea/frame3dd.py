#!/usr/bin/env python
from __future__ import division
from numpy import *
import csv
import datetime
import sys
#from pfeautil import magnitudes

def read_lowest_mode(filename):
    freq = -1
    with open(filename+'.out', "r") as f:
        searchlines = f.readlines()
        f.close()
    for i,line in enumerate(searchlines):
        if "* The Stiffness Matrix is not positive-definite *\n" in line:
            return -1 #return -1 if the result can't be trusted
        elif "M A S S   N O R M A L I Z E D   M O D E   S H A P E S \n" in line:
            return float(searchlines[i+2].split(' ')[11])

def compute_mass(nodes,beam_sets):
    '''
    For sets of beams, compute the mass
    '''
    m = []
    for beams,args in beam_sets:
        if args['cross_section']=='circular':
            l = magnitudes(nodes[beams[...,1]]-nodes[beams[...,0]])
            Ro = .5*args['d1']
            a = pi*(Ro**2 - (Ro-args['th'])**2)
            m.append(sum(l*a*args['rho']))
        elif args['cross_section']=='rectangular':
            l = magnitudes(nodes[beams[...,1]]-nodes[beams[...,0]])
            a = args['d1']; b = args['d2']
            m.append(sum(l*a*b*args['rho']))            
        else:
            raise(NotImplementedError)
    return m

def write_frame3dd_file(nodes,global_args,beam_sets,constraints,loads):
    '''
    Write an input file with nodes, beams, constraints, and loads.
    By convention, units are N,m,kg

    nodes is a numpy array of floats of shape (-1,3) specifying spatial location of nodes
    global_args contains information about the whole problem:
        node_radius is a rigid node radius
        n_modes is the number of dynamic modes to compute
        length_scaling is a spatial scale for numerical purposes.  All lengths are multiplied, other units adjusted appropriately.
        frame3dd_filename is the name of the CSV input file for frame3dd that is written.
    beam_sets is a list of tuples beams,args.  
        beams is a numpy array of ints of shape (-1,2) referencing into nodes
        args is a dictionary containing properties of the corresponding beam_set.  It must contain
            'E': young's modulus
            'nu': poisson ratio
            'rho':density
            'dx': x dimension of cross section
            'dy': y dimension of cross section
    constraints and loads are both lists of dictionaries with keys ['node','DOF','value']
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
        node_radius=zeros(shape(nodes)[0])
    filename = global_args['frame3dd_filename']
    n_beams = sum(map(lambda x: shape(x[0])[0], beam_sets))    
    nodes = nodes*length_scaling

    #aggregate constraints and loads by node
    full_constraints = zeros((shape(nodes)[0],6))
    full_displacements = zeros((shape(nodes)[0],6))
    full_loads = zeros((shape(nodes)[0],6))
    for c in constraints:
        full_constraints[c['node'],c['DOF']] = 1
        full_displacements[c['node'],c['DOF']] = c['value']*length_scaling
    full_displacements *= length_scaling
    constrained_nodes = where(any(full_constraints!=0, axis=1))[0]
    displaced_nodes = where(any(full_displacements!=0, axis=1))[0]
    for l in loads:
        full_loads[l['node'],l['DOF']] = l['value']

    loaded_nodes = where(any(full_loads!=0, axis=1))[0]

    #beams is flattened beam_sets
    beams = vstack([beams for beams,args in beam_sets])
    beam_division_array = hstack([args['beam_divisions']*ones(shape(bs)[0],dtype=int) for bs,args in beam_sets])
    #print(shape(beams)[0])
    #print(shape(beam_division_array)[0])
    assert(shape(beams)[0]==shape(beam_division_array)[0])
    #subdivide elements
    #add new_nodes after existing nodes
    #create a mapping from old elements to new elements, so can apply loads
    #beam_mapping = arange(shape(beams)[0])[...,None] #what new beams each original beam maps to
    if any(beam_division_array > 1):
        new_beams = []
        new_nodes = []
        beam_mapping = []
        cur_node = shape(nodes)[0] #start new nodes after existing
        cur_beam = 0
        for j,b in enumerate(beams):
            this_map = []
            bdj = beam_division_array[j]
            if bdj==1:
                new_beams.append(b)
                this_map.append(cur_beam); cur_beam += 1
            else:
                for i in range(bdj):
                    t0 = i/bdj
                    t1 = (i+1)/bdj
                    x0 = nodes[b[0]]*(1-t0) + nodes[b[1]]*t0
                    x1 = nodes[b[0]]*(1-t1) + nodes[b[1]]*t1
                    if i==0:
                        new_beams.append([b[0],cur_node])
                        this_map.append(cur_beam); cur_beam += 1
                    elif i==bdj-1:
                        new_nodes.append(x0);   
                        new_beams.append([cur_node,b[1]])
                        this_map.append(cur_beam); cur_beam += 1
                        cur_node += 1
                    else:
                        new_nodes.append(x0)
                        new_beams.append([cur_node,cur_node+1])
                        this_map.append(cur_beam); cur_beam += 1
                        cur_node += 1
            beam_mapping.append(this_map)

        if len(new_nodes)>0:
            nodes = vstack((nodes,asarray(new_nodes)))
            node_radius = hstack((node_radius,zeros(len(new_nodes))))
            n_beams += shape(new_nodes)[0]
        beams = asarray(new_beams)
        beam_mapping = asarray(beam_mapping)
    else:
        beam_mapping = arange(n_beams).reshape(-1,1)
    with open(filename+'.csv', 'wb') as csvfile:
        f = csv.writer(csvfile, delimiter=',')
        def write_row(items):
            #write items, including commas out to 13 spots
            base = [""]*13
            for i,v in enumerate(items): base[i] = v
            f.writerow(base)
        write_row(["Input data file for Frame3dd (N, m/%.3f, kg) "%(length_scaling)+"%s"%datetime.datetime.today()])
        write_row([])
        write_row([shape(nodes)[0],"","#number of nodes"])
        write_row([])
        for i,n in enumerate(nodes):
            write_row([i+1,n[0],n[1],n[2],node_radius[i]]) #node num, x, y, z, r
        write_row([])
        write_row([shape(constrained_nodes)[0],"","#number of nodes with reactions"])
        write_row([])
        for cn in constrained_nodes:
            write_row(map(int,[cn+1]+list(full_constraints[cn])) )
        write_row([])
        write_row([n_beams,"","#number of frame elements"])
        write_row(["#.e","n1","n2","Ax","Asy","Asz","Jxx","Iyy","Izz","E","G","roll","density"])
        write_row([])
        beam_num_os = 0
        for beamset,args in beam_sets:
            E = args['E']/length_scaling/length_scaling
            nu = args['nu']
            d1 = args['d1']*length_scaling
            roll = args['roll']
            if args['cross_section']=='circular':
                Ro = .5*d1
                th = args['th']*length_scaling
                assert(0<th<=Ro)
                Ri = Ro-th
                Ax = pi*(Ro**2-Ri**2)
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
                Q = .33333 - 0.2244/(max(d1,d2)/min(d1,d2) + 0.1607); 
                Jxx = Q*max(d1,d2)*min(d1,d2)**3 
            G = E/2./(1+nu)
            rho = args['rho']/length_scaling/length_scaling/length_scaling
            for i,b in enumerate(beamset):
                for bi in beam_mapping[i+beam_num_os]:
                    write_row([bi+1,int(beams[bi,0]+1),int(beams[bi,1]+1), Ax, Asy, Asz, Jxx, Iyy, Izz, E, G, roll, rho])
            beam_num_os += shape(beamset)[0]
        write_row([])
        write_row([])
        write_row([1,"","#whether to include shear deformation"])
        write_row([1,"","#whether to include geometric stiffness"])
        write_row([1,"","#exagerrate static mesh deformations"])
        write_row([2.5,"","#zoom scale for 3d plotting"])
        write_row([1.,"","#x axis increment for internal forces"])
        write_row([])
        write_row([1,"","#number of static load cases"])
        write_row([])
        write_row(["#Gravitational loading"])
        write_row([0,0,0])
        write_row([])
        write_row(["#Point Loading"])
        write_row([shape(loaded_nodes)[0],"","#number of nodes with point loads"])
        write_row([])
        for ln in loaded_nodes:
            write_row(["%d"%(ln+1)]+list(full_loads[ln]) )
        write_row([])

        try:
            write_row([sum([shape(args['loads'])[0]*args['beam_divisions'] for beams,args in beam_sets]),"","#number of uniformly distributed element loads"])
            beam_num_os = 0
            for beams,args in beam_sets:
                for el in args['loads']:
                    for new_el in beam_mapping[el['element']+beam_num_os]:
                        write_row([ new_el+1]+list(el['value']/length_scaling)) #force/length needs scaling
                beam_num_os += shape(beams)[0]
        except(IndexError):
            write_row([0,"","#number of uniformly distributed element loads"])

        write_row([0,"","#number of trapezoidal loads"])
        write_row([0,"","#number of internal concentrated loads"])
        try:
            write_row([sum([shape(args['prestresses'])[0]*args['beam_divisions'] for beams,args in beam_sets]),"","#number of temperature loads"])
            beam_num_os = 0
            for beams,args in beam_sets:
                for el in args['prestresses']:
                    C=1; #proxy for coefficient of thermal expansion
                    for new_el in beam_mapping[el['element']+beam_num_os]:
                        cool = -el['value']/(C*args['E']*args['d1']*args['d2'])*length_scaling
                        write_row([new_el+1,C,args['d1']*length_scaling,args['d2']*length_scaling,cool,cool,cool,cool])
                beam_num_os += shape(beams)[0]
        except(KeyError,IndexError):
            write_row([0,"","#number of temperature loads"])


        write_row([])

        write_row([shape(displaced_nodes)[0],"","#number of nodes with prescribed displacements"])
        for dn in displaced_nodes:
            write_row([dn+1]+list(full_displacements[dn]) )

        write_row([])
        write_row([n_modes,"","#number of dynamic modes"])
        if n_modes != 0:
            write_row([1,"","#1= Subspace-Nacobi iteration, 2= Stodola (matrix iteration) method"])
            write_row([0,"","#0= consistent mass matrix, 1= lumped mass matrix"])
            write_row([.0001,"","#frequency convergence tolerance  approx 1e-4"])
            write_row([0.,"","#frequency shift-factor for rigid body modes, make 0 for pos.def. [K]"])
            write_row([.5,"","#exaggerate modal mesh deformations"])

            write_row([0,"","#number of nodes with extra node mass or rotary inertia"])
            write_row([0,"","#number of frame elements with extra node mass"])
            write_row([4,"","#number of modes to be animated"])
            write_row([1,2,3,4,"","#list of modes to be animated, increasing"])
            write_row([1.,"","#pan rate of the animation"])
            write_row([])
            write_row([1,"# matrix condensation method :0=none, 1=static, 2=Guyan, 3=dynamic"])
        write_row(["#End input"])

def read_frame3dd_results(filename):
    f = open(filename+'.out','r')
    reading=False
    nodes = []
    reactions = []
    for l in f.readlines():
        if 'R E A C T I O N S' in l:
            reading = True; continue
        if 'R M S    R E L A T I V E    E Q U I L I B R I U M    E R R O R' in l:
            reading = False; continue
        if reading:
            items = [item for item in l.strip('\n').split(' ') if item != '']
            try:
                nodes.append(int(items[0])-1) #undo 1 indexing from frame3dd
                reactions.append(map(float,items[1:]))
            except(ValueError):
                continue
    return asarray(nodes),asarray(reactions)

def read_frame3dd_displacements(filename):
    f = open(filename+'_out.CSV','r')
    reading=False
    displacements = []
    for l in f.readlines():
        if 'N O D E   D I S P L A C E M E N T S    (global)' in l:
            reading = True; continue
        if 'F R A M E   E L E M E N T   E N D   F O R C E S  (local)' in l:
            reading = False; continue
        if reading:
            items = [item.strip(',') for item in l.strip('\n').split(' ') if item != '']
            try:
                displacements.append(map(float,items[1:]))
            except(ValueError):
                continue
    return asarray(displacements)
