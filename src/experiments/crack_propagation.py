import matplotlib.pyplot as plt


if global_args["debug_plot"]:
	### Right now the debug plot only does x-y-z displacements, no twisting
	xs = []
	ys = []
	zs = []

	dxs = []
	dys = []
	dzs = []

	rxs = []
	rys = []
	rzs = []

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_aspect('equal')
	frame_coords = []

	for i,node in enumerate(nodes):
		xs.append(node[0])
		ys.append(node[1])
		zs.append(node[2])
		dxs.append(node[0]+def_displace[i][0])
		dys.append(node[1]+def_displace[i][1])
		dzs.append(node[2]+def_displace[i][2])
		rxs.append(node[0]+res_displace[i][0])
		rys.append(node[1]+res_displace[i][1])
		rzs.append(node[2]+res_displace[i][2])

	intensity = np.absolute(def_displace-res_displace)
	max_intense = np.amax(intensity)

	for i,frame in enumerate(dframes):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		frame_intensity = (0.5*(intensity[nid1][0]+intensity[nid2][0])/max_intense+0.5*(intensity[nid1][1]+intensity[nid2][1])/max_intense+0.5*(intensity[nid1][1]+intensity[nid2][1])/max_intense)/3.0
		#frame_intensity = frame_intensity**2
		if frame_intensity > 0.1:
			ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=frame_intensity)
	'''
	for dframe in dframes:
		nid1 = int(dframe[0])
		nid2 = int(dframe[1])
		dstart = [dxs[nid1],rys[nid1],rzs[nid1]]
		dend   = [dxs[nid2],rys[nid2],rzs[nid2]]
		ax.plot([dstart[0],dend[0]],[dstart[1],dend[1]],[dstart[2],dend[2]],color='b', alpha=0.1)
	'''	

	#ax.scatter(xs,ys,zs, color='r',alpha=0.1)
	#ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
	plt.show()
	#print(frames)
