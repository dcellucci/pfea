import matplotlib.pyplot as plt
import matplotlib
import numpy as np

subs = np.array([2,3,4,5,6,7,8])
mods = np.array([-315209395.683,-301921878.0,-299640809.026,-299879091.929,-300367739.003,-300882856.081,-301328159.259,])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(subs,mods*-1)
plt.show()