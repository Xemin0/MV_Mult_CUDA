'''
Plotting data matrices from .dat as heatmaps

'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


fname = "FLOPrate_base.dat"


if "FLOPrate_base.dat" == fname:
    title_str = "TeraFLOP/s with Varying Matrix Dimensions(log)\n1 thread per row\n for $C = C+AB$"
elif "FLOPrate_single.dat" == fname:
    title_str = "TeraFLOP/s with Varying Matrix Dimensions(log)\n1 block per row\n for $C = C+AB$"
elif "FLOPrate_multi.dat" == fname:
    title_str = "TeraFLOP/s with Varying Matrix Dimensions(log)\nmultiple block per row\n for $C = C+AB$"
else:
    print("File name not yet supported.")

fname = "../data/" + fname


# Read and load as np arrays 
data = np.array([d.strip().split() for d in open(fname).readlines()], dtype = float)

n,m = data.shape

# plot the heat map of FLOPrate w.r.t. dimensions of the matrix
im = plt.matshow(data, cmap = 'autumn')
# add colorbar
cbar = plt.colorbar(im, fraction = 0.046, pad = 0.04)

# add annotations/values
for i in range(n):
    for j in range(m):
        plt.annotate(f'{data[i][j]:.2e}', xy = (j , i),
                         ha = 'center', va = 'center', color = 'black')

plt.title(title_str)

plt.xlabel("$M = 10^{1 + m/2}$")
plt.ylabel("$N = 10^{1 + n/2}$")


plt.show()
