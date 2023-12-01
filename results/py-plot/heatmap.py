'''
Plotting data matrices from .dat as heatmaps

'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


numBlocks = 1

fname = "e_us" + str(numBlocks) + "_cpu.dat"

fname = "../data/" + fname

title_str = f"Total Time (milisecond, ms) for Matrix-Vector Multiplication with\n {numBlocks} Streams in CUDA"


# Read and load as np arrays
# Convert from microsecond to milisecond
data = np.array([d.strip().split() for d in open(fname).readlines()], dtype = float) / 1000.0

n,m = data.shape

# plot the heat map of FLOPrate w.r.t. dimensions of the matrix
im = plt.matshow(data, cmap = 'autumn')
# add colorbar
cbar = plt.colorbar(im, fraction = 0.046, pad = 0.04)

# add annotations/values
for i in range(n):
    for j in range(m):
        plt.annotate(f'{data[i][j]:.2g}', xy = (j , i),
                         ha = 'center', va = 'center', color = 'black')

plt.title(title_str)

plt.xlabel("$M = 1000 + 200m$")
plt.ylabel("$N = 1000 + 200n$")


plt.show()
