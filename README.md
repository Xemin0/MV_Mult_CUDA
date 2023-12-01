# Evaluating Performances of Matrix-Vector Multiplication with CUDA Streams
Evaluating the performance of $y = Ax$ with $K$ Streams for $K = 1, 2, \dots, 8$


## Folder Structure
#### C++/CUDA Part
- `./MV_GPU/lib/MyUtils.h`
- `./MV_GPU/MyUtils.cpp` (separate file that include C/C++ utilities)
- `./MV_GPU/lib/MV_GPU.h`
- `./MV_GPU/MV_GPU.cu` (kernels and kernel launching methods with CUDA based utilities)
- `./MV_GPU/main.cu` (the main driver)
- `./MV_GPU/MV_GPU.sh` (SLURM script to run on Oscar@CCV with CUDA)

#### Python Part (For Plotting)
- `results/python_plot/heatmap.py` (Plotting data matrices as heatmaps)
- `results/python_plot/RFmodel.py` (Plotting Roofline model for the tested GPU)

#### Miscellaneous Folders
- `./MV_GPU/data/` **REQUIRED** for output files of performance measurements
- `./MV_GPU/Results/` **REQUIRED** if using `SLURM` script 

## Compile and Run
*Only tested with `CUDA11.2.0` and `gcc10.2` on NVIDIA RTX 3090 and 6000*

`-arch sm_86` flag (or higher) specifying the architecture required to use `atomicAdd()` with `cuda>=6.x`


```bash
nvcc -arch sm_86 -c MV_GPU.cu -o MV_GPU.o
nvcc -arch sm_86 -c main.cu -o main.o
g++ -c MyUtils.cpp -o MyUtils.o

# Link everything together
nvcc main.o MV_GPU.o MyUtils.o -o testStreams.out
```

To run locally simply use
`./testStreams.out`

To profile with CUDA NSight
`nsys profile ./testStreams.out`

To run on Oscar@CCV use the provided SLURM script
`sbatch ./MV_GPU.sh`

## Methodology 
As in the kernel `MV_KBlocks_kernel` and the kernel launching method `MV_KBlocks` in `MV_GPU.cu`

Since we are using $K$ streams/blocks to take care of the Matrix-Vector Multiplication of $$y = Ax$$, each block will potentailly computes for multiple rows of the matrix $A$.
- Vector $x$ is allocated by `cudaHostAlloc` as pinned memory on HOST accessible by both the HOST and the DEVICE
- Matrix $A$ is divided into $K$ submatrices and copy to GPU by each stream
- Each block/stream compute rows separately.
- The results are atomically added to each entry of $y$ for each row
