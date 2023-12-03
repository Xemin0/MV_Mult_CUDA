# Evaluating Performances of Matrix-Vector Multiplication with CUDA Streams

### IMPORTANT NOTES ON COMPILING FLAG
**This is important to include the architecture specification when compiling on a cluster!**
- `-arch sm_86` targets the compute capability 8.6 which is for `Ampere` architecture
- For `Turing` architecture, consider the lower of the config from GPU used (e.g. `sm_75` for RTX6000)
- (Optional) It is possible to do cross-compiling on a cluster for multiple architectures

```bash
# Example for Cross Compiling
nvcc -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -c *.cu -o *.o
```

### IMPORTANT NOTES ON MEMORY ALLOCATORS (on HOST)
Since CUDA stream's asynchronous operations require pinned or non-pageable memory allocation on HOST, therefore `cudaMalloc` will not work. Instead, `cudaHostAlloc` or `cudaHostRegister` should be used to allocate HOST memory, before transfering data from HOST to GPU


## Objective of the Practice Project
Evaluating the performance of $y = Ax$ with:
- $K$ Streams for $K = 1, 2, \dots, 8$
- The dimensions of the matrix $A$: $M$-by-$N$ for $M,N = 1000 + 200i,\quad i = 0, 1, \dots, 5$

and the time is measured for the following steps:
- For each `streams[i]` copy a submatrix `A_sub[i]` to GPU
- Launch the kernel on each stream
- For each `streams[i]` copy the corresponding (partial) result of $y$ back to CPU

Data Results (CPU time) `.dat` files without `_cpu` or `_gpu` are obtained on RTX3090 with CUDA11.2.0

The others are obtained on RTX6000 with CUDA11.2.0 (The current results are taken separately. They thus may not be comparable)


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


```bash
nvcc -arch sm_75 -c MV_GPU.cu -o MV_GPU.o
nvcc -arch sm_75 -c main.cu -o main.o
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

## TO-DOs
- Combine the `gpuTimer` and `cpuTimer` into one, so that the GPU time and CPU time measurements are comparable
- ~Reduce the number of calls for `atomicAdd` in the kernel; instead should do Manual Reduction before writing the final result (the corresponding entry) with a single `atomicAdd`~
- Provide subroutines to finalize the pipeline of Matrix-Vector Multiplication with CUDA stream
- Verify the correctness of the designed kernels
- Add `CheckCUDA` method used to check the status of any CUDA API calls, for better debugging 

## Methodology 
As in the kernel `MV_KBlocks_kernel` and the kernel launching method `MV_KBlocks` in `MV_GPU.cu`

Since we are using $K$ streams/blocks to take care of the Matrix-Vector Multiplication of $$y = Ax$$, each block will potentailly computes for multiple rows of the matrix $A$.
- All memories on HOST for the input (matrix, vector) and the output (the resulting vector of the product) are allocated with `cudaHostAlloc` (or `cudaHostRegister`) required for asynchronous operations.
- Matrix $A$ is divided into $K$ submatrices and copy to GPU by each stream
- Each block/stream compute rows separately.
- Inside the kernel the vector's entries in the MV-Multiplication are shared within the BLOCK (`BLOCK_SIZE` = `WARP_SIZE`) using the warp-level primitive `__shfl_sync()`, instead of using a shared memory
- Manual reduction over tiled summations (kept in a shared memory within a block for each thread) from each thread within a Warp
- Atomically add the reduced summations to the output vector

## Visualizing the Asynchronous Streams in NVIDIA NSIGHT
The asynchronous executions by CUDA streams are observed
![Stair-wise Asynchronous Executions](results/pix/stream0.png?raw=true "Title") 
