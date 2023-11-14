# Evaluating Performances of Matrix-Vector Multiplication with CUDA

## Folder Structure
#### C++/CUDA Part
- ./MV_GPU/lib/MyUtils.h
- ./MV_GPU/MyUtils.cpp (separate file that include C/C++ utilities)
- ./MV_GPU/lib/MV_GPU.h
- ./MV_GPU/MV_GPU.cu (kernels and kernel launching methods with CUDA based utilities)
- ./MV_GPU/main.cu (the main driver)
- ./MV_GPU/MV_GPU.sh (SLURM script to run on Oscar@CCV with CUDA)

#### Python Part (For Plotting)
- results/python_plot/heatmap.py (Plotting data matrices as heatmaps)
- results/python_plot/RFmodel.py (Plotting Roofline model for the tested GPU)

## Compile and Run
*Only tested with `CUDA11.2.0` and `gcc10.2` on NVIDIA RTX 3090 and 6000*

`-arch sm_86` flag (or higher) specifying the architecture required to use `atomicAdd()` with `cuda>=6.x`


```bash
nvcc -arch sm_86 -c MV_GPU.cu -o MV_GPU.o
nvcc -arch sm_86 -c main.cu -o main.o
g++ -c MyUtils.cpp -o MyUtils.o

# Link everything together
nvcc main.o MV_GPU.o MyUtils.o -o testMV_GPU.o
```

To run locally simply use
`./testMV_GPU.o`

To run on Oscar@CCV use the provided SLURM script
`sbatch ./MV_GPU.sh`

## Usage of the Primary Function in `main.cu`
`eval_FLOPrate(bool base, bool single, unsigned int n = 5)`

This function evaluate the achieved FLOP rate for different sized Matrix in Matrix-Matrix Multiplication of $C = C+ AB$ with three different parallel designs, by taking the average of $n$ ($n = 5$ by default) runs for each configuration excluding the warmup runs (3 by default), with the option to use 
- baseline  : baseline method 1 thread per row
- single    : single thread block per row
- multiple  : multiple thread block per row
The matrices used are randomly generated with entries ranging from -1e6 to 1e6 for their sizes variying from $O(10)$ to $O(10^4)$

Input:
- base  : whether to use baseline method
- single: whether to use single block or multiple block method
Output:
- elapsed_time: in Microsecond
- FLOP rate: in TeraFLOP/s


**matrix of elapsed time in `./data/e_us*.dat` while matrix of FLOPrate in `./data/FLOPrate*.dat`**

## To-dos
- Verify the correctness of current three kernels for Matrix-Vector Multiplication
- Finish the kernel leveraging 'Instruction Level Parallelism' 
- Add ErrorCheck for all CUDA function calls
- Test performances for larger matrix/vector sizes
- Try more WARP Level Primitives
- Incorporate `OpenMP` (Shared Memory) or `MPI` (Distributed Memory)


## Summary and Comments on the Methodologies
CUDA programming in general is constituted by following steps:
1. Check CUDA devices and environments
2. Prepare data on Host 
3. Declare variables and allocate memory on Device
4. Copy data from Host to Device
5. Launch kernels to perform computations on Device
6. Copy data from Device back to Host
7. Free up memory on both Device and Host

The key step lies in designing the kernels for the problem of interest, and thus we have separate the kernels and the corresponding kernel launching methods in 'MV_GPU.cu' for readability. 

There are conceptually two major steps in defining a kernel for Matrix-Vector Multiplication:
- Frobenius Multiplication
- Reduced Summation

Both steps invite an amalgam of parallelizing techniques in CUDA.

In **Frobenius Multiplication** step: matrix can be partitioned in various ways (block and warps level) for parallelism while as the vector is repeatedly accessed in Matrix-Vector Multiplication (used for each row of the matrix), it can be loaded into some predefined shared memory local to each block or shared inside a block by `__shfl_sync()` method

In **Reduced Summation** step: since GPU executes instructions on the Warp level, and given the visible scope of data on Block level, we could also leverage the advantages of Warp level primitives provided in CUDA and that of shared memory or shuffle methods.

In **Multiple Block Method** we adopted **Two-Phase Reduction** for matrix-vector multiplication, where two consecutive kernels - one for Frobenius Multiplication along with `atomicAdd` for partial summation within a block, the other for Reduced Summation using `__shfl_down_sync()` - are deployed, instead of using nested kernel calls which is only supported in newer GPU architectures.

Besides, **Instruction Level Parallelism** (for ILP = 2) is also considered.

For didactic and learning purpose of this project, we would thus like to implement and test different combinations of aforementioned techniques, instead of directly comparing any two of them in a more rigorous setting.

Considering the fact that Warp level operations are mostly abstracted by GPU, we simply set `BLOCKSIZE` to be equal to `WARPSIZE`, in the hope to imitate and explore the handlings of warps (as blocks).


In summary, we provide three major kernels to compute Matrix-Vector Multiplication on GPU with `BLOCKSIZE` = `WARPSIZE`  = 32

- Baseline Method assigning 1 thread for each row of matrix (corresponds to each element in the resulting vector)
- Single Block method assigning 1 block for each row 
- Multiple Block method assigning multiple block for each row with Two-Phase Reduction
