# Evaluating Performances of Matrix-Vector Multiplication with CUDA
Primarily focus on the performance (in the form of elapsed time on GPU with achieved FLOP rate) of different methods on $C = C + Ax$, with basic memory allocation methods, change the main drivier `main.cu` file for different memory allocators

*For* $y = Ax$ *refer to the end of each kernel defined in* `MV_GPU.cu` for the small modifications

### IMPORTANT NOTES ON COMPILING FLAG
This is important to include the architecture specification when compiling on a cluster!
- `-arch sm_86` targets the compute capability 8.6 which is for `Ampere` architecture
- For `Turing` architecture, consider the lower of the config from GPU used (e.g. `sm_75` for RTX6000)
- (Optional) It is possible to do cross-compiling on a cluster for multiple architectures

```bash
# Example for Cross Compiling
nvcc -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -c *.cu -o *.o
```

### Matrix Multiplication with CUDA Stream
Project codes and some results are in branch `withCUDAstream`.

Swtich to that branch by `git checkout withCUDAstream`

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
nvcc main.o MV_GPU.o MyUtils.o -o testMV_GPU.o
```

To run locally simply use
`./testMV_GPU.o`

To run on Oscar@CCV use the provided SLURM script
`sbatch ./MV_GPU.sh`

## Usage of the Primary Function in `main.cu`
`eval_FLOPrate(bool base, bool single, unsigned int n = 5)`

This function evaluate the achieved FLOP rate for different sized Matrix in Matrix-Matrix Multiplication of $$C = C+ Ax$$ (**Remove the + sign for $y=Ax$ in respective kernels in `MV_Mult.cu`**) 

with three different parallel designs, by taking the average of $n$ ($n = 5$ by default) runs for each configuration excluding the warmup runs (3 by default), with the option to use 
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
- ~Ensure that `eval_MV_Mult()` calls in `main.cu` does not return negative values~
- Fix the issue with using `BLOCK_SIZE` and `blockDim.x` interchangeably in kernels
- Fix the logic of the random generator used (Seeding)
- Add a subroutine to compute Matrix-Vector Multiplication and return the resulting vector
- Add a subroutine to convert a matrix between 2-D and row-major/column-major representations
- Finish the kernel with 'Instruction Level Parallelism' 
- Prevent idling threads/blocks maybe by doing reduction on HOST
- Adding CUDA streams
- Test with advanced memory allocators
- Devise a MakeFile
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
- Hadamard Multiplication
- Reduced Summation

Both steps invite an amalgam of parallelizing techniques in CUDA.

In **Hadamard Multiplication** step: matrix can be partitioned in various ways (block and warps level) for parallelism while as the vector is repeatedly accessed in Matrix-Vector Multiplication (used for each row of the matrix), it can be loaded into some predefined (**Memory Size Cannot be a Runtime Variable, Must be a Constant; unless using the `extern` keyword and manually pass in the memory size when the kernel is launched**) shared memory local to each block or shared inside a block by `__shfl_sync()` method

In **Reduced Summation** step: since GPU executes instructions on the Warp level, and given the visible scope of data on Block level, we could also leverage the advantages of Warp level primitives provided in CUDA and that of shared memory or shuffle methods.

In **Multiple Block Method** we adopted **Two-Phase Reduction** for matrix-vector multiplication, where two consecutive kernels - one for Hadamard Multiplication along with `atomicAdd` for partial summation within a block, the other for Reduced Summation using `__shfl_down_sync()` - are deployed, instead of using nested kernel calls which is only supported in newer GPU architectures.

Besides, **Instruction Level Parallelism** (for ILP = 2) is also considered.

For didactic and learning purpose of this project, we would thus like to implement and test different combinations of aforementioned techniques, instead of directly comparing any two of them in a more rigorous setting.

Considering the fact that Warp level operations are mostly abstracted by GPU, we simply set `BLOCKSIZE` to be equal to `WARPSIZE`, in the hope to imitate and explore the handlings of warps (as blocks).


In summary, we provide three major kernels to compute Matrix-Vector Multiplication on GPU with `BLOCKSIZE` = `WARPSIZE`  = 32

- Baseline Method assigning 1 thread for each row of matrix (corresponds to each element in the resulting vector)
- Single Block method assigning 1 block for each row 
- Multiple Block method assigning multiple block for each row with Two-Phase Reduction
