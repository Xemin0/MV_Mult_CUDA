/*
 * MV_GPU.cu
 *
 *  Created on: Nov 8, 2023
 *
 * Description: Definitions of Kernels for Matrix-Vector Multiplication on GPU
 * 				along with the corresponding Kernel Launching methods
 * 
 *              Matrix-Vector Multiplication C = C + Ax
 *
 *              C : N dimensional vector
 *              A : N by M matrix flattened into a (N*M) vector, 
 *                  Row-Major order for contiguous memory access by threads
 *              x : M dimensional vector
 *
 *
 * ******* for y = Ax, ********
 * ******* check the summation rule commented at the end of each kernel ***
 *
 * ## Current Issue: BLOCK_SIZE and blockIdx.x are used interchangeably ##
 */


#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "./lib/MyUtils.h"

// Block size to conincide with Warp size for simplicity
#define BLOCK_SIZE 32
#define FULL_MASK 0xffffffff

struct gpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    gpuTimer()
    {   
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }   

    ~gpuTimer()
    {   
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }   

    void Start()
    {   
        cudaEventRecord(start, 0); 
    }   

    void Stop()
    {   
        cudaEventRecord(stop, 0); 
    }   

    float Elapsed() // millisecond (ms)
    {   
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }   
};




__global__ void MV_base_kernel(
        double *d_A,
        double *d_x,
        double *d_C,
        unsigned int n_rows,
        unsigned int m_cols){
    /*  
     * Baseline Kernel:
     * each thread handles an entire row of matrix
     * while loading elements of the vector is evenly distributed for each thread
     * Shuflle method instead of shared memory for summation
     */
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;

    double x_shfl_src, x_shfl_dest;

    double y_val = 0.0;

    #pragma unroll
    for (int n_blocks = 0; n_blocks < ((m_cols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++n_blocks)
    {
        // load some element of d_x per thread 
        if ((n_blocks * BLOCK_SIZE + threadIdx.x) < m_cols)
            x_shfl_src = d_x[threadIdx.x + n_blocks * BLOCK_SIZE];
        else
            x_shfl_src = 0.0f;

        __syncthreads();

        // accumulate sum for each row with Warp Level Primitive __shfl
        // Thus BLOCK_SIZE = WARP_SIZE in this case
        #pragma unroll
        for (int e = 0; e < BLOCK_SIZE; ++e)
        {
            x_shfl_dest = __shfl_sync(FULL_MASK, x_shfl_src, e);
            y_val += d_A[row * m_cols + (e + BLOCK_SIZE * n_blocks)] * x_shfl_dest;
        }

        __syncthreads();
    }

    if (row < n_rows) d_C[row] += y_val; // ### remove + if eval y = Ax
}

void MV_base(
        double *d_A,
        double *d_x,
        double *d_C,
        unsigned int n_rows,
        unsigned int m_cols){
    /*
     * Corresponding Kernel Launching method
     * Since each row is taken care of by exactly one thread
     * The total number of threads should be exactly n_rows
     */
    dim3 nthreads(BLOCK_SIZE, 1, 1);    // BlockDim
    dim3 nblocks((n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);         // GridDim

    MV_base_kernel<<<nblocks, nthreads, 0, 0>>>(d_A, d_x, d_C, n_rows, m_cols);
}


//==================

__global__ void MV_single_kernel(
		double *d_A,
		double *d_x,
		double *d_C,
		unsigned int n_rows,
		unsigned int m_cols){
	/*
	 * Single warp for each matrix row
     * Using shared memory for both the partial sum and the x on device
     * because they require multiple access in MV Multiplication,
     * and its access advantages over global memory
	 */
    unsigned int row = blockIdx.x;  // Each block takes care of one row
    unsigned int col = threadIdx.x; // Each thread in the block handles multiple elements in the dot product
                           // in case m_cols > BLOCK_SIZE

    __shared__ double partialSum[blockDim.x]; // Shared memory for partial sums
    __shared__ double x_shared[blockDim.x];   // Shared memory for vector x

    // Initialize partial sum    
    partialSum[threadIdx.x] = 0;

    #pragma unroll
    for (int i = col; i < m_cols; i += blockDim.x)
    {
        // Load d_x into local shared memory 
        if (i < m_cols)
            x_shared[threadIdx.x] = d_x[i];

        __syncthreads();

        // Accumulate partial sums with stride = blockDim.x per row
        // ** Or alternatively, each thread handles a subvector of size ceil(m_cols / blockDim.x) **
        if (row < n_rows)
            partialSum[threadIdx.x] += d_A[row * m_cols + i] * x_shared[threadIdx.x];

        __syncthreads();
    }

    // Manually Reduce partial sums (within a block / a row) to a single sum
    // ** Alternatively, use Shuffle Methods within a Warp **
    for (int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (col < s)
            partialSum[col] += partialSum[col + s];

        __syncthreads();
    }

    // The first thread of a block writes the result to the output vector
    if (0 == col && row < n_rows)
        d_C[row] += partialSum[0]; // #### remove + for y = Ax
}

void MV_single(
        double *d_A,
        double *d_x,
        double *d_C,
        unsigned int n_rows,
        unsigned int m_cols){
    /*
     * Corresponding Kernel Launching method
     * Since each row is taken care of by exactly one warp/block
     * The number of blocks per grid should be exactly n_rows
     */
    dim3 nthreads(BLOCK_SIZE, 1, 1);    // BlockDim
    dim3 nblocks(n_rows, 1, 1);         // GridDim

    MV_single_kernel<<<nblocks, nthreads, 0, 0>>>(d_A, d_x, d_C, n_rows, m_cols);
}

//==================

__global__ void MV_multi_kernel(
		double *d_A,
		double *d_x,
		double *d_partialSums,
		unsigned int n_rows,
		unsigned int m_cols){
    /*
     * Multiple warps for each matrix row
     * First Part of a Two-Phase Reduction for MV operation
     * that writes partial sum of a block to an intermediate global memory
     */
    unsigned int blocksPerRow = (m_cols + blockDim.x - 1) / blockDim.x;
    unsigned int row = blockIdx.x / blocksPerRow;
    unsigned int startCol = (blockIdx.x % blocksPerRow) * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * blocksPerRow;

    __shared__ double x_shared[blockDim.x];

    // Load vector in to shared memory
    if (startCol < m_cols)
        x_shared[threadIdx.x] = d_x[startCol];

    __syncthreads();

    double sum = 0.0f;
    // for loop in case the entire row won't be covered by assigned blocks of specified size
    for (int col = startCol; col < m_cols; col += stride)
        if (col < m_cols)
            sum += d_A[row * m_cols + col] * x_shared[threadIdx.x];

    // no need for __syncthreads if using atomicAdd
    // each thread add result to the partial sum corresponding to its block
    atomicAdd(&d_partialSums[blockIdx.x], sum);
}


void MV_multi(
        double *d_A,
        double *d_x,
        double *d_partialSums,
        unsigned int n_rows,
        unsigned int m_cols){
    /*
     * Corresponding Kernel Launching method
     * Since each row is taken care of by multiple blocks
     * where each thread corresponds to one entry in the matrix
     * the total number of working threads should be (n_rows * m_cols)
     */
    dim3 nthreads(BLOCK_SIZE, 1, 1);    // BlockDim
    dim3 nblocks((n_rows * m_cols + nthreads.x - 1) / nthreads.x, 1, 1);         // GridDim

    MV_multi_kernel<<<nblocks, nthreads, 0, 0>>>(d_A, d_x, d_partialSums, n_rows, m_cols);
}


__global__ void multi_reduction_kernel(
        double *d_partialSums,
        double *d_C,
        unsigned int n_rows,
        unsigned int m_cols){
    /* 
     * Multiple warps for each matrix row
     * Second Part of a Two-Phase Reduction for MV operation
     * that perform summation with one block per row
     *
     * Require BLOCK_SIZE = WARP_SIZE to use __shfl_down_sync method
     */
    unsigned int row = blockIdx.x;
    unsigned int valsPerRow = (m_cols + blockDim.x - 1) / blockDim.x;  // partial sum values per row
    unsigned int startIdx = threadIdx.x;
    unsigned int offset = blockDim.x;

    double sum = 0.0f;
    // tile the block to cover the entire current row
    for (int i = startIdx; i < valsPerRow; i += offset)
        sum += d_partialSums[i + row * valsPerRow];

    __syncwarp(); // might be redundant when BLOCK_SIZE = WARP_SIZE
    // reduce within a warp/block
    for (int i = BLOCK_SIZE / 2; i > 0; i >>= 1)
        sum += __shfl_down_sync(FULL_MASK, sum, i);

    // the first thread in each block has the reduced sum
    // and writes to the output
    if (0 == threadIdx.x)
        d_C[row] += sum; // ### Remove + for y = Ax
}


void multi_reduction(
        double *d_partialSums,
        double *d_C,
        unsigned int n_rows,
        unsigned int m_cols){
    /*  
     * Corresponding Kernel Launching method
     * Since each row is taken care of by one block for reduction
     * the total number of blocks is exactly n_rows                              
     */
    dim3 nthreads(BLOCK_SIZE, 1, 1); // BlockDim
    dim3 nblocks(n_rows, 1, 1); // GridDim

    multi_reduction_kernel<<<nblocks, nthreads, 0, 0>>>(d_partialSums, d_C, n_rows, m_cols);
}


//==================


__global__ void MV_multi_ILP2_kernel(
		double *d_A, 
		double *d_x,
		double *d_partialSums,
		unsigned int n_rows,
		unsigned int m_cols){
    /*
     * Multiple warps for each matrix row
     * First Part of a Two-Phase Reduction for MV operation
     * that writes partial sum of a block to an intermediate global memory
     * with Instruction Level Parallelism of 2
     * 
     * ** ILP = 2 implies that the kernel should be launched with half of the original number of threads
     */

}

void MV_multi_ILP2(
        double *d_A,
        double *d_x,
        double *d_partialSums,
        unsigned int n_rows,
        unsigned int m_cols){
}

//========= Utilities  =========//

float eval_MV_Mult(unsigned int N, unsigned int M,\
        bool base = true, bool single = false) {
    /*
     * Evaluate the time elapsed in microsecond calling `MV_Mult()`
     * for the specified dimensions
     *
     * C = C + Ax
     *
     *      C (res) : N dimensional vector
     *      A (mat) : N by M matrix flattened into a (N*M) vector
     *      x (vec) : M dimensional vector
     *
     * Input:
     *      - N, M  : Matrix Dimensions
     *      - base  : if using the baseline kernel method
     *      - single: if using the single/multi blocks for each row of the matrix
     * 
     * Output:
     *      - elapsed time in microsecond (us)
     */
    // Check CUDA Device
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    //printf("ncuda_devices = %d\n", ncuda_devices);

    if (0 == ncuda_devices)
    {
        fprintf(stderr, "NO CUDA DEVICES, EXITING\n");
        return -1;                                                                                     
    }
    cudaSetDevice(0);

    // Misc vars
    gpuTimer timer;

    // Data Preparation on Host
    double* mat = rand_vec(N*M);
    double* vec = rand_vec(M);
    double* res = rand_vec(N);

    double *d_mat, *d_vec, *d_res, *partialSums;

    // Allocate Memory on Device
    cudaMalloc( (void**) &d_mat, sizeof(double)*N*M);
    cudaMalloc( (void**) &d_vec, sizeof(double)*M);
    cudaMalloc( (void**) &d_res, sizeof(double)*N);

    if (false == single) // partialSums for each block as intermediate results
    {
        // total number of blocks
        unsigned int nblocks = (N*M + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc( (void**) &partialSums, sizeof(double)*nblocks);
    }
   
    // Copy Data from HOST to DEVICE
    cudaMemcpy(d_mat, mat, sizeof(double)*N*M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, sizeof(double)*M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, sizeof(double)*N, cudaMemcpyHostToDevice);


    // Compute on Device
    if (base) // Baseline method
    {
        // Starting time
        timer.Start();
        
        MV_base(d_mat, d_vec, d_res, N, M);
    }
    else if (single) // Single Block Per row
    {
        timer.Start();

        MV_single(d_mat, d_vec, d_res, N, M);
    }
    else // Multiple Blocks per row
    {
        timer.Start();

        // Phase 1: partial sum within each block
        MV_multi(d_mat, d_vec, partialSums, N, M);
        // Phase 2: reduced sum
        multi_reduction(partialSums, d_res, N, M);
    }

    cudaDeviceSynchronize();   

    // Ending time
    timer.Stop();

    // Copy data from Device to HOST; and show some results
    // cudaMemcpy(res, d_res, sizeof(double)*N, cudaMemcpyDeviceToHost);
    // show_vec(N, res)

    // Free the memory paired with malloc() in ../lib/MyUtils.cpp
    free(mat);
    free(vec);
    free(res);

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_res);

    if (false == single)
        cudaFree(partialSums);

    // Calculate and return the elapsed time in microsecond
    return timer.Elapsed() / 1000.0;
}


//#endif /* LIB_MV_GPU_H_ */
