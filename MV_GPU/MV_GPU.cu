/*
 * MV_GPU.cu
 *
 *  Created on: Nov 24, 2023
 *
 * Description: Definitions of Kernels for Matrix-Vector Multiplication on GPU
 * 				along with the corresponding Kernel Launching methods
 * 
 *              Matrix-Vector Multiplication y = Ax
 *
 *              y : N dimensional vector
 *              A : N by M matrix flattened into a (N*M) vector, 
 *                  Row-Major order for contiguous memory access by threads
 *              x : M dimensional vector
 *
 * ## Current Issue: BLOCK_SIZE and blockIdx.x are used interchangeably ##
 */


#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "./lib/MyUtils.h"
#include "./lib/MV_GPU.h"

// Block size to conincide with Warp size for simplicity
#define BLOCK_SIZE 1024
#define WARP_SIZE 32
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

    if (row < n_rows) d_C[row] = y_val;
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

    __shared__ double partialSum[BLOCK_SIZE]; // Shared memory for partial sums
    __shared__ double x_shared[BLOCK_SIZE];   // Shared memory for vector x

    // Initialize partial sum    
    partialSum[threadIdx.x] = 0;

    #pragma unroll
    for (int i = col; i < m_cols; i += blockDim.x)
    {
        // Load d_x into local shared memory 
        if (i < m_cols)
            x_shared[threadIdx.x] = d_x[i];

        __syncthreads();

        // Accumulate partial sums with stride = BLOCK_SIZE per row
        // ** Or alternatively, each thread handles a subvector of size ceil(m_cols / BLOCK_SIZE) **
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
        d_C[row] = partialSum[0];
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

    __shared__ double x_shared[BLOCK_SIZE];

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
        d_C[row] = sum;
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

// Kernel with specified number of blocks and each block handles multiple rows of the matrix
__global__ void MV_KBlocks_kernel(
                double *d_A, // Matrix on DEVICE
                double *d_x, 
                double *d_y,
                unsigned int n_rows,
                unsigned int m_cols,
                unsigned int k_blocks)
{
    /*
     * Handling the Whole Matrix with a Specified Number of Blocks
     * while each block is processed by a CUDA stream
     *
     * There's actually no need to use shared memory or __shfl_sync 
     * for the vector d_x, 
     * as each thread will only calculate one product at a time
     */
    unsigned int rowsPerBlock = (n_rows + k_blocks - 1) / k_blocks; // Each Block will handle these rows respectively
    unsigned int startCol = threadIdx.x;
    unsigned int stride = blockDim.x;

    unsigned int warpId = threadIdx.x / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE;
    unsigned int nwarps = blockDim.x / WARP_SIZE;

    //__shared__ double partial_sums[BLOCK_SIZE]; // shared memory for partial sum
    __shared__ double warpSums[BLOCK_SIZE / WARP_SIZE]; // reduced sum within each warp
    unsigned int curr_row;
    double x_element;

    // in case the entire row is not covered by the assigned block size
    for (int col = startCol; col < m_cols; col += stride)
    {
        // load the vector x
        x_element = (col < m_cols) ? d_x[col] : 0.0;

        // For each row that the block is responsible for 
        #pragma unroll
        for (int i = 0; i < rowsPerBlock; i++)
        {
            double sum = 0.0;
            curr_row = blockIdx.x * rowsPerBlock + i;

            if (curr_row < n_rows)
                sum += d_A[curr_row * m_cols + col] * x_element;

            /*
            partial_sums[threadIdx.x] = sum;
            __syncthreads();

            
            // Reduciton in the shared memory 
            for (int s = blockDim.x / 2; s > 0; s >>=1)
            {
                if (threadIdx.x < s)
                    partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];

                __syncthreads();
            }
            */

            // Reduction *** with __shfl_down_sync()
            // Intra-Warp Reduction
            for (int s = WARP_SIZE / 2; s > 0; s >>= 1)
                sum += __shfl_down_sync(FULL_MASK, sum, s);

            __syncwarp();
            // Inter-Warp Reduction
            if (nwarps > 1) // more than 1 one warp existing
            {
                if (0 == laneId)
                    warpSums[warpId] = sum;

                __syncthreads();

                // Use the first warp to perform Inter-Warp reduction
                if (0 == warpId)
                {
                    for (int s = nwarps / 2; s > 0; s >>= 1)
                        if (threadIdx.x < s)
                            warpSums[threadIdx.x] += warpSums[threadIdx.x + s];
                    __syncthreads();
                }
                
                /*
                // Use the first thread to do simple summation
                if (0 == threadIdx.x)
                    for (int i = 1; i < nwarps; i++)
                        sum += warpSums[i];
                */
            }

            // the first thread to write the result for the current row
            if (0 == threadIdx.x && curr_row < n_rows)
                //atomicAdd(&d_y[curr_row], partial_sums[0]);
                atomicAdd(&d_y[curr_row], warpSums[0]);
        }
    }
}


float MV_KBlocks(
        double *h_A, // Matrix on HOST
        double *d_x, 
        double *d_y,
        double *h_y,
        unsigned int n_rows,
        unsigned int m_cols,
        unsigned int k_blocks, // # of Streams
        bool isCPU) 
{
    /*
     * Corresponding Kernel Launching Method with Streams
     * that also return a time for stream operations in microsecond (us)
     * Since there are total of K Blocks for the N by M matrix
     *** The block size could be anything ***
     *** for consistency, it is hereby set as BLOCK_SIZE for the learning purpose ***
     *the transfer of matrix A from CPU to GPU will be done by each stream respectively
     */
    dim3 nthreads(BLOCK_SIZE, 1, 1); // BlockDim
    dim3 nblocks(k_blocks, 1, 1); // GridDim

    unsigned int rowsPerBlock = (n_rows + k_blocks - 1) / k_blocks; 
    // Each Block/Stream will handle these rows respectively
    size_t sizePerBlock = rowsPerBlock * m_cols * sizeof(double);

    gpuTimer timerG; // should use CPU time to include the time used to copy data between host and device
    cpuTimer timer;

    // Temporary Memory for submatrices of A on GPU
    // Dividing the Matrix A into k_blocks sub matrices
    double *d_A_sub[k_blocks];
    for (int i = 0; i < k_blocks; i++)
        cudaMalloc(&d_A_sub[i], sizePerBlock);
    last_cuda_error("submatrix");

    // Creating Streams
    cudaStream_t streams[k_blocks];
    for (int i = 0; i < k_blocks; i++)
        cudaStreamCreate(&streams[i]);
    last_cuda_error("streams");

    timerG.Start();
    timer.Start();

    // Asynchronous Operations
    // Each stream Copies a block of rows of Matrix A into GPU
    for (int i = 0; i < k_blocks; i++)
    {
        // offset for the current block
        size_t offset_A = i * rowsPerBlock * m_cols;
        size_t offset_y = i * rowsPerBlock;

        if (k_blocks - 1 == i && n_rows % k_blocks != 0) // remainder rows
        {
            // Copy of submatrix to GPU
            cudaMemcpyAsync(d_A_sub[i], h_A + offset_A, (n_rows % k_blocks) * m_cols * sizeof(double), cudaMemcpyHostToDevice, streams[i]);

            // Compute multiplication of each block/stream
            // Launch kernel on the current stream
            MV_KBlocks_kernel <<< dim3(1, 1, 1), nthreads, 0, streams[i] >>> (d_A_sub[i], d_x, d_y + i * rowsPerBlock, n_rows % k_blocks, m_cols, 1);

            // Copy value of y back to CPU
            cudaMemcpyAsync(h_y + offset_y, d_y + offset_y, (n_rows % k_blocks) * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);

            last_cuda_error("memcpy D2H");
        }
        else
        {
            // Copy of submatrix to GPU
            cudaMemcpyAsync(d_A_sub[i], h_A + offset_A, sizePerBlock, cudaMemcpyHostToDevice, streams[i]);

            last_cuda_error("memcpy H2D");
            // Compute multiplication of each block/stream
            // Launch kernel on the current stream
            MV_KBlocks_kernel <<< dim3(1, 1, 1), nthreads, 0, streams[i] >>> (d_A_sub[i], d_x, d_y + i * rowsPerBlock, rowsPerBlock, m_cols, 1);

            last_cuda_error("launching kernel");
            // Copy value of y back to CPU
            cudaMemcpyAsync(h_y + offset_y, d_y + offset_y, rowsPerBlock * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);

            last_cuda_error("memcpy D2H");
        }

    }
    // Sync devices 
    // or simply sync each stream by cudaStreamSynchronize()
    cudaDeviceSynchronize();

    timerG.Stop();
    timer.Stop();

    // Clean up
    for (int i = 0; i < k_blocks; i++)
    {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_A_sub[i]);
    }

    // in microsecond (us)
    if (isCPU)
        return timer.Elapsed();
    else
        return timerG.Elapsed() * 1000.0;
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
void last_cuda_error(std::string event)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA Error at %s: %s\n", event.c_str(), cudaGetErrorString(err));
    }
}

float eval_MV_Mult(unsigned int N, unsigned int M,\
        bool base, bool single) {
    /*
     * Evaluate the time elapsed in microsecond calling `MV_Mult()`
     * for the specified dimensions
     *
     * C = C + Ax
     *
     *      C (res) : M dimensional vector
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
    double* mat = (double*)malloc(N * M * sizeof(double));
    double* vec = (double*)malloc(M * sizeof(double));
    double* res = (double*)malloc(N * sizeof(double));

    rand_vec(mat, N*M);
    rand_vec(vec, M);
    rand_vec(res, N);

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
    return timer.Elapsed() * 1000.0;
}

float eval_MV_Mult_streams(unsigned int N, unsigned int M, unsigned int Kstreams, bool isCPU)
{
    /*
     * Evaluate the time elapsed in microsecond calling `MV_KBlocks()`
     *
     * y = Ax
     *
     *      y (res) : N dimensional vector
     *      A (mat) : N by M matrix flattened into a (N*M) vector
     *      x (vec) : M dimensional vector // *** cudaHostAlloc
     *
     * Input:
     *      - N, M      : Matrix Dimensions
     *      - Kstreams  : Number of streams used for Matrix-Vector Multiplication
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

    // Data Preparation on Host
    //
    double *h_mat, *h_vec, *h_res;
    // Using pinned host memory for everything on host
    cudaHostAlloc((void **)&h_mat, N * M * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_vec, M * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_res, N * sizeof(double), cudaHostAllocDefault);
    // Randomize the entries for inputs
    rand_vec(h_mat, N*M);
    rand_vec(h_vec, M);
    // zeros for the output
    for (int i = 0; i < N; i++)
        h_res[i] = 0;


    // Allocate Memory on DEVICE for the vector and result
    double *d_vec, *d_res;
    cudaMalloc( (void**) &d_vec, sizeof(double)*M );
    cudaMalloc( (void**) &d_res, sizeof(double)*N );

    // Copy the vector from HOST to DEVICE
    cudaMemcpy(d_vec, h_vec, sizeof(double) * M, cudaMemcpyHostToDevice);

    // Compute and time the operations on DEVICE
    float t = MV_KBlocks(h_mat,
                         h_vec,
                         d_res,
                         h_res,
                         N,
                         M,
                         Kstreams,
                         isCPU);


    // Clean up
    cudaFreeHost(h_mat);
    cudaFreeHost(h_vec);
    cudaFreeHost(h_res);

    cudaFree(d_vec);
    cudaFree(d_res);

    return t;
}
