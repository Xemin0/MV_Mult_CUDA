/*
 * MV_GPU.h
 *
 *  Created on: Nov 22, 2023
 *      Author: Shilai Yang
 *              Yuhong Zhang
 *
 * Description: Declarations of Kernels for Matrix-Vector Multiplication on GPU
 * 				along with the corresponding Kernel Launching methods
 * 
 *              Matrix-Vector Multiplication y = Ax
 *
 *              y : N dimensional vector
 *              A : N by M matrix flattened into a (N*M) vector, 
 *                  Row-Major order for contiguous memory access by threads
 *              x : M dimensional vector
 *
 */

//#ifndef LIB_MV_GPU_H_
//#define LIB_MV_GPU_H_

#pragma once //Attempted replacement for the Include_Guard


#include <cuda.h>


// Utils
struct gpuTimer;

//=================

// Kernels and Kernel Launching Methods

__global__ void MV_base_kernel(
		double *d_A,
		double *d_x,
		double *d_C,
		unsigned int n_rows,
		unsigned int m_cols);
	/*
     * Baseline Kernel:
     * each thread handles an entire row of matrix
     * while loading one element of the vector
     * Shuflle method instead of shared memory for summation 
	 */

void MV_base(
		double *d_A,
		double *d_x,
		double *d_C,
		unsigned int n_rows,
		unsigned int m_cols);
    /*
     * Corresponding Kernel Launching method
     * Since each row is taken care of by exactly one thread
     * The total number of threads should be exactly n_rows
     */

//=================

__global__ void MV_single_kernel(
		double *d_A,
		double *d_x,
		double *d_C,
		unsigned int n_rows,
		unsigned int m_cols);
    /*
     * Single warp for each matrix row
     * Using shared memory for both the partial sum and the x on device
     * because they require multiple access in MV Multiplication,
     * and its access advantages over global memory
     */

void MV_single(
        double *d_A,
        double *d_x,
        double *d_C,
        unsigned int n_rows,
        unsigned int m_cols);
    /*
     * Corresponding Kernel Launching method
     * Since each row is taken care of by exactly one warp/block
     * The number of blocks per grid should be exactly n_rows
     */

//==================

__global__ void MV_multi_kernel(
		double *d_A,
		double *d_x,
		double *d_partialSums,
		unsigned int n_rows,
		unsigned int m_cols);
    /*                                                                           
     * Multiple warps for each matrix row
     * First Part of a Two-Phase Reduction for MV operation
     * that writes partial sum of a block to an intermediate global memory
     */


void MV_multi(
        double *d_A,
        double *d_x,
        double *d_partialSums,
        unsigned int n_rows,
        unsigned int m_cols);
    /*  
     * Corresponding Kernel Launching method
     * Since each row is taken care of by multiple blocks
     * where each thread corresponds to one entry in the matrix
     * the total number of working threads should be (n_rows * m_cols)
     */


__global__ void multi_reduction_kernel(
		double *d_partialSums,
        double *d_C,
		unsigned int n_rows,
		unsigned int m_cols);
    /* 
     * Multiple warps for each matrix row
     * Second Part of a Two-Phase Reduction for MV operation
     * that perform summation with one block per row
     *
     * Require BLOCK_SIZE = WARP_SIZE to use __shfl_down_sync method
     */


void multi_reduction(
        double *d_partialSums,
        double *d_C,
        unsigned int n_rows,
        unsigned int m_cols);
    /*  
     * Corresponding Kernel Launching method
     * Since each row is taken care of by one block for reduction
     * the total number of blocks is exactly n_rows
     */


//=================

__global__ void MV_KBlocks_kernel(
                double *d_A, // Matrix on DEVICE
                double *d_x, // cudaHostAlloc
                double *d_y,
                unsigned int n_rows,
                unsigned int m_cols,
                unsigned int k_blocks);
    /*
     * Handling the Whole Matrix with a Specified Number of Blocks
     * while each block is processed by a CUDA stream
     *
     * the vector x will be allocated by cudaHostAlloc accessible by both the HOST and the DEVICE
     * Using shared memory to load the vector x 
     * as it will be used repeatedly for multiple rows of the matrix within each block
     */


float MV_KBlocks(
        double *h_A, // Matrix on HOST
        double *d_x, // cudaHostAlloc
        double *d_y,
        double *h_y,
        unsigned int n_rows,
        unsigned int m_cols,
        unsigned int k_blocks); // # of streams
    /*
     * Corresponding Kernel Launching Method with Streams
     * that also returns a time of stream operations in microsecond (us)
     * Since there are total of K Blocks for the N by M matrix
     *** The block size could be anything ***
     *** for consistency, it is hereby set as BLOCK_SIZE for the learning purpose ***
     * the transfer of matrix A from CPU to GPU will be done by each stream respectively
     */



//=================


__global__ void MV_multi_ILP2_kernel(
		double *d_A, 
		double *d_x,
		double *d_partialSums,
		unsigned int n_rows,
		unsigned int m_cols);
    /*      
     * Multiple warps for each matrix row
     * First Part of a Two-Phase Reduction for MV operation
     * that writes partial sum of a block to an intermediate global memory
     * with Instruction Level Parallelism of 2
     * 
     * ** ILP = 2 implies that the kernel should be launched with half of the original number of threads
     */


void MV_multi_ILP2(
        double *d_A,
        double *d_x,
        double *d_partialSums,
        unsigned int n_rows,
        unsigned int m_cols);

//========= Utilities  =========//

float eval_MV_Mult(unsigned int N, unsigned int M,\
        bool base = true, bool single = false);

float eval_MV_Mult_streams(unsigned int N, unsigned int M, unsigned int Kstreams);
//#endif /* LIB_MV_GPU_H_ */
