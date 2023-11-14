/*
 * MyUtils.h
 *
 *  Created on: Nov 8, 2023
 * Description: A separate file to keep track of utility functions used for evaluating
 * 				Matrix-Vector Multiplication's Performances
 */

#ifndef LIB_MYUTILS_H_
#define LIB_MYUTILS_H_


/*
 * Matrix Related Subroutines
 */





void show_mat(unsigned int N, unsigned int M, double** mat); // display the matrix in a readable format
																// specified as a N by M matrix

void show_vec(unsigned N, double* vec); // display the N dimensional vector as a column vector

double** rand_mat(unsigned int N, unsigned int M, bool contiguous = false, double lower_bound = -100, double upper_bound = 100); // generate a N by M matrix w or w\o contiguous memory allocation
																			// with malloc() or calloc()

double* rand_vec(unsigned int N, double lower_bound = -100, double upper_bound = 100); // generate a random N dimensional vector


/*
 * Performance Related Subroutines
 */
unsigned long get_time(); // Return current sys time in microsecond;

unsigned long tot_FLOP(unsigned int N, unsigned int M); // total FLOPs for an N by M matrix-vector multiplication

//unsigned long eval_MV_Mult(unsigned int N, unsigned int M,\
//		bool contiguous = false, bool unroll = false, bool padding = false); // evaluate the time elapsed calling MV_Mult for specified dimensions
														 // unit in microsecond



#endif /* LIB_MYUTILS_H_ */
