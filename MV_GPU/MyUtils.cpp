/*
 * MyUtils.cpp
 *
 *  Created on: Nov8, 2023
 * Description: Implementation details of declared utility functions
 * 				for Matrix-Vector Multiplication
 */

#include <sys/time.h>
//#include <cstdlib>  // for rand() and srand()
				   	// dated and could be replaced by <random>
					// std::uniform_real_distribution
#include <time.h> // using time as the seed for rand()
#include <stdlib.h> // malloc() and calloc()

#include <iomanip> // for formated number printing

#include <iostream>
using namespace std;

/*
 * Matrix Related Subroutines
 */


void show_mat(unsigned int N, unsigned int M, double** mat) {
	/*
	 * Display the N by M matrix
	 */

	// set the precision to 4 decimal places
	// otherwise use printf
	cout << setprecision(4) << fixed;

	// Indices
	unsigned int i, j;
	// each row
	for (i = 0; i < N; i++) {
		// each column
		for (j = 0; j < M; j++) {
			cout << *(*(mat + i) + j) << " ";
		}
		cout << "\n";
	}
}

void show_vec(unsigned int N, double* vec) {
	/*
	 *  Display the N dimensional vector as a column vector
	 */
	for (unsigned int i = 0; i < N; i++) {
		cout << *(vec + i) << "\n";
	}
}


double** rand_mat(unsigned int N, unsigned int M, bool contiguous = false, double lower_bound = -100, double upper_bound = 100) {
	/*
	 * Generate a random N by M matrix
	 * w or w\o contiguous memory allocation, i.e. malloc() or calloc()
	 *
	 * Entry values range from -100 to 100 by default
	 */

	// indices
	unsigned int i, j;
	// matrix object to be returned
	double** mat = (double**)malloc(N * sizeof(double*));
	// Allocate Memory contiguously or not
	if (contiguous) {
		/*
		 * malloc() a trunk of memory of total size N*M*sizeof(double)
		 */
		double* contiguous_mem = (double*)malloc(N * M * sizeof(double));
		// Partition the trunk memory and assign them to the to-be-returned object
		// ??? But is it necessary to partition and assign the specific memory locations to the pointers of pointers "mat" ???
		for (i = 0; i < N; i++) {
			*(mat + i) = contiguous_mem + (i * M);
		}
	}
	else {
		// Allocate memory for each row separately
		for (i = 0; i < N; i++) {
			*(mat + i) = (double*)malloc(M * sizeof(double));
		}
	}

	// Generate random entries of type double to fill in the matrix
	// the old-school way :) with rand() or C++ std random()
		// Set the seed for generator
		//srand(time(0));

		// Since rand() returns integers
		// to get double-type entries we will need a non-zero denominator

	// Setting the seed for generator
	srandom(time(NULL));
	const long max_num = 1e6L;
	double range = upper_bound - lower_bound;

	// for each row
	for (i = 0; i < N; i++) {
		// for each column
		for (j = 0; j < M; j++) {
			mat[i][j] = lower_bound + range * (random() % max_num) / (max_num + 0.0);
		}
	}

	return mat;
}

double* rand_vec(unsigned int N, double lower_bound = -1e6, double upper_bound = 1e6) {
	/*
	 * Generate a random N dimensional vector
	 *
	 * Entry value ranges from -100 to 100 by default
	 */

	const long max_num = 1e8L;
	double range = upper_bound - lower_bound;
	// Setting the seed for the generator
	// srandom(time(NULL));

	// initialize the vector
	double* vec = (double*)malloc(N * sizeof(double));
	// Assign random numbers to each entry
	for (unsigned int i = 0; i < N; i++) {
		vec[i] = lower_bound + range * (random() % max_num) / (max_num + 0.0);
	}

	return vec;
}


/*
 * Performance Related Subroutines
 */

unsigned long get_time() {
	/*
	 * Get the current system time using gettimeofday() in microsecond
	 * ** gettimeofday() returns a struct timeval object that has two elements:
	 * 	1. seconds
	 * 	2. microseconds (1e-6 second)
	 *
	 */
	struct timeval curr_time;  // current time
	gettimeofday(&curr_time, 0); // get the current system time

	return (curr_time.tv_sec * 1e6) + curr_time.tv_usec; // Combining the two returned elements in microsecond
}


unsigned long tot_FLOP(unsigned int N, unsigned int M) {
	/*
	 * Total FLOPs for N by M matrix - vector multiplication
	 * C = C + Ax
	 *
	 * 		C : N dimensional vector
	 * 		A : N by M matrix
	 * 		x : M dimensional vector
	 *
	 * 		M multiplication + (M-1) additions for each of the N rows,
	 * 		along with N element-wise additions
	 * 		yielding a total of N*(2M-1) + N FLOPs
	 */
	return N*2*M;
}


