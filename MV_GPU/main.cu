/*
 * main.cpp
 *
 *  Created on: Sep 20, 2023
 *
 * Description: The main driver source file
 * 				to perform different tasks evaluating the performance of MV multiplication on GPU
 *
 */

#include <cstdio>
#include <fstream> // file processing
#include <cmath>

#include <iostream>
using namespace std;

#include "./lib/MyUtils.h" // Customized utility functions for Matrix-Vector Multiplication
#include "./lib/MV_GPU.h" // kernels and kernel launching methods for MV multiplication on GPU

void eval_FLOPrate(bool base = true, bool single = false, unsigned int n = 5) {
	/*
	 * Task 1 and 2: FLOPrate for varying matrix dimensions
     * with 
     *      - Baseline Method: Single thread for each row
     *      - Single Block for each row
     *      - Multiple Blocks for each row
	 *
	 * FLOPrate (TeraFLOP per second) = total FLOPs / total elapsed time
	 *
	 * Dimensions N and M will be chosen as 10^(1 + 0.5k) 
     * for k = 0, 1, ... , 7 = log_max_size.
     *
	 * Total FLOPs is calculated as (2*N*M) in tot_FLOP()
	 * Total elapsed time is recorded as the average of 5 eval_MV_Mult()
	 *
	 * The results will be output into separate files.
	 *
	 * input:
	 * 		n       : number of runs to take average of; Excluding warm up loops
	 * 		base    : if using baseline method
     *      single  : if using single/multiple blocks per row
     *
	 * Output to files the matrices of following values for each dimension config:
	 * 		1. Average time elapsed './e_us.dat'
	 * 		2. Average FLOP rate  './FLOPrate.dat'
	 */

	// Misc vars
	unsigned int i, j, k; // Indices
	unsigned int N, M; // Dimensions
	float t, curr_t; // time in microsecond

	unsigned long n_flop; // number of FLOPs

    unsigned int log_max_size = 8, count; // max size = 10^[1 + 0.5 * (log_max_size - 1)]

	// Prepare to write results to a file
	// - Average time elapsed in microsecond
	// - FLOPrate in teraFLOP/s

	// File names handling
	string file_t = "./data/e_us";
	string file_fr = "./data/FLOPrate";

	if (base)
	{
		file_t += "_base";
		file_fr += "_base";
	}
	else if (single)
	{
		file_t += "_single";
		file_fr += "_single";
	}
    else
    {
        file_t += "_multi";
        file_fr += "_multi";
    }

	file_t += ".dat";
	file_fr += ".dat";


	// Open the files in write mode
	// if they alrdy exist, truncates the contents
	fstream outfile_t, outfile_fr;
	outfile_t.open(file_t.c_str(), ios::out | ios::trunc);
	outfile_fr.open(file_fr.c_str(), ios::out | ios::trunc);

	// Varying N and M logarithmically
	for (i = 0; i < log_max_size; i++) {
		N = pow(10.0, 1 + 0.5 * i); // converted to unsigned int
		for (j = 0; j < log_max_size; j ++) {
			M = pow(10.0, 1 + 0.5 * j);

			// Number of FLOPs
			n_flop = tot_FLOP(N, M);

			t = 0.0; // reset the total recorded time
            count = 0;

			//Warm up loops
			for (k = 0; k < 3; k++) {
				eval_MV_Mult(N, M, base, single);
			}


			for (k = 0; k < n; k++) { // taking the average of n runs
				curr_t = eval_MV_Mult(N, M, base, single);
                if (curr_t <= 0)
                    cout << "runtime is not positive" << endl;
                else
                {
                    t += curr_t;
                    count++;
                }
			}
			t /= count; // taking the average time in microsecond

			// Write the result to file_loc
			outfile_t << t << " "; // in microsecond
			outfile_fr << n_flop / (1e6 * (t + 0.0)) << " "; // teraFLOP/s
		}
		outfile_t << endl;
		outfile_fr << endl;

	}

	outfile_t.close();
	outfile_fr.close();
}


int main() {
	// ------------------ Validation for each imported subroutine ------------------//

	/*
	double* vec = rand_vec(3);
	show_vec(3, vec);
	cout << endl;

	double** mat = rand_mat(2,3, false);
	show_mat(2, 3, mat);
	cout << endl;

	double* res = rand_vec(2);
    eval_MV_Mult(3, 2, true, flase);

     */

	//------------------------------------------------------------------//
	// Task0: FLOPrate for different matrix dimensions 
    // Baseline : 1 thread per row
	// base     : true
	// single   : false
	eval_FLOPrate(true, false);
	cout << "Task0 done" << endl;

	// Task1: FLOPrate for different matrix dimensions 
    // 1 block per row
	// base     : false
	// single   : true
	eval_FLOPrate(false, true);
	cout << "Task1 done" << endl;

	// Task2: FLOPrate for different matrix dimensions
    // Multiple blocks per row
	// base     : false
	// single   : false
	eval_FLOPrate(false, false);
	cout << "Task2 done" << endl;

	return 0;
}



