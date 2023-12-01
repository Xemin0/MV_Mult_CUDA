/*
 * main.cpp
 *
 *  Created on: Nov 22, 2023
 *
 * Description: The main driver source file
 * 				to perform different tasks evaluating the performance of MV multiplication on GPU with Streams
 *
 */

#include <cstdio>
#include <fstream> // file processing

#include <iostream>
using namespace std;

#include "./lib/MyUtils.h" // Customized utility functions for Matrix-Vector Multiplication
#include "./lib/MV_GPU.h" // kernels and kernel launching methods for MV multiplication on GPU

void compare_performance(bool isCPU = true, unsigned int n = 5) {
	/*
	 * Task: Compare the Performances
     * with varying
     *      - Matrix size (N,M) ranging 1000 - 2000
     *      - Number of Streams (K) ranging 1 - 8
	 *
	 *
	 * Dimensions N and M will be chosen as 1000 + 200k
     * for k = 0, 1, ... , 5 
     *
	 * The results will be output into separate files.
	 *
	 * input:
	 * 		n       : number of runs to take average of; Excluding warm up loops
     *
	 * Output to files the matrices
     * whose sizes are of the above values for each number M of streams:
	 * 		1. Average time elapsed './e_us[K].dat'
	 */

	// Misc vars
	unsigned int i, j, k; // Indices
	unsigned int N, M; // Dimensions
    unsigned int Kstreams; // Number of Streams
	float t, curr_t; // time in microsecond
    unsigned int count;


	// Prepare to write results to a file
	// - Average time elapsed in microsecond

	// File names handling
	string file_t;


    // Varying the Number of Streams
	for (Kstreams = 1; Kstreams <= 8; Kstreams++) {
        // Prepare to write into a file
        file_t = "./data/e_us" + to_string(Kstreams);
        if (isCPU)
            file_t += "_cpu.dat";
        else
            file_t += "_gpu.dat";
        // Open the files in write mode
        // if they alrdy exist, truncates the contents
        fstream outfile_t;
        outfile_t.open(file_t.c_str(), ios::out | ios::trunc);

	    // Varying Matrix Dimensions N and K 
		for (i = 0; i < 6; i++) {
			N = 1000 + 200 * i;
            for (j = 0; j < 6; j++)
            {
                M = 1000 + 200 * j;

                t = 0.0; // reset the total recorded time
                count = 0;

                //Warm up loops
                for (k = 0; k < 3; k++) {
                    eval_MV_Mult_streams(N, M, Kstreams, isCPU); 
                }


                for (k = 0; k < n; k++) { // taking the average of n runs
                    curr_t = eval_MV_Mult_streams(N, M, Kstreams, isCPU); //#######
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
            }
		    outfile_t << endl;
		}
	    outfile_t.close();
	}

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
	// Task0: Different number of streams to handle MV multiplication of different sizes
	compare_performance();
	cout << "Task0 done" << endl;

	return 0;
}



