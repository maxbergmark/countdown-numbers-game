# Countdown numbers game

Inspired by [this youtube video](https://www.youtube.com/watch?v=cVMhkqPP2YI&ab_channel=Computerphile), this repo contains code to bruteforce the [Countdown Numbers Game](https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_round).

The code for brute forcing solutions is written in C++ with support for both OpenMP and MPI. The code for analyzing the results is written in Python3.

## Compiling C++

To compile without MPI to run on a single machine, use 

	make

To compile with MPI support, use 

	make mpi

To clean up and remove compiled executables, use

	make clean

## Running

To run the executable without MPI support, use

    ./countdown.out <num_threads> > output.csv

If you're running the MPI version, instead use

    mpirun -np <num_processes> ./countdown_mpi.out <num_threads_per_process> > output.csv

In some cases, a hostfile can be needed. Depending on your MPI version, you might also need `--bind-to none` to enable OpenMP for each process:

	mpirun --bind-to none -f hostfile -np 4 ./countdown_mpi.out > output.csv

In both of these cases, `stdout` should be directed to a file, while `stderr` is used to show the progress of the script while running. 

## Partial results

An important aspect of the Countdown numbers game is that the contestant doesn't have to use all numbers in their calculation. To simulate this, the output of each RPN expression is a vector of numbers describing all partial results along with the final result of the calculation. 

### Example

If we have the RPN expression:

	21+21+*3*3*

We parse it from left to right, adding numbers to the stack as we go along. After each token in the expression is parsed, we check the size of the stack. If there is only one element on the stack, we save that element as a partial result in an output array. We would parse it like this:

	Token	Stack	Output
	2		[2]		[2]
	1		[2,1]		[2]
	+		[3]		[2,3]
	2		[3,2]		[2,3]
	1		[3,2,1]		[2,3]
	+		[3,3]		[2,3]
	*		[9]		[2,3,9]
	3		[9,3]		[2,3,9]
	*		[27]		[2,3,9,27]
	3		[27,3]		[2,3,9,27]
	*		[81]		[2,3,9,27,81]

From this single expression, we get 5 reachable target numbers (although none of them are 3-digit). All 5 of these results will be recorded in the output file.

## Output format

The C++ script outputs CSV to `stdout`. This output contains information about all 13243 boards possible in the Coundown numbers game. 

The output consists of 13243 rows, each one describing a set of board numbers. The 6 first columns of each row are the board numbers themselves, and the following 1000 columns describe how many ways you can reach each target number using that specific set of board numbers. The total file size of the output is around 55MB.

### Example

Here's the first row of the output:

	1,1,2,2,3,3,290745,418471,399627,427345,328416,247992,328538,102224,97027,127811,66516,46318,87900,22880,23082,37544,24134,10552,46278,6992,16696,16476,6608,2688,24772,4720,3132,7784,4064,1784,16492,1688,2760,2052,928,2232,23540,2520,1392,1200,2344,176,2976,96,352,3048,240,144,5192,304,328,136,152,120,4248,240,144,144,0,0,1248,0,0,144,144,0,0,0,0,0,0,0,1320,0,0,0,0,0,0,0,0,120,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

The first 6 numbers are `1,1,2,2,3,3`, representing the board numbers. We know that the highest possible target number that is possible to reach using these board numbers is `81 = (1+2)*(1+2)*3*3`. If we look at the remaining columns:

	290745,418471,399627,427345,328416,247992,328538,102224,97027,127811,66516,46318,87900,22880,23082,37544,24134,10552,46278,6992,16696,16476,6608,2688,24772,4720,3132,7784,4064,1784,16492,1688,2760,2052,928,2232,23540,2520,1392,1200,2344,176,2976,96,352,3048,240,144,5192,304,328,136,152,120,4248,240,144,144,0,0,1248,0,0,144,144,0,0,0,0,0,0,0,1320,0,0,0,0,0,0,0,0,120,0,0, ...

we see the number `120` at the end of the non-zero numbers. This number is in position 81 (zero-indexed), and describes the number of RPN permutations that resulted in the number 81. 

### Caution

The exact count in the output file should be taken with a grain of salt. As an example, the following calculations might appear to be identical:

	(((2 + 1) x (2 + 1)) x 3) x 3 = 81    (RPN: 21+21+*3*3*)
	((2 + 1) x (2 + 1)) x (3 x 3) = 81    (RPN: 21+21+*33**)

However, since their RPN representation is different, they count as two different solutions. Thus, the real benefit out the output file is checking whether the number of ways to reach a specific target is zero or non-zero. 

## Benchmarks

### Dell XPS 9560 (4 cores, 8 threads)


	$ time ./countdown.out 8 > output_test.csv
	...
	sum: 119547486361

	real	223m13,043s
	user	1718m59,454s
	sys	1m34,863s

### Dell XPS 9560 (4 cores, 8 threads) using MPI

	// TODO

### Amazon EC2 c5a.16xlarge (32 cores, 64 threads)


	$ time ./countdown.out 64 > output.csv
	...
	sum: 119547486361

	real	16m49.434s
	user	1071m17.573s
	sys	0m0.824s

### RK3399 cluster (4x 6 threads)

In this benchmark, the calculation ran on a cluster of 4 NanoPi M4 SBCs using MPI (6 cores, 1.4-1.8GHz). Thus, the `user` time reported should be multiplied by 4 (I didn't capture process times on each node in the cluster, only on the root node). This results in a `user` time of `4069m32.736s`, 

	$ time mpirun --bind-to none -f hostfile -np 4 ./countdown_mpi.out 6 > output.csv
	...
	sum: 119547486361

	real    245m58.023s
	user    1017m23.184s
	sys     9m29.176s


## Future ideas

### OpenCL

An attempt to rewrite the algorithm using OpenCL was made, but it proved to be slower compared to running on the CPU. With the MPI support, there isn't much need to use OpenCL, since the only way it would be faster is if the calculation could be split over multiple GPUs, but then it could just as easily be run on multiple CPUs.

### MPI dynamic batching

A big problem with the MPI version of the algorithm is that while the work is split into equal parts in terms of size, it is difficult to predict the amount of work required for each set of numbers. As such, the amount of actual work per process can vary by a substantial amount, which decreases average CPU utilization for the entire duration of the runtime. 

To solve this problem, the MPI implementation could use dynamic batching to divide the work among the nodes in the cluster. However, this is likely to interfere with the general structure of the non-MPI code, which is why it has been left out of scope for now. 
