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

In both of these cases, `stdout` should be directed to a file, while `stderr` is used to show the progress of the script while running. 

## Benchmarks

### Dell XPS 9560 (4 cores, 8 threads)

	// TODO

### Amazon EC2 c5a.16xlarge (32 cores, 64 threads)


	[ec2-user@ip-172-31-38-192 ~]$ time ./countdown.out 64 > output.csv
	...
	sum: 119547486361

	real	16m49.434s
	user	1071m17.573s
	sys	0m0.824s

### RK3399 cluster (4x 6 threads)

	// TODO