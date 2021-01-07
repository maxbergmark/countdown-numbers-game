# Countdown numbers game

Inspired by [this youtube video](https://www.youtube.com/watch?v=cVMhkqPP2YI&ab_channel=Computerphile), this repo contains code to bruteforce the [Countdown Numbers Game](https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_round).

The code for brute forcing solutions is written in C++ with support for both OpenMP and MPI. The code for analyzing the results is written in Python 3. An OpenCL implementation also exists for accelerated computing, which can be even faster than the C++ version.

## OpenCL implementation

After realizing that a major reason for speedup in the C++ version comes from not having to evaluate all `11!` permutations, the OpenCL version could also benefit from this fact. 

Since we still want to use a SIMD approach, the data is batched based on the total number of permutations that exists for each specific RPN expression. Then, each data batch is ran through the OpenCL kernel, where each work item in a batch requires practically the exact same amount of work. 

After a batch has been processed, we save the partial result in a dictionary. Once all batches are completed, the dictionary is transformed to the CSV output that we got from the C++ program. The MD5 checksum verifies that the CSV output is exactly correct.

## Compiling C++

To compile without MPI to run on a single machine, use 

	make

To compile with MPI support, use 

	make mpi

To clean up and remove compiled executables, use

	make clean

## Running

### C++

To run the executable without MPI support, use

    ./countdown.out <num_threads> > output.csv

If you're running the MPI version, instead use

    mpirun -np <num_processes> ./countdown_mpi.out <num_threads_per_process> > output.csv

In some cases, a hostfile can be needed. Depending on your MPI version, you might also need `--bind-to none` to enable OpenMP for each process:

	mpirun --bind-to none -f hostfile -np 4 ./countdown_mpi.out > output.csv

In both of these cases, `stdout` should be directed to a file, while `stderr` is used to show the progress of the script while running. 

### OpenCL

The OpenCL version is implemented using PyOpenCL. To run it, use:

	python countdown_cl.py <output_filename>

In this case, we don't redirect output to a file, since the script handles the file output on its own. Instead we supply the filename as an optional command line argument. 

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

### Checksums

Both the MPI version and the OpenMP version have identical file checksums:

	$ md5sum output.csv
	2c805d07a13038ffea535c84d33019b3  output.csv

	$ sha256sum output.csv
	cb792af51d27aff87aa14c63389c40bd07ad15bc65455d6d9d22e761ed1bc176  output.csv

### Caution

The exact count in the output file should be taken with a grain of salt. As an example, the following calculations might appear to be identical:

	(((2 + 1) x (2 + 1)) x 3) x 3 = 81    (RPN: 21+21+*3*3*)
	((2 + 1) x (2 + 1)) x (3 x 3) = 81    (RPN: 21+21+*33**)

However, since their RPN representation is different, they count as two different solutions. Thus, the real benefit out the output file is checking whether the number of ways to reach a specific target is zero or non-zero. 

## Performance

One important factor for improving performance comes from using `std::next_permutation`. Since each RPN expression consists of 11 tokens, we should expect there to be `11! = 39916800` ways to order these tokens. But this is only correct if all tokens are unique. When accounting for RPN expressions with duplicate tokens, the total number of RPN evaluations drops by a significant amount.

Without considering duplicate tokens, the total number of RPN evaluations is `13243 * 56 * 11! = 2.960262e+13`. If we account for duplicates, we instead end up evaluating `2.498643e+12` RPN expressions, a work reduction of 91.56%.

Of these `2.498643e+12` permutations, most will end up being invalid RPN expressions. And among the valid RPN expressions, there are instances where division is attempted when the result wouldn't be an integer, and there are instances where subtraction is attempted that would yield a result below 0. By recording each such occurence, we get the following data:

	Permutation fails:        2271493324800 (2.271e+12)
	Permutation successes:     227149332480 (2.271e+11)
	    Division fails:        121047625426 (1.210e+11)
	    Subtraction fails:      73850835651 (7.385e+10)
	    Total evaluations:     119547486361 (1.195e+11)

About 90.9% of all permutations are invalid, and do not need to be evaluated. As such, it is of utmost importance that the `check_rpn` function is as fast as possible. When it comes to evaluating RPN expressions, we have a work reduction of about 99.23% compared to evaluating all permutations.

## Running analysis

The analysis script is written in Python 3, and uses the packages in the `requirements.txt` file. To install everything and run, use:

	python -m pip install -r requirements.txt
	python analyze.py

## Benchmarks

### OpenCL, Intel 5820K + GTX 1080Ti (3584 CUDA cores)

With the OpenCL version, we record additional stats, the time is slightly slower than optimal:

	$ time python3 countdown_cl.py output.csv
	...
	real	6m4,459s
	user	6m2,912s
	sys	0m2,053s

With a bare bone solution, the runtime would be around 4 minutes. 

### Dell XPS 9560 (4 cores, 8 threads)

	$ time ./countdown.out 8 > output.csv
	...
	sum: 119547486361

	real	223m13,043s
	user	1718m59,454s
	sys	1m34,863s

### Dell XPS 9560 (4 cores, 8 threads) using MPI

	$ mpirun -np 4 time ./countdown_mpi.out 2 > output.csv
	....
	sum: 119547486361

	25732.60 user
	22.99 system 
	3:42:40 elapsed 
	192% CPU

	25922.14 user
	17.89 system 
	3:42:40 elapsed
	194% CPU

	25000.80 user
	43.57 system 
	3:42:40 elapsed 
	187% CPU

	25480.52 user
	37.87 system
	3:43:00 elapsed
	190% CPU

By combining the results of each process, we get a total `user` time of `102136.06s = 1702m16.06s`, which is almost identical to the pure OpenMP result.

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

### MPI dynamic batching

A big problem with the MPI version of the algorithm is that while the work is split into equal parts in terms of size, it is difficult to predict the amount of work required for each set of numbers. As such, the amount of actual work per process could vary by a substantial amount, which decreases average CPU utilization for the entire duration of the runtime. 

To solve this problem, the MPI implementation could use dynamic batching to divide the work among the nodes in the cluster. However, this is likely to interfere with the general structure of the non-MPI code, which is why it has been left out of scope for now. 
