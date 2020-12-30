#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <chrono>
#include <omp.h>
#include "countdown_expression.h"
#include <iomanip>
#include <cstdlib>

// #define COMPILE_MPI 1
#ifdef COMPILE_MPI
#include "countdown_mpi.h"
#endif

int get_next_index(int &current_index) {
	int i;
	#pragma omp critical
	{
		i = current_index;
		current_index++;
	}
	return i;
}

void print_stats(int i, int min_index, int max_index, 
	std::chrono::steady_clock::time_point begin) {
	
	int tid = omp_get_thread_num();
	if (tid >= 0) {
		float progress = (float) (i - min_index) / (max_index - min_index);
		std::chrono::steady_clock::time_point end 
			= std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration_cast
			<std::chrono::microseconds>(end - begin).count() / 1e6;
		double estimate = elapsed / progress;
		double remaining = estimate - elapsed;

		std::cerr << i << ":\t"
		<< std::fixed << std::setprecision(2)
		<< progress << "\t" << (remaining / 3600) << "h" << std::endl;
	}
}

int get_min_index(std::vector<NumberSet> numbers) {
#if COMPILE_MPI
	return mpi_get_min_index(numbers, -1);
#else
	return 0;
#endif
}

int get_max_index(std::vector<NumberSet> numbers) {
#if COMPILE_MPI
	return mpi_get_max_index(numbers, -1);
#else
	return numbers.size();
#endif
}

std::vector<NumberSet> create_number_sets() {
	std::vector<NumberSet> numbers = NumberSet::generate_numbers();

	int max_index = get_max_index(numbers);
	int min_index = get_min_index(numbers);
	int current_index = min_index;

	#pragma omp parallel
	{
		std::chrono::steady_clock::time_point begin 
			= std::chrono::steady_clock::now();

		int i;
		while ((i = get_next_index(current_index)) < max_index) {
			print_stats(i, min_index, max_index, begin);
			NumberSet &n = numbers[i];
			n.evaluate();
		}
	}

	return numbers;	
}

void output_results(std::vector<NumberSet> numbers) {
	uint64_t sum = 0;
	for (int i = 0; i < (int) numbers.size(); i++) {
		for (int j = 0; j < (int) numbers[i].counts.size(); j++) {
			NumberSet::global_counts[j] += numbers[i].counts[j];
			sum += numbers[i].counts[j];
		}
		std::cout << numbers[i] << std::endl;
	}
	std::cerr << "sum: " << sum << std::endl;

}

int get_world_rank() {
#if COMPILE_MPI
	return mpi_world_rank();
#else
	return 0;
#endif
}

int get_world_size() {
#if COMPILE_MPI
	return mpi_world_size();
#else
	return 1;
#endif
}

int get_num_threads(int argc, char **argv) {
	if (argc == 2) {
		return atoi(argv[1]);
	} else {
		return 1;
	}
}

int main(int argc, char **argv) {
#if COMPILE_MPI
	setup_mpi();
#endif
	int threads = get_num_threads(argc, argv);
	omp_set_num_threads(threads);
	NumberSet::generate_symbols();
	std::vector<NumberSet> numbers = create_number_sets();
	int world_rank = get_world_rank();
#if COMPILE_MPI
	finalize_mpi(numbers);
#endif
	if (world_rank == 0) {
		output_results(numbers);		
	}
}