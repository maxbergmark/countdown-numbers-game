#include <mpi.h>
#include <vector>
#include <iostream>

#include "countdown_expression.h"
#include "countdown_mpi.h"

#define NUMBER_MAX_SIZE 1000

struct MPINumbers {
	int numbers[6];
	int counts[NUMBER_MAX_SIZE];
};

void send(std::vector<NumberSet> &numbers,
	int world_rank, int world_size, MPI_Datatype dt_mpi_numbers) {

	int min_ind = mpi_get_min_index(numbers, -1);
	int n_points = mpi_get_max_index(numbers, -1) - min_ind;
	std::cerr << "sending " << n_points << " data points from " << world_rank 
		<< " to " << 0 << std::endl;
	MPINumbers *data = (MPINumbers*) malloc(n_points * sizeof(MPINumbers));
	for (int i = 0; i < n_points; i++) {
		for (int j = 0; j < 6; j++) {
			data[i].numbers[j] = numbers[min_ind + i].numbers[j];
		}
		for (int j = 0; j < NUMBER_MAX_SIZE; j++) {
			data[i].counts[j] = numbers[min_ind + i].counts[j];
		}
	}
	MPI_Send(data, n_points, dt_mpi_numbers, 0, 1, MPI_COMM_WORLD);
	free(data);

}

void receive(std::vector<NumberSet> &numbers,
	int world_rank, int world_size, MPI_Datatype dt_mpi_numbers) {

	for (int i = 1; i < world_size; i++) {
		int min_ind = mpi_get_min_index(numbers, i);
		int n_points = mpi_get_max_index(numbers, i) - min_ind; 

		std::cerr << "receiving " << n_points << " data points from " 
			<< i << std::endl;
		MPINumbers *data = (MPINumbers*) malloc(n_points * sizeof(MPINumbers));
		MPI_Recv(data, n_points, dt_mpi_numbers,
				i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for (int j = 0; j < n_points; j++) {
			for (int k = 0; k < 6; k++) {
				if (data[j].numbers[k] != numbers[min_ind + j].numbers[k]) {
					std::cout << "ERROR!!!\n";
					for (int l = 0; l < 6; l++) {
						std::cout << " " << data[j].numbers[l];
					}
					std::cout << std::endl;
					print(numbers[min_ind + j].numbers);
					std::cout << std::endl;
					break;
				}
			}
			for (int k = 0; k < NUMBER_MAX_SIZE; k++) {
				numbers[min_ind + j].counts[k] = data[j].counts[k];
			}
		}
		free(data);
	}
}

void communicate(std::vector<NumberSet> &numbers) {

	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_Datatype dt_mpi_numbers;
	MPI_Type_contiguous(NUMBER_MAX_SIZE + 6, MPI_INT, &dt_mpi_numbers);
	MPI_Type_commit(&dt_mpi_numbers);
	if (world_rank == 0) {
		receive(numbers, world_rank, world_size, dt_mpi_numbers);
	} else {
		send(numbers, world_rank, world_size, dt_mpi_numbers);
	}
}

void setup_mpi() {
	MPI_Init(NULL, NULL);	
}

void finalize_mpi(std::vector<NumberSet> &numbers) {
	communicate(numbers);
	MPI_Finalize();
}

int mpi_world_rank() {
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	return world_rank;
}

int mpi_world_size() {
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	return world_size;
}

int mpi_get_min_index(std::vector<NumberSet> numbers, int world_rank) {
	if (world_rank == -1) {
		world_rank = mpi_world_rank();
	}
	int world_size = mpi_world_size();
	int current_index = world_rank * ((numbers.size()-1) / world_size + 1);
	return current_index;	
}

int mpi_get_max_index(std::vector<NumberSet> numbers, int world_rank) {
	if (world_rank == -1) {
		world_rank = mpi_world_rank();
	}
	int world_size = mpi_world_size();
	int current_index = (world_rank+1) * ((numbers.size()-1) / world_size + 1);
	if (world_rank == world_size - 1) {
		current_index = numbers.size();
	}
	return current_index;	
}
