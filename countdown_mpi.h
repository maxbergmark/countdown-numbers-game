#pragma once

void communicate(std::vector<NumberSet> &numbers);
void setup_mpi();
void finalize_mpi(std::vector<NumberSet> &numbers);
int mpi_world_rank();
int mpi_world_size();
int mpi_get_min_index(std::vector<NumberSet> numbers, int world_rank);
int mpi_get_max_index(std::vector<NumberSet> numbers, int world_rank);
