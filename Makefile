standard:
	g++ -O3 countdown_numbers_game.cpp countdown_expression.cpp -fopenmp -o countdown.out -g -Wall --std=c++17
mpi:
	mpic++ -O3 *.cpp -fopenmp -o countdown_mpi.out -g -Wall --std=c++17 -D COMPILE_MPI
clean:
	rm *.out
