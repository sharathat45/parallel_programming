BASE_NAME = shared_mem
EXECUTABLE = $(BASE_NAME)
SRC_C = $(BASE_NAME).cpp
SRC_H = $(BASE_NAME).h
CC = g++
CFLAGS = -O3 -std=c++11 

seq_build: sequential.cpp
	$(CC) -o sequential sequential.cpp $(CFLAGS) -DDEBUG=0 -DMAT_N=8 -DTHREADS=8

seq :  seq_build
	./sequential

pthread_pipe_build: pthread_pipe.cpp
	$(CC) -lpthread -o pthread_pipe pthread_pipe.cpp $(CFLAGS) -DDEBUG=0 -DMAT_N=8 -DTHREADS=16

pthread_pipe : pthread_pipe_build
	./pthread_pipe

pthread_build: pthread.cpp
	$(CC) -lpthread -o pthread pthread.cpp $(CFLAGS) -DDEBUG=0 -DMAT_N=8 -DTHREADS=4

pthread : pthread_build
	./pthread

mpi_build: mpi.cpp
	mpic++ -o mpi mpi.cpp -DMAT_N=4 -DDEBUG=0

mpi : mpi_build
	mpirun -np 4 ./mpi 

mpi_pipe_build: mpi_pipe.cpp
	mpic++ -o mpi_pipe mpi_pipe.cpp -DMAT_N=8 -DDEBUG=0

mpi_pipe : mpi_pipe_build
	mpirun -np 4 ./mpi_pipe 

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f *.bin *.o *.out $(EXECUTABLE) sequential pthread mpi prog program mpi_pipe pthread_pipe ./out/*.txt

.PHONY: clean