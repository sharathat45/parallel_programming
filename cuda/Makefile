BASE_NAME = matrix
EXECUTABLE = $(BASE_NAME)
SRC_C = $(BASE_NAME).cu
SRC_H = $(BASE_NAME).h
CC = nvcc
# CFLAGS = -O3 -std=c++11 

%: %.cu
	$(CC) $< -o $@
	./$@

all: saxpy simple

clean:
	rm -f *.bin *.o *.out saxpy simple matrix

.PHONY: clean