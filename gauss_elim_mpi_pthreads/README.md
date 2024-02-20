# Gaussian Elimination of NxN matrix using Pthread and MPI

This project includes both broadcast and pipelining versions.

Each version is timed and speedup results with respect to the sequential program are placed in `/out/output.txt` using the script `run_all.py`.

```bash
python3 run_all.py
```

## Running individual scripts

To run individual scripts, use the `make` command:

```bash
make seq
make pthread
make mpi
make pthread_pipe
make mpi_pipe
```
To change matrix size change `-DMAT_N=n` in makefile rule, where n is matriz size (nxn).

To change thread count change `-DTHREADS=n` in makefile rule, where n is number of threads.



## Checking correctness of parallel programme
To check the correctness of the code, use `check_correctness.py` which executes both the sequential and parallel code and places the output matrix in `/out/*.txt`. These can then be diffed with sequential.txt.

```bash
python3 check_correctness.py
```
