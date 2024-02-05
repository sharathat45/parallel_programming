#include <mpi.h>
#include <vector>
#include <random>
#include <time.h>

using namespace std;
typedef vector<vector<double>> Matrix;

void Eliminate(Matrix &A)
{
    int tid, num_threads;

    MPI_Comm_rank(MPI_COMM_WORLD, &tid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);
    MPI_Request send_request, recv_request;

    for (int k = 0; k < MAT_N; k++)
    {
        if (k % num_threads == tid)
        {
            for (int j = k + 1; j < MAT_N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1;

            MPI_Isend(A[k].data(), MAT_N, MPI_DOUBLE, (tid + 1) % num_threads, 0, MPI_COMM_WORLD, &send_request);
        }
        else
        {
            MPI_Recv(A[k].data(), MAT_N, MPI_DOUBLE, (tid - 1 + num_threads) % num_threads, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // if the pivot row is not sent by nxt thread
            if (k % num_threads != (tid + 1) % num_threads)
            {
                MPI_Isend(A[k].data(), MAT_N, MPI_DOUBLE, (tid + 1) % num_threads, 0, MPI_COMM_WORLD, &send_request);
            }
        }

        for (int i = k + 1; i < MAT_N; i++)
        {
            if (i % num_threads == tid)
            {
                for (int j = k + 1; j < MAT_N; j++)
                {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                A[i][k] = 0;
            }
        }
        
    }
}

void printMatrix(Matrix &A)
{
    for (auto row : A)
    {
        for (auto elem : row)
        {
            cout << elem << ' ';
        }
        cout << '\n';
    }
}

int main()
{
    int count = 1, n = MAT_N, tid;
    Matrix A(n, vector<double>(n));

    random_device rd;  // Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    uniform_int_distribution<> distrib(1, 10);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[i][j] = distrib(gen);
        }
    }

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &tid);

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_MONOTONIC, &start);
    Eliminate(A);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time = (end.tv_sec - start.tv_sec);
    elapsed_time += (end.tv_nsec - start.tv_nsec) / 1000000000.0; // Convert nanoseconds to seconds
    elapsed_time *= 1000;                                         // Convert to milliseconds
    cout << "Elapsed Time: " << elapsed_time << endl;             // Convert to milliseconds

    if (tid == 0)
    {
        printMatrix(A);
    }

    MPI_Finalize();
    
    return 0;
}
