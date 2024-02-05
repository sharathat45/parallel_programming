#include <vector>
#include <iostream>
#include <pthread.h>
#include <time.h>
#include <random>

using namespace std;
typedef vector<vector<double>> Matrix;

pthread_barrier_t barrier;

pthread_mutex_t mutexsum;

struct ThreadData
{
    Matrix *A;
    int tid;
};

void *EliminateRows(void *arg)
{
    ThreadData *data = (ThreadData *)(arg);
    Matrix &A = *(data->A);
    int tid = data->tid;
    int n = MAT_N;
 
    for (int k = 0; k < MAT_N; k++)
    {
        if (k % THREADS == tid)
        {
            for (int j = k + 1; j < MAT_N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1;
        }

        pthread_barrier_wait(&barrier);

        for (int i = k + 1; i < MAT_N; i++)
        {
            if (i % THREADS == tid)
            {
                for (int j = k + 1; j < MAT_N; j++)
                {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                A[i][k] = 0;
            }
        }
    }

    pthread_exit(NULL);
}

void Eliminate(Matrix &A)
{
    vector<pthread_t> threads(THREADS);
    vector<ThreadData> threadData(THREADS);
    pthread_barrier_init(&barrier, NULL, THREADS);

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int tid = 0; tid < THREADS; tid++)
    {
        threadData[tid] = {&A, tid};
        pthread_create(&threads[tid], NULL, EliminateRows, &threadData[tid]);
    }

    for (int tid = 0; tid < THREADS; tid++)
    {
        pthread_join(threads[tid], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time = (end.tv_sec - start.tv_sec);
    elapsed_time += (end.tv_nsec - start.tv_nsec) / 1000000000.0; // Convert nanoseconds to seconds
    elapsed_time *= 1000;                                         // Convert to milliseconds
    cout << elapsed_time << endl;                                 // Convert to milliseconds

    pthread_barrier_destroy(&barrier);
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
    int count = 1, n = MAT_N;
    Matrix A(n, vector<double>(n));

    random_device rd;  // Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    uniform_int_distribution<> distrib(1, 10);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = distrib(gen);
        }
    }

    Eliminate(A);
#if DEBUG == 1
    printMatrix(A);
#endif
    return 0;
}
