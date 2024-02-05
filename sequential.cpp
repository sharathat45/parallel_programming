/*
procedure Eliminate(A) // triangularize the matrix A
begin
    for k <- 0 to n-1  do //loop over all diagonal (pivot) elements
    
        for j <- k+1 to n-1 do         //for all elements in the row of, and to the right of, the pivot element
            А(k,j) = А(k,j) / A(k,k) ; // divide by pivot element
        endfor

        A(k,k) = 1;
        for i <- k+1 to n-1 do    //for all rows below the pivot row
            for j <- k+1 to n-1 do //for all elements in the row
                A(i,j) = A(i,j) - A(i,k) * A(k,j) ; //subtract pivot row times pivot column element
            endfor
            A(i,k) = 0;
        endfor
    endfor

end procedure
*/

#include <vector>
#include <iostream>
#include <time.h>
#include <random>

using namespace std;

typedef vector<vector<double>> Matrix;

void Eliminate(Matrix &A)
{
    int n = A.size();

    for (int k = 0; k < n; k++)
    {
        for (int j = k+1 ; j < n; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1;

        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
               
            #if DEBUG==1
                cout << "(" << i << "," << j << ")"
                     << " - "
                     << "(" << i << "," << k << ")"<< " * "
                     << "(" << k << "," << j << ")" << endl;
            #endif

            }
            A[i][k] = 0;
        }
    }
}

void printMatrix( Matrix &A)
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
    // printMatrix(A);
    // cout<<'\n';

    struct timespec start, end;
    double elapsed_time;

    clock_gettime(CLOCK_MONOTONIC, &start);
    Eliminate(A);
    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed_time = (end.tv_sec - start.tv_sec);
    elapsed_time += (end.tv_nsec - start.tv_nsec) / 1000000000.0; // Convert nanoseconds to seconds
    elapsed_time *= 1000;                                         // Convert to milliseconds
    cout << elapsed_time << endl;                            // Convert to milliseconds

    printMatrix(A);

    return 0;
}

