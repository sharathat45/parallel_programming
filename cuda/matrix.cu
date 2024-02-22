#include <iostream>
using namespace std;
#define TILE_WIDTH 32

__global__ 
void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  float Pvalue = 0;

  for (int p = 0; p < Width/TILE_WIDTH; ++p) {
    ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];
    ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*Width + Col];

    __syncthreads();
    for (int i = 0; i < TILE_WIDTH; ++i)
      Pvalue += ds_M[ty][i] * ds_N[i][tx];
    __syncthreads();
  }
  
  P[Row*Width+Col] = Pvalue;
}



int main()
{
  const int size = 1024;
  float* M = new float[size*size];
  float* N = new float[size*size];
  float* P = new float[size*size];

  cudaMallocManaged(&M, size*size*sizeof(float));
  cudaMallocManaged(&N, size*size*sizeof(float));
  cudaMallocManaged(&P, size*size*sizeof(float));

  for (int i = 0; i < size*size; i++) {
    M[i] = 1.0;
    N[i] = 1.0;
  }

  dim3 DimGrid(ceil(size/TILE_WIDTH), ceil(size/TILE_WIDTH), 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  MatrixMulKernel<<<DimGrid,DimBlock>>>(M,N,P,size);
  cudaDeviceSynchronize();

  cout << P[0]<< endl;

  cudaFree(M);
  cudaFree(N);
  cudaFree(P);
}
