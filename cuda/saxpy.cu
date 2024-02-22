#include <iostream>
using namespace std;

__global__  
void saxpy_gpu (float* x, float* y, float scale, int size) {
	
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	
    if(i<size)
	{
		y[i] = scale * x[i] + y[i];
	}
}

int main() {

    int n = 10;
    float *d_x, *d_y;

	cudaMallocManaged(&d_x, n*sizeof(float));
	cudaMallocManaged(&d_y, n*sizeof(float));

    for(int i=0; i<n; i++)
    {
        d_x[i] = i*1.0;
        d_y[i] = i*1.0;
    }

	dim3 DimGrid(ceil(n/256.0),1,1); //(n+255)/256
	dim3 DimBlock(256,1,1);

	saxpy_gpu<<<DimGrid,DimBlock>>>(d_x, d_y, 2, n);
    cudaDeviceSynchronize();
    
    for(int i=0; i<n; i++)
    {
        cout<< d_y[i] << " ";
    }
    cout<<endl;

    cudaFree(d_x);
    cudaFree(d_y);

	return 0;
}
