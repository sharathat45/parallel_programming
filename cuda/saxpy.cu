/*
nvprof (--print-gpu-trace) ./a.out -> profiling

cuobjdump -sass/-ptx a.out -> disassembly

float* data;

cudaMallocManaged(&data, n*sizeof(float)) -> accessible by both CPU and GPU (causes page fault for gpu and then copies data to gpu)
cudaMalloc(&data, n*sizeof(float)) -> accessible by GPU only
cudaMemcpy(data, src, n*sizeof(float), cudaMemcpyHostToDevice) -> copy from CPU to GPU
cudaFree(data);

cudaDeviceSynchronize() -> tell CPU to wait for the kernel to finish

dim3 dimBlock(x,y,z);
dim3 dimGrid(x,y,z);
someKernel<<<dimGrid, dimBlock>>>(d_data);

cudaMemAdvise(data, n*sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId); -> set some hints abt the data prefetch
cudaMemPrefetchAsync(data, n*sizeof(float), cudaGetDeviceId(&id)); -> prefetch data to GPU

cudaMemAdviseSetPreferredLocation or cudaMemAdviseSetReadMostly or cudaMemAdviseSetAccessedBy -> set hints for prefetching
readMostly -> such that cpu write don't inavlidate gpu one or copy it for read from cpu

__syncwarp() -> sync within a warp
___syncthreads() -> sync within a block
__shared__ float sdata[256]; -> shared memory

*/

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
