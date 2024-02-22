#include <iostream>
#include <vector>
#include <random>
#include <chrono> 
#include <iomanip>
#include <curand_kernel.h>

using namespace std;

int monte_carlo_cpu(uint64_t iterationCount, uint64_t sampleSize)
{
    auto tStart = chrono::high_resolution_clock::now();

    random_device random_device;
    uniform_real_distribution<float> dist(0.0, 1.0);

    float x, y;
    uint64_t hitCount = 0;
    uint64_t totalHitCount = 0;

    for (int iter = 0; iter < iterationCount; ++iter)
    {
        hitCount = 0;

        for (uint64_t idx = 0; idx < sampleSize; ++idx)
        {
            x = dist(random_device);
            y = dist(random_device);

            if (int(x * x + y * y) == 0)
            {
                ++hitCount;
            }
        }
        totalHitCount += hitCount;
    }

    //	Calculate Pi
    float approxPi = ((double)totalHitCount / sampleSize) / iterationCount;
    approxPi = approxPi * 4.0f;

    cout << setprecision(10);
    cout << "Estimated Pi = " << approxPi << "\n";

    auto tEnd = chrono::high_resolution_clock::now();

    chrono::duration<double> time_span = (tEnd - tStart);
    cout << "It took CPU: " << time_span.count() << " seconds.";

    return 0;
}


__global__
void monte_carlo_gpu(uint64_t iterationCount, uint64_t sampleSize, float* pi)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState_t rng;
    curand_init(clock64(), tid, 0, &rng);

    __shared__ float thread_blk_hitcount;
    if(threadIdx.x == 0)
    {
        thread_blk_hitcount = 0;
    }   

    __syncthreads();

    float hitcount = 0;
    if (tid < sampleSize)
    {
        for(int i = 0; i < iterationCount; i++)
        {
            float x = curand_uniform(&rng);
            float y = curand_uniform(&rng);

            if (int(x * x + y * y) == 0)
            {
                ++hitcount;
            }
        }
    }

   atomicAdd(&thread_blk_hitcount, hitcount);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        //	Calculate Pi
        float approxPi = ((double)thread_blk_hitcount / sampleSize) / iterationCount;
        approxPi = approxPi * 4.0f;
        atomicAdd(pi, approxPi);
    }
}

int main(){

uint64_t iterationCount = 100; 
uint64_t sampleSize = 1000000;

// monte_carlo_cpu(iterationCount, sampleSize);

auto tStart = chrono::high_resolution_clock::now();

float *pi;
cudaMallocManaged(&pi, sizeof(float));
*pi = 0;

dim3 DimGrid(ceil(float(sampleSize) / 256.0), 1, 1);
dim3 DimBlock(256, 1, 1);
monte_carlo_gpu<<<DimGrid, DimBlock>>>(iterationCount, sampleSize, pi);
cudaDeviceSynchronize();

auto tEnd = chrono::high_resolution_clock::now();

cout << setprecision(10);
cout << "Estimated Pi = " << *pi << "\n";
chrono::duration<double> time_span = (tEnd - tStart);
cout << "It took GPU: " << time_span.count() << " seconds."<<endl;

}