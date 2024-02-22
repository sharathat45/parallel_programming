nvprof (--print-gpu-trace) ./a.out -> profiling

cuobjdump -sass/-ptx a.out -> disassembly

float* data;
cudaMallocManaged(&data, n*sizeof(float)) -> accessible by both CPU and GPU (causes page fault for gpu and then copies data to gpu)
cudaMalloc(&data, n*sizeof(float)) -> accessible by GPU only
cudaMemcpy(data, src, n*sizeof(float), cudaMemcpyHostToDevice) -> copy from CPU to GPU
cudaFree(data);

dim3 dimBlock(x,y,z);
dim3 dimGrid(x,y,z);
someKernel<<<dimGrid, dimBlock>>>(d_data);

cudaMemAdvise(data, n*sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId); -> set some hints abt the data prefetch
cudaMemPrefetchAsync(data, n*sizeof(float), cudaGetDeviceId(&id)); -> prefetch data to GPU

cudaMemAdviseSetPreferredLocation or cudaMemAdviseSetReadMostly or cudaMemAdviseSetAccessedBy -> set hints for prefetching
readMostly -> such that cpu write don't inavlidate gpu one or copy it for read from cpu

___syncthreads() -> sync within a block
__shared__ float sdata[256]; -> shared memory

cudaDeviceSynchronize() -> tell CPU to wait for the kernel to finish
