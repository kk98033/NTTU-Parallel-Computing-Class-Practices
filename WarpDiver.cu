/*
./WarpDiver 
./WarpDiver using Device 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Data size 64 Execution Configure (block 64 grid 1)
mathKernel1 <<<    1   64 >>> elapsed 1.24090 msec
mathKernel2 <<<    1   64 >>> elapsed 0.01024 msec 
mathKernel3 <<<    1   64 >>> elapsed 0.00934 msec 

*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if (tid % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    bool ipred = (tid % 2 == 0);
    if (ipred)
        ia = 100.0f;
    else
        ;
    if (!ipred)
        ib = 200.0f;
    else
        ;
    c[tid] = ia + ib;
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaEvent_t time1, time2;
    float kernelExecutionTime;

    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);
    // set up data size
    int size = 64;
    int blocksize = 64;
    if (argc > 1)
        blocksize = atoi(argv[1]);
    if (argc > 2)
        size = atoi(argv[2]);
    printf("Data size %d ", size);
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float **)&d_C, nBytes);
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    // run kernel 1
    cudaEventRecord(time1, 0);
    mathKernel1<<<grid, block>>>(d_C);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("mathKernel1 <<< %4d %4d >>> elapsed %7.5f msec\n", grid.x, block.x, kernelExecutionTime);
    // run kernel 2
    cudaEventRecord(time1, 0);
    mathKernel2<<<grid, block>>>(d_C);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("mathKernel2 <<< %4d %4d >>> elapsed %7.5f msec \n", grid.x, block.x, kernelExecutionTime);

    // run kernel 3
    cudaEventRecord(time1, 0);
    mathKernel3<<<grid, block>>>(d_C);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("mathKernel3 <<< %4d %4d >>> elapsed %7.5f msec \n", grid.x, block.x, kernelExecutionTime);
    cudaFree(d_C);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}