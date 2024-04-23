/*
./vectoradd
dimblock.x = 1024, dimblock.y = 1, dimblock.z = 1
dimgrid.x = 1, dimgrid.y = 1, dimgrid.z = 1
Using Device 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU
h_C[1023] ==   50.00
Kernel Execution =    0.09 ms

*/

#include <stdio.h>
#include <cuda_runtime.h>
// Kernel definition
__global__ void VecAdd(float *A, float *B, float *C)
{
    // 0D grid of 1D block 只有一個block; 整體資料只允許1024 Bytes; block為1維
    int index = threadIdx.x;
    // 0D grid of 2D blocks 只有一個block; 整體資料只允許1024 Bytes; block為2維
    // int index = threadIdx.y * blockDim.x + threadIdx.x;
    // 1D grid of 1D blocks
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    C[index] = A[index] + B[index];
}

int main()
{
    int N = 1 << 10;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    cudaEvent_t time1, time2;
    float kernelExecutionTime;
    // 0d grid of 1D block
    dim3 dimblock(1024, 1, 1);
    dim3 dimgrid(1, 1, 1);
    /* 0d grid of 2D block
    dim3 dimblock (16, 64, 1);
    dim3 dimgrid (1,1,1); */
    /* 1D grid of 1D block
    dim3 dimblock (1536,1);
    dim3 dimgrid ((N+dimblock.x-1)/dimblock.x, 1, 1);
    */
    printf("dimblock.x = %d, dimblock.y = %d, dimblock.z = %d\n", dimblock.x,
           dimblock.y, dimblock.z);
    printf("dimgrid.x = %d, dimgrid.y = %d, dimgrid.z = %d\n", dimgrid.x,
           dimgrid.y, dimgrid.z);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev,
           deviceProp.name);
    h_A = (float *)malloc(N * sizeof(float));
    h_B = (float *)malloc(N * sizeof(float));
    h_C = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 20.0f;
        h_B[i] = 30.0f;
        h_C[i] = 1.0f;
    }

    cudaMalloc((float **)&d_A, N * sizeof(float));
    cudaMalloc((float **)&d_B, N * sizeof(float));
    cudaMalloc((float **)&d_C, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    cudaEventRecord(time1, 0);
    VecAdd<<<dimgrid, dimblock>>>(d_A, d_B, d_C);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    printf("h_C[%d] == %7.2f\n", N - 1, h_C[N - 1]);
    printf("Kernel Execution = %7.2f ms\n", kernelExecutionTime);
    free(h_A);
    free(h_B);
    free(h_C);
    return (0);
}