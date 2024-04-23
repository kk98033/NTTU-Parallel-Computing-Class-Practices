#include <stdio.h>
#include <cuda_runtime.h>
// Kernel definition
__global__ void VecAdd(float *A, float *B, float *C)
{
    int index = threadIdx.x;

    // 修改處
    extern __shared__ float shared[];
    float *s_A = shared;
    float *s_B = &shared[blockDim.x];
    s_A[index] = A[index];
    s_B[index] = B[index];
    __syncthreads();
    C[index] = s_A[index] + s_B[index];
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

    // 修改處 
    // Allocate pinned memory for host vectors
    cudaHostAlloc((void **)&h_A, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, N * sizeof(float), cudaHostAllocDefault);

    // 修改處
    // Initialize host memory
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 20.0f;
        h_B[i] = 30.0f;
    }
    
    // 修改處
    // Allocate memory for device vectors
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // 修改處
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&time1);
    cudaEventCreate(&time2);

    cudaEventRecord(time1, 0);

    // 修改處
    VecAdd<<<dimgrid, dimblock, 2 * dimblock.x * sizeof(float)>>>(d_A, d_B, d_C);
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

    // 修改處
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    return (0);
}