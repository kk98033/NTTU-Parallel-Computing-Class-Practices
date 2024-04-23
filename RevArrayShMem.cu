/*
./RevArrayShMem
n=1024
Kernel Elapsed time: 0.736256003379821777 ms
a[1023]=1023, a[1022]=1022, a[1021]=1021, 
d[1023]=  0, d[1022]=  1, d[1021]=  2, 
*/

#include <cuda_runtime.h>
#include <stdio.h>
__global__ void RevArrShMem(int *out, int *in)
{
    extern __shared__ int s[]; // 使用動態型共享記憶體
    int gid_in = blockDim.x * blockIdx.x + threadIdx.x;
    // 將輸入陣列的資料於各個block內作資料反轉，以儲存至shared memory
    s[blockDim.x - 1 - threadIdx.x] = in[gid_in];
    __syncthreads();
    // 以整個block為單位，反轉(交換)對應的block，以儲存至輸出陣列
    int gid_out = blockDim.x * (gridDim.x - 1 - blockIdx.x) + threadIdx.x;
    out[gid_out] = s[threadIdx.x];
}

__global__ void RevArr(int *out, int *in)
{
    // 一般未使用共享記憶體的陣列反轉
    int gid_in = blockDim.x * blockIdx.x + threadIdx.x;
    int out_offset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
    int gid_out = out_offset + (blockDim.x - 1 - threadIdx.x);
    // 反轉
    out[gid_out] = in[gid_in];
}

int main(int argc, char *argv[])
{
    // const int n = 1024;
    int n = 1024;
    if (argc == 2)
        n = 1 << atoi(argv[1]);
    printf("n=%d\n", n);
    int a[n], r[n], d[n];
    cudaEvent_t time1, time2;
    float kernelExecutionTime;
    dim3 block(512, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    int blocksize = block.x;
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        r[i] = n - i - 1;
        d[i] = 0;
    }
    int *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMemcpy(d_in, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(time1, 0);
    RevArrShMem<<<grid, block, blocksize * sizeof(int)>>>(d_out,
                                                          d_in);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("Kernel Elapsed time: %20.18f ms\n",
           kernelExecutionTime);
    cudaMemcpy(d, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        if (d[i] != n - i - 1)
            printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], n - i - 1);
    for (int i = n - 1; i > n - 4; i--)
        printf("a[%d]=%3d, ", i, a[i]);
    printf("\n");
    for (int i = n - 1; i > n - 4; i--)
        printf("d[%d]=%3d, ", i, d[i]);
    printf("\n");
    cudaFree(d_in);
    cudaFree(d_out);
}