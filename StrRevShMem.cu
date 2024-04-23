/*
./StrRevShMem 
RevShmem elapsed time: 1.536000013351440430 ms
a[0]=  0, a[1]=  1, a[2]=  2, 
d[0]= 63, d[1]= 62, d[2]= 61, 

*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void RevShmem(int *d, int n)
{
    extern __shared__ int s[]; // 動態共享記憶體
    int t = threadIdx.x; // 定義每個執行緒的索引
    int tr = n - t - 1;  // 計算反轉的索引
    s[t] = d[t]; // 將全域記憶體的資料複製到共享記憶體
    __syncthreads(); // 確保所有數據都已寫入共享記憶體
    d[t] = s[tr]; // 從共享記憶體複製反轉後的資料回全域記憶體
}

int main(void)
{
    const int n = 64;
    int a[n], r[n], d[n];
    cudaEvent_t time1, time2;
    float kernelExecutionTime;
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        r[i] = n - i - 1;
        d[i] = 0;
    }
    int *d_d;
    cudaMalloc(&d_d, n * sizeof(int));
    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(time1, 0);
    // 動態共享記憶體
    RevShmem<<<1, n, n * sizeof(int)>>>(d_d, n);
    cudaEventRecord(time2, 0);
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("RevShmem elapsed time: %20.18f ms\n", kernelExecutionTime);
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
    for (int i = 0; i < 3; i++)
        printf("a[%d]=%3d, ", i, a[i]);
    printf("\n");
    for (int i = 0; i < 3; i++)
        printf("d[%d]=%3d, ", i, d[i]);
    printf("\n");

    cudaFree(d_d);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
