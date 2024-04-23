/*
./CompareMulMatTiledShMem 
N = 64, TILE_WIDTH = 16
dimGrid.x = 4, dimGrid.y = 4
MulMatCPU			 elapsed time: 0.696287989616394043ms
MulMatTiledShMem		 elapsed time: 0.017408000305294991 ms
MulMatTiledShMemNoBankConflicts	 elapsed time: 0.007615999784320593 ms
simpleMatMulKernel		 elapsed time: 0.005024000070989132 ms
Results Matched!
h_C[0][0] = 87360
*/

#include <cuda_runtime.h>
#include <stdio.h>
#define TILE_WIDTH 16
void MulMatCPU(uint *A, uint *B, uint *C, uint N)
{
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

/* 
## Practice

 實作一簡易矩陣相乘的kernel函式，支援一維的
grid（即blockDim.x>=1), 並和本節其它矩陣相
乘方法所需時間作比較。
*/
__global__ void simpleMatMulKernel(uint *A, uint *Nd, uint *Pd, int Width) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= Width || ty >= Width) return;

    uint Pvalue = 0;
    for (int k = 0; k < Width; k++) {
        Pvalue += A[ty * Width + k] * Nd[k * Width + tx];
    }
    Pd[ty * Width + tx] = Pvalue;
}

__global__ void MulMatTiledShMem(uint *Md, uint *Nd, uint *Pd, int Width)
{
    __shared__ uint Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ uint Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    uint Pvalue = 0;
    // 反覆計算Pd[Row][Col]所需之Md 和Nd tiles的資料，並計算乘積的迴圈
    for (int m = 0; m < Width / TILE_WIDTH; m++)
    {
        Mds[ty][tx] = Md[Row * Width + (m * TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[Col + (m * TILE_WIDTH + ty) * Width];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) // 小塊矩陣相乘運算
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    Pd[Row * Width + Col] = Pvalue; // 將Pd[Row][Col]所需之各小塊結果累加得到最終結果
}

__global__ void MulMatTiledShMemNoBankConflicts(uint *Md, uint *Nd, uint *Pd, int Width)
{
    __shared__ uint Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ uint Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    uint Pvalue = 0;
    for (int m = 0; m < Width / TILE_WIDTH; m++)
    {
        Mds[ty][tx] = Md[Col * Width + (m * TILE_WIDTH + ty)];
        Nds[ty][tx] = Nd[Row + (m * TILE_WIDTH + tx) * Width];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            Pvalue += Mds[k][tx] * Nds[ty][k];
        __syncthreads();
    }
    Pd[Col * Width + Row] = Pvalue;
}

int main(int argc, char **argv)
{
    int N = 64;
    if (argc == 2)
        N = 1 << atoi(argv[1]);
    printf("N = %d, TILE_WIDTH = %d\n",
           N, TILE_WIDTH);
    cudaEvent_t time1, time2;
    float kernelExecutionTime;
    uint *h_A, *h_B, *h_C, *cpu_C;
    uint *d_A, *d_B, *d_C;

    h_A = (uint *)malloc(N * N * sizeof(uint));
    h_B = (uint *)malloc(N * N * sizeof(uint));
    h_C = (uint *)malloc(N * N * sizeof(uint));
    cpu_C = (uint *)malloc(N * N * sizeof(uint));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = i + j;
            h_B[i * N + j] = i + j + 1;
            h_C[i * N + j] = 0;
            cpu_C[i * N + j] = 0;
        }

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(N / TILE_WIDTH, N / TILE_WIDTH);

    printf("dimGrid.x = %d, dimGrid.y = %d\n",
           dimGrid.x, dimGrid.y);

    cudaEventCreate(&time1);
    cudaEventCreate(&time2);

    cudaEventRecord(time1, 0);
    MulMatCPU(h_A, h_B, cpu_C, N);
    cudaEventRecord(time2, 0);

    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime,
                         time1, time2);
    printf("MulMatCPU\t\t\t elapsed time: %20.18fms\n", kernelExecutionTime);
    
    cudaMalloc((void **)&d_A, N * N * sizeof(uint));
    cudaMalloc((void **)&d_B, N * N * sizeof(uint));
    cudaMalloc((void **)&d_C, N * N * sizeof(uint));

    cudaMemcpy(d_A, h_A, N * N * sizeof(uint),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(uint),
               cudaMemcpyHostToDevice);
    cudaEventRecord(time1, 0);
    
    MulMatTiledShMem<<<dimGrid,
                       dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(time2, 0);

    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);

    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("MulMatTiledShMem\t\t elapsed time: %20.18f ms\n", kernelExecutionTime);
    cudaMemcpy(h_C, d_C, N * N * sizeof(uint),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (h_C[i * N + j] != cpu_C[i * N + j])
            {
                printf("h_C[%d][%d]==%d != cpu_C[%d][%d]==%d\n", i, j, i, j, h_C[i * N + j], cpu_C[i * N + j]);
                return -1;
            }
    cudaEventRecord(time1, 0);
    MulMatTiledShMemNoBankConflicts<<<dimGrid,
                                      dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime,
                         time1, time2);
    printf("MulMatTiledShMemNoBankConflicts\t elapsed time: %20.18f ms\n", kernelExecutionTime);
    cudaMemcpy(h_C, d_C, N * N * sizeof(uint),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (h_C[i * N + j] != cpu_C[i * N + j])
            {
                printf("h_C[%d][%d]==%d != cpu_C[%d][%d]==%d\n", i, j, i, j, h_C[i * N + j], cpu_C[i * N + j]);
                return -1;
            }

    /*
        ## Practice

         實作一簡易矩陣相乘的kernel函式，支援一維的
        grid（即blockDim.x>=1), 並和本節其它矩陣相
        乘方法所需時間作比較。
    */
    dim3 dimBlockSimple(256);
    dim3 dimGridSimple((N + 255) / 256);

    cudaEventRecord(time1, 0);
    simpleMatMulKernel<<<dimGridSimple, dimBlockSimple>>>(d_A, d_B, d_C, N);
    cudaEventRecord(time2, 0);

    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("simpleMatMulKernel\t\t elapsed time: %20.18f ms\n", kernelExecutionTime);
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (h_C[i * N + j] != cpu_C[i * N + j])
            {
                printf("h_C[%d][%d]==%d != cpu_C[%d][%d]==%d\n", i, j, i, j, h_C[i * N + j], cpu_C[i * N + j]);
                return -1;
            }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    printf("Results Matched!\n");
    printf("h_C[0][0] = %d\n", h_C[0]);
    free(h_A);
    free(h_B);
    free(h_C);
    free(cpu_C);
    return 0;
}