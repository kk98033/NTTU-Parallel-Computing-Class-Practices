/*
./sumMatrix 
./sumMatrix Starting...
Using Device 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Matrix size: nx 1024 ny 1024
nxy=1048576
sumMatrixOnHost elapsed    2.40 ms
sumMatrixOnGPU_2D1D_v1 <<<(1024,1), (1,1024)>>> elapsed  19.322 ms
Results match.
sumMatrixOnGPU_2D1D_v2 <<<(1024,1), (1,1024)>>> elapsed  20.162 ms
Results match.
sumMatrixOnGPU_1D1D_v1 <<<(1024,1), (1024,1)>>> elapsed   0.077 ms
Results match.
sumMatrixOnGPU_1D1D_v2 <<<(1,1), (1024,1)>>> elapsed   0.372 ms
Results match.
sumMatrixOnGPU_2D2D_v1 <<<(32,32), (32,32)>>> elapsed   0.076 ms
Results match.
sumMatrixOnGPU_2D2D_v2 <<<(32,32), (32,32)>>> elapsed   0.093 ms
Results match.

*/

#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU_1D1D_v1(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    MatC[index] = MatA[index] + MatB[index];
}

__global__ void sumMatrixOnGPU_1D1D_v2(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

/* Practice: 新增2D grids of 1D block的v1與v2函式及相關的對應程式碼 */
__global__ void sumMatrixOnGPU_2DGrid1DBlock_v1(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < nx) {
        for (int iy = 0; iy < ny; iy++) {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

// 這個版本將使用類似的配置，但會對每個線程使用一個更嚴格的界限檢查，確保不會有任何越界錯誤。
__global__ void sumMatrixOnGPU_2DGrid1DBlock_v2(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= nx) return;  // Ensure ix is within bounds before proceeding.

    for (int iy = 0; iy < ny; iy++) {
        int idx = iy * nx + ix;
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}
/**/

__global__ void sumMatrixOnGPU_2D2D_v1(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int bIdx = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int index = bIdx * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    // unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // if (ix < nx && iy < ny)
    MatC[index] = MatA[index] + MatB[index];
}

__global__ void sumMatrixOnGPU_2D2D_v2(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

void initialData(float *ip, unsigned int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    for (unsigned int i = 0; i < size; i++)
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void checkResult(float *hostRef, float *gpuRef,
                 const int N)
{
    double epsilon = 1.0E-8;
    int match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Results do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Results match.\n");
    return;
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv)
{
    int dev = 0;
    double iStart = 0, iElaps = 0;
    cudaDeviceProp deviceProp;
    cudaEvent_t time1, time2;
    float kernelExecutionTime;
    printf("%s Starting...\n", argv[0]);
    
    // set up device
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // 使用者資料維度為nx * ny
    int nx = 1 << 10;
    int ny = 1 << 10;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // 初始化使用者資料
    printf("nxy=%d\n", nxy);
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    iStart = cpuSecond();

    // 執行CPU矩陣相加函式
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnHost elapsed %7.2f ms\n",
           iElaps * 1000);
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes,
               cudaMemcpyHostToDevice);
    int dimx11v1 = 1024;
    int dimy11v1 = 1;
    dim3 block11v1(dimx11v1, dimy11v1);
    dim3 grid11v1((nx * ny + block11v1.x - 1) / block11v1.x);
    int dimx11v2 = 1024;
    int dimy11v2 = 1;
    dim3 block11v2(dimx11v2, dimy11v2);
    dim3 grid11v2((nx + block11v2.x - 1) / block11v2.x);

    int dimx22v1 = 32;
    int dimy22v1 = 32;
    dim3 block22v1(dimx22v1, dimy22v1);
    dim3 grid22v1((nx + block22v1.x - 1) / block22v1.x, (ny + block22v1.y - 1) / block22v1.y);
    int dimx22v2 = 32;
    int dimy22v2 = 32;
    dim3 block22v2(dimx22v2, dimy22v2);
    dim3 grid22v2((nx + block22v2.x - 1) / block22v2.x, (ny + block22v2.y - 1) / block22v2.y);
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);

    // 2D 1D v1
    int threadsPerBlock = 1024;
    dim3 block2D1Dv1(threadsPerBlock);
    dim3 grid2D1Dv1((nx + block2D1Dv1.x - 1) / block2D1Dv1.x, ny);

    // 2D Grids of 1D Blocks v1
    cudaEventRecord(time1, 0);
    sumMatrixOnGPU_2DGrid1DBlock_v1<<<grid2D1Dv1, block2D1Dv1>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaEventRecord(time2, 0);

    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);

    printf("sumMatrixOnGPU_2D1D_v1 <<<(%d,%d), (%d,%d)>>> elapsed %7.3f ms\n",
           block2D1Dv1.x, block2D1Dv1.y, grid2D1Dv1.x, grid2D1Dv1.y, kernelExecutionTime);

    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nxy);


    // 2D 1D v2
    threadsPerBlock = 1024;
    dim3 block2D1Dv2(threadsPerBlock);
    dim3 grid2D1Dv2((nx + block2D1Dv2.x - 1) / block2D1Dv2.x, ny);

    // 2D Grids of 1D Blocks v2
    cudaEventRecord(time1, 0);
    sumMatrixOnGPU_2DGrid1DBlock_v2<<<grid2D1Dv2, block2D1Dv2>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaEventRecord(time2, 0);

    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);

    printf("sumMatrixOnGPU_2D1D_v2 <<<(%d,%d), (%d,%d)>>> elapsed %7.3f ms\n",
           block2D1Dv2.x, block2D1Dv2.y, grid2D1Dv2.x, grid2D1Dv2.y, kernelExecutionTime);

    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // 1D-1D v1
    cudaEventRecord(time1, 0);
    sumMatrixOnGPU_1D1D_v1<<<grid11v1, block11v1>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("sumMatrixOnGPU_1D1D_v1 <<<(%d,%d), (%d,%d)>>> elapsed %7.3f ms\n", grid11v1.x,
           grid11v1.y, block11v1.x, block11v1.y, kernelExecutionTime);
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // 1D-1D v2
    cudaEventRecord(time1, 0);
    sumMatrixOnGPU_1D1D_v2<<<grid11v2, block11v2>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("sumMatrixOnGPU_1D1D_v2 <<<(%d,%d), (%d,%d)>>> elapsed %7.3f ms\n",
           grid11v2.x, grid11v2.y, block11v2.x, block11v2.y, kernelExecutionTime);
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nxy);

    // 2D-2D v1
    cudaEventRecord(time1, 0);
    sumMatrixOnGPU_2D2D_v1<<<grid22v1, block22v1>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("sumMatrixOnGPU_2D2D_v1 <<<(%d,%d), (%d,%d)>>> elapsed %7.3f ms\n",
           grid22v1.x, grid22v1.y, block22v1.x, block22v1.y, kernelExecutionTime);
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nxy);

    // 2D-2D v2
    cudaEventRecord(time1, 0);
    sumMatrixOnGPU_2D2D_v2<<<grid22v2, block22v2>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("sumMatrixOnGPU_2D2D_v2 <<<(%d,%d), (%d,%d)>>> elapsed %7.3f ms\n",
           grid22v2.x, grid22v2.y, block22v2.x, block22v2.y, kernelExecutionTime);
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nxy);
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    cudaDeviceReset();
    return (0);
}