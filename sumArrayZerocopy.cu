#include <cuda_runtime.h>
#include <stdio.h>
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    return;
}
void initialData(float *ip, int size)
{
    int i;
    for (i = 0; i < size; i++)
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}
__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaSetDevice(dev);
    cudaEvent_t time1, time2, time3, time4;
    float kernelExecutionTime, MemcpyfromHtoDTime, MemcpyfromDtoHTime, TotalTime;
    double start, elapsed_time;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    // check if support mapped memory
    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    printf("Using Device %d: %s ", dev, deviceProp.name);
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    cudaEventCreate(&time3);
    cudaEventCreate(&time4);

    int ipower = 10; // set up data size of vectors
    if (argc > 1)
        ipower = atoi(argv[1]);
    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);
    if (ipower < 18)
    {
        printf("Vector size %d power %d nbytes %3.0f KB\n", nElem, ipower,
               (float)nBytes / (1024.0f));
    }
    else
    {
        printf("Vector size %d power %d nbytes %3.0f MB\n", nElem, ipower,
               (float)nBytes / (1024.0f * 1024.0f));
    }
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    cudaEventRecord(time1, 0);
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime,
                         time1, time2);
    printf("cpu computation: %20.5f ms\n",
           kernelExecutionTime);
    // 配置在device上的一般global memory變數
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);
    cudaEventRecord(time1, 0);
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(time2, 0);
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaEventRecord(time3, 0);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(time4, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventSynchronize(time3);
    cudaEventSynchronize(time4);
    cudaEventElapsedTime(&MemcpyfromHtoDTime, time1, time2);
    cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
    cudaEventElapsedTime(&MemcpyfromDtoHTime, time3, time4);
    cudaEventElapsedTime(&TotalTime, time1, time4);
    printf("Normal(Pageabled memory) H->D: %7.4f ms, kernel computation: %7.4f ms, D->H: %7.4f ms, TOTAL: %7.4f ms\n", MemcpyfromHtoDTime, kernelExecutionTime,
           MemcpyfromDtoHTime, TotalTime);

    // check device results
    checkResult(hostRef, gpuRef, nElem);
    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // free host memory
    free(h_A);
    free(h_B);
    free(gpuRef);
    // part 2: 使用zero-copy memory
    // 配置zero-copy memory
    cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped);
    cudaHostAlloc((void **)&gpuRef, nBytes, cudaHostAllocMapped);
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    // add at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // device pointer與host pointer的zero-copy對應
    cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
    cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);
    cudaHostGetDevicePointer((void **)&d_C, (void *)gpuRef, 0);
    cudaEventRecord(time1, 0);
    // 直接使用上述的device指標執行kernel函式，不用透過cudaMemcpy作h->d的資料傳送
    sumArraysZeroCopy<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaEventRecord(time2, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);
    printf("Zerocopy(Pinned memory) kernel computation: %7.4f ms\n", kernelExecutionTime);
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    // CHECK(cudaFreeHost(h_A));
    // CHECK(cudaFreeHost(h_B));
    // CHECK(cudaFreeHost(gpuRef));
    free(hostRef);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}