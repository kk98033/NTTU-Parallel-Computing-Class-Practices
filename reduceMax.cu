#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
// Recursive CPU function of Interleaved Pair
int recursiveReduce(int *data, int const size)
{
    if (size == 1)
        return data[0]; // renew the stride
    int const stride = size / 2;
    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] = max(data[i], data[i + stride]);
    }
    return recursiveReduce(data, stride); // call recursively
}

// kernel 1: Neighbored pair with divergence
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 用輸入的陣列起始位址＋區塊位址偏移量來產生thread所屬區塊的起始位址
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n)
        return; // 邊界檢查
    // in-place reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
            idata[tid] = max(idata[tid], idata[tid + stride]);
        // 同一block內的threads同步，即先到者等待所有其它threads抵達
        __syncthreads();
    }
    // 將此block計算的結果寫到輸出陣列
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// kernel 2: Neighbored pair with less divergence
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 用輸入的陣列起始位址＋區塊位址偏移量來產生thread所屬區塊的起始位址
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n)
        return; // 邊界檢查
    // in-place reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // 將tid轉換成idata所需的索引
        int index = 2 * stride * tid;
        if (index < blockDim.x)
            idata[index] = max(idata[index], idata[index + stride]);
        // 同一block內的threads同步，即先到者等待所有其它threads抵達
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// kernel 3: Interleaved Pair with less divergence
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 用輸入的陣列起始位址＋區塊位址偏移量來產生thread所屬區塊的起始位址
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n)
        return; // 邊界檢查
    // in-place reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            idata[tid] = max(idata[tid], idata[tid + stride]);
        // 同一block內的threads同步，即先到者等待所有其它threads抵達
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// kernel 5: reduceUnrolling4 (迴圈展開法)
__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;
    // unrolling 4
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = max(max(a1, a2), max(a3, a4)); // Compare and find the max among four!!
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            idata[tid] = max(idata[tid], idata[tid + stride]);
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);
    bool bResult = false;

    // total number of elements to reduce
    int size = 1 << 24;
    printf("with array size %d ", size);

    int blocksize = 512; // initial block size
    if (argc > 1)
        blocksize = atoi(argv[1]);
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);
    
    // initialize the array
    for (int i = 0; i < size; i++)
        // mask off high 2 bytes to force max num to 255
        h_idata[i] = (int)(rand() & 0xFF);
    memcpy(tmp, h_idata, bytes);
    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce elapsed %7.3f msec cpu_sum: %d\n", 1000 * iElaps, cpu_sum);

    // Invoking kernel 1: reduceNeighbored with divergence
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 在此將kernel分別計算出的每個block總和再加總
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum = max(gpu_sum, h_odata[i]);
    printf("gpu Neighbored elapsed %7.3f msec gpu_sum: %d <<<grid %d block %d>>>\n",
           1000 * iElaps, gpu_sum, grid.x, block.x);

    // Invoking kernel 2: reduceNeighbored with less divergence
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum = max(gpu_sum, h_odata[i]);
    printf("gpu Neighbored2 elapsed %7.3f msec gpu_sum: %d <<<grid %d block %d>>>\n",
           1000 * iElaps, gpu_sum, grid.x, block.x);

    // Invoking kernel 3: reduceInterleaved with less divergence
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum = max(gpu_sum, h_odata[i]);
    printf("gpu Interleaved elapsed %7.3f msec gpu_sum: %d <<<grid %d block %d>>>\n",
           1000 * iElaps, gpu_sum, grid.x, block.x);

    // Invoking kernel 5: reduceUnrolling4
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost);

    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum = max(gpu_sum, h_odata[i]);
    printf("gpu Unrolling4 elapsed %7.3f msec gpu_sum: %d <<<grid %d block %d>>>\n",
           1000 * iElaps, gpu_sum, grid.x / 4, block.x);

    // Cleanup
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaDeviceReset();
    bResult = (gpu_sum == cpu_sum);
    if (!bResult)
        printf("Test failed!\n");
    return EXIT_SUCCESS;
}