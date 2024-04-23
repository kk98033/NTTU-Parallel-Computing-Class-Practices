#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

void printMatrix(float *C, const int nx, const int ny)
{
    float *ic = C;
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            printf("%5.2f ", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void transposeMatrixGPU(float *MatIn, float *MatOut, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix < nx && iy < ny)
    {
        unsigned int idx_in = iy * nx + ix;
        unsigned int idx_out = ix * ny + iy;
        MatOut[idx_out] = MatIn[idx_in];
    }
}

void initialData(float *ip, unsigned int size)
{
    time_t t;
    srand((unsigned int) time(&t));
    for (unsigned int i = 0; i < size; i++)
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

int main(int argc, char **argv)
{
    cudaEvent_t time1, time2;
    float kernelExecutionTime;
    printf("%s Starting...\n", argv[0]);

    int nx = 1 << 4; // smaller size for easier debugging and visualization
    int ny = 1 << 4;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *h_A, *h_C;
    h_A = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nxy);

    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    cudaEventCreate(&time1);
    cudaEventCreate(&time2);

    cudaEventRecord(time1, 0);
    transposeMatrixGPU<<<grid, block>>>(d_A, d_C, nx, ny);
    cudaEventRecord(time2, 0);
    

    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&kernelExecutionTime, time1, time2);

    printf("sumMatrixOnGPU_2D1D_v2 elapsed %7.10f ms\n", kernelExecutionTime);

    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    printf("Original Matrix:\n");
    printMatrix(h_A, nx, ny);
    printf("Transposed Matrix:\n");
    printMatrix(h_C, ny, nx);

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);

    return 0;
}
