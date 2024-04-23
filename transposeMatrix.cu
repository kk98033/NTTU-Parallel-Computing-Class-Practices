/*
./transposeMatrix
./transposeMatrix Starting...
sumMatrixOnGPU_2D1D_v2 elapsed 1.2646399736 ms
Original Matrix:
12.70  3.60 21.50 18.10 11.10 24.80  1.40 13.70  1.50  9.10 12.70 13.90 11.50 22.00 14.50 13.70 
15.40 20.00 10.00 21.00  6.70 22.80  4.00  9.80 21.20 21.60 16.40  9.80 12.20  3.50 22.30 24.90 
 7.10 18.20 17.50 18.20 17.40 18.90  6.30 19.00  2.50 19.00  7.30 14.00 15.40 21.80  2.10  5.30 
16.20 12.10  0.70 23.00  9.30  4.70  7.20  5.00  0.80 23.60 14.80 13.00  1.50 11.50 12.30  8.60 
 4.20  4.20  1.20 21.60 23.20  7.50 15.00  0.10  0.90 22.30 14.10 16.40 18.50 16.20 21.70  9.20 
 2.70 22.40  6.60 12.10  1.50 13.80 17.10  2.30 11.90  6.30 15.30 13.40 17.90  2.10 22.10 22.10 
 6.30 23.30 18.10  3.90  5.30  7.60  4.00  6.20  4.30 18.10 22.60 22.90  8.70 18.70  6.50 11.50 
15.50 13.10 23.60 17.10  1.30 15.10 19.40 13.20 21.40  9.20  1.10 13.70 11.30 23.20 10.20 17.60 
20.90  2.80 21.60  0.60 10.40  0.00  6.90 14.70 18.20  3.90 12.00  1.30 22.70 18.50 12.80 12.60 
 6.00 10.80  4.10  7.40  0.30 23.60 20.60 21.80  7.20 21.70  9.90 18.50 19.30 20.20 10.50 14.70 
23.00  6.50 15.30  7.80  6.60 22.20 22.50 24.80  0.60  9.00  0.50 23.30  1.90 13.40 10.30  8.00 
24.20 14.50 15.40 24.60 12.50 10.40 20.80 19.70  6.60  5.10 12.60  0.30 25.30 23.10 15.00 22.70 
 4.10  4.80  4.90 10.70  1.40  1.90  9.90  2.00 10.90 10.40 25.30 12.80 23.80 10.10 20.80 22.50 
24.60 10.60 21.50 11.50 21.10 16.70  5.60  2.10 21.80 18.20  2.40 21.60 15.70 17.50 18.70 19.80 
22.30 23.70  4.90 23.70  0.00 14.80  0.20 10.90 25.30 25.50 23.70 23.50 10.00 19.00 20.40  9.00 
 4.00 16.30 20.50 25.10  7.40  0.50  1.60  3.70 18.70  4.10 25.30  8.90 21.60 18.40  3.10 18.30 

Transposed Matrix:
12.70 15.40  7.10 16.20  4.20  2.70  6.30 15.50 20.90  6.00 23.00 24.20  4.10 24.60 22.30  4.00 
 3.60 20.00 18.20 12.10  4.20 22.40 23.30 13.10  2.80 10.80  6.50 14.50  4.80 10.60 23.70 16.30 
21.50 10.00 17.50  0.70  1.20  6.60 18.10 23.60 21.60  4.10 15.30 15.40  4.90 21.50  4.90 20.50 
18.10 21.00 18.20 23.00 21.60 12.10  3.90 17.10  0.60  7.40  7.80 24.60 10.70 11.50 23.70 25.10 
11.10  6.70 17.40  9.30 23.20  1.50  5.30  1.30 10.40  0.30  6.60 12.50  1.40 21.10  0.00  7.40 
24.80 22.80 18.90  4.70  7.50 13.80  7.60 15.10  0.00 23.60 22.20 10.40  1.90 16.70 14.80  0.50 
 1.40  4.00  6.30  7.20 15.00 17.10  4.00 19.40  6.90 20.60 22.50 20.80  9.90  5.60  0.20  1.60 
13.70  9.80 19.00  5.00  0.10  2.30  6.20 13.20 14.70 21.80 24.80 19.70  2.00  2.10 10.90  3.70 
 1.50 21.20  2.50  0.80  0.90 11.90  4.30 21.40 18.20  7.20  0.60  6.60 10.90 21.80 25.30 18.70 
 9.10 21.60 19.00 23.60 22.30  6.30 18.10  9.20  3.90 21.70  9.00  5.10 10.40 18.20 25.50  4.10 
12.70 16.40  7.30 14.80 14.10 15.30 22.60  1.10 12.00  9.90  0.50 12.60 25.30  2.40 23.70 25.30 
13.90  9.80 14.00 13.00 16.40 13.40 22.90 13.70  1.30 18.50 23.30  0.30 12.80 21.60 23.50  8.90 
11.50 12.20 15.40  1.50 18.50 17.90  8.70 11.30 22.70 19.30  1.90 25.30 23.80 15.70 10.00 21.60 
22.00  3.50 21.80 11.50 16.20  2.10 18.70 23.20 18.50 20.20 13.40 23.10 10.10 17.50 19.00 18.40 
14.50 22.30  2.10 12.30 21.70 22.10  6.50 10.20 12.80 10.50 10.30 15.00 20.80 18.70 20.40  3.10 
13.70 24.90  5.30  8.60  9.20 22.10 11.50 17.60 12.60 14.70  8.00 22.70 22.50 19.80  9.00 18.30 

*/

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
