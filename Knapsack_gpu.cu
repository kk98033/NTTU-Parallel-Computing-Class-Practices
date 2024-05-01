#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void knapsack(int *d_val, int *d_weight, int *d_knapsack, int N, int W) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    if (n > 0 && n <= N && w <= W) {
        if (w == 0) {
            d_knapsack[n * (W + 1) + w] = 0;
        } else if (d_weight[n - 1] <= w) {
            d_knapsack[n * (W + 1) + w] = max(d_val[n - 1] + d_knapsack[(n - 1) * (W + 1) + (w - d_weight[n - 1])],
                                              d_knapsack[(n - 1) * (W + 1) + w]);
        } else {
            d_knapsack[n * (W + 1) + w] = d_knapsack[(n - 1) * (W + 1) + w];
        }
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s N W\n", argv[0]);
        return 1;
    }

    unsigned int N = atoi(argv[1]);
    unsigned int W = atoi(argv[2]);

    unsigned int *h_val = (unsigned int *)malloc(N * sizeof(unsigned int));
    unsigned int *h_weight = (unsigned int *)malloc(N * sizeof(unsigned int));
    unsigned int *h_knapsack = (unsigned int *)malloc((N + 1) * (W + 1) * sizeof(unsigned int));

    // Example values for val and weight
    h_val[0] = 60;
    h_val[1] = 100;
    h_val[2] = 120;

    h_weight[0] = 10;
    h_weight[1] = 20;
    h_weight[2] = 30;

    // Initialize knapsack array
    for (int i = 0; i < (N + 1) * (W + 1); i++) {
        h_knapsack[i] = 0;
    }

    int *d_val, *d_weight, *d_knapsack;
    cudaMalloc((void **)&d_val, N * sizeof(unsigned int));
    cudaMalloc((void **)&d_weight, N * sizeof(unsigned int));
    cudaMalloc((void **)&d_knapsack, (N + 1) * (W + 1) * sizeof(unsigned int));

    cudaMemcpy(d_val, h_val, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_knapsack, h_knapsack, (N + 1) * (W + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (W + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    knapsack<<<dimGrid, dimBlock>>>(d_val, d_weight, d_knapsack, N, W);
    cudaDeviceSynchronize();

    cudaMemcpy(h_knapsack, d_knapsack, (N + 1) * (W + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Max Value: %d\n", h_knapsack[N * (W + 1) + W]);

    free(h_val);
    free(h_weight);
    free(h_knapsack);
    cudaFree(d_val);
    cudaFree(d_weight);
    cudaFree(d_knapsack);

    return 0;
}
