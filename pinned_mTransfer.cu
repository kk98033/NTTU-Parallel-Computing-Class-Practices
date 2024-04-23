/*
./pinned_mTransfer 
./pinned_mTransfer starting at device 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU memory size 4194304 nbyte 16.00MB

*/

#include <cuda_runtime.h>
#include <stdio.h>
int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaSetDevice(dev);
    // memory size
    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);
    // get device information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB\n", dev,
           deviceProp.name, isize, nbytes / (1024.0f * 1024.0f));
    // allocate the pinned memory
    float *h_aPinned;
    cudaError_t status =
        cudaMallocHost((void **)&h_aPinned, nbytes);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error returned from pinned host memory allocation\n");
        exit(1);
    } // allocate the device memory
    float *d_a;
    cudaMalloc((float **)&d_a, nbytes);
    // initialize the host memory
    for (unsigned int i = 0; i < isize; i++)
        h_aPinned[i] = 0.5f;
    // transfer data from the host to the device
    cudaMemcpy(d_a, h_aPinned, nbytes,
               cudaMemcpyHostToDevice);
    // transfer data from the device to the host
    cudaMemcpy(h_aPinned, d_a, nbytes,
               cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}