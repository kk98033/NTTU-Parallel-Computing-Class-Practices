#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>
#define DATAMB(bytes) (bytes / 1024 / 1024)
#define DATABW(bytes, timems) ((float)bytes / (timems * 1.024 * 1024.0 * 1024.0))
typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int ui;
uch *myimg, *dupimg;
uch *gpuimg, *gpudupimg, *gpuresult;
struct ImgInfo
{
    int hpxs;
    int vpxs;
    uch hdinfo[54];
    ul hbytes;
} inf;
#define INFB inf.hbytes
#define INFH inf.hpxs
#define INFV inf.vpxs
#define IMAGESIZE (INFB * INFV)
#define IMAGEPIX (INFH * INFV)

// 垂直翻轉影像的Vrev kernel函式, 每個thread 只負責翻轉單一的(R,G,B)像素
__global__ void Vrev(uch *dstimg, uch *srcimg,
                     ui hpxs, ui vpxs)
{
    ui ThrPerBlk = blockDim.x;
    ui bid = blockIdx.x;
    ui tid = threadIdx.x;
    ui gtid = ThrPerBlk * bid + tid;
    ui BlkPerRow = (hpxs + ThrPerBlk - 1) /
                   ThrPerBlk; // ceil
    ui RowBytes = (hpxs * 3 + 3) & (~3);
    ui row = bid / BlkPerRow;
    ui col = gtid - row * BlkPerRow * ThrPerBlk;
    if (col >= hpxs)
        return; // col out of range
    ui mirrorrow = vpxs - 1 - row;
    ui srcOffset = row * RowBytes;
    ui dstOffset = mirrorrow * RowBytes;
    ui srcIndex = srcOffset + 3 * col;
    ui dstIndex = dstOffset + 3 * col;
    // swap pixels RGB @col , @mirrorcol
    dstimg[dstIndex] = srcimg[srcIndex];
    dstimg[dstIndex + 1] = srcimg[srcIndex + 1];
    dstimg[dstIndex + 2] = srcimg[srcIndex + 2];
}

// 水平翻轉影像的Hrev kernel函式, 每個thread 只負責翻轉單一的(R,G,B)像素
__global__ void Hrev(uch *dstimg, uch *srcimg,
                     ui hpxs)
{
    ui ThrPerBlk = blockDim.x;
    ui bid = blockIdx.x;
    ui tid = threadIdx.x;
    ui gtid = ThrPerBlk * bid + tid;
    ui BlkPerRow = (hpxs + ThrPerBlk - 1) /
                   ThrPerBlk; // ceil
    ui RowBytes = (hpxs * 3 + 3) & (~3);
    ui row = bid / BlkPerRow;
    ui col = gtid - row * BlkPerRow * ThrPerBlk;
    if (col >= hpxs)
        return; // col out of range
    ui mirrorcol = hpxs - 1 - col;
    ui offset = row * RowBytes;
    ui srcIndex = offset + 3 * col;
    ui dstIndex = offset + 3 * mirrorcol;
    // swap pixels RGB @col , @mirrorcol
    dstimg[dstIndex] = srcimg[srcIndex];
    dstimg[dstIndex + 1] = srcimg[srcIndex + 1];
    dstimg[dstIndex + 2] = srcimg[srcIndex + 2];
};

// 拷貝影像的Pixdup kernel函式
__global__ void Pixdup(uch *dstimg, uch *srcimg, ui FS)
{
    ui ThrPerBlk = blockDim.x;
    ui bid = blockIdx.x;
    ui tid = threadIdx.x;
    ui gtid = ThrPerBlk * bid + tid;
    if (gtid > FS)
        return; // outside the allocated memory
    dstimg[gtid] = srcimg[gtid];
}

// 讀取影像檔案成一維線性資料格式的
//*bmp_read_1D()函式
uch *bmp_read_1D(char *fn)
{
    static uch *img;
    FILE *f = fopen(fn, "rb");
    if (f == NULL)
    {
        printf("\n\n%s NOT FOUND\n\n", fn);
        exit(EXIT_FAILURE);
    }
    uch hdinfo[54];
    // read the 54-byte header
    fread(hdinfo, sizeof(uch), 54, f);
    // extract image height and width from header
    int width = *(int *)&hdinfo[18];
    inf.hpxs = width;
    int height = *(int *)&hdinfo[22];
    inf.vpxs = height;
    int RowBytes = (width * 3 + 3) & (~3);
    inf.hbytes = RowBytes;
    // save header for reuse
    memcpy(inf.hdinfo, hdinfo, 54);
    printf("\n Input File name: %17s (%u x %u) File Size=%u", fn, inf.hpxs, inf.vpxs,
           IMAGESIZE);
    // allocate memory to store
    // the main image (1-dimensional array)
    img = (uch *)malloc(IMAGESIZE);
    if (img == NULL)
        return img;
    // read the image from disk
    fread(img, sizeof(uch), IMAGESIZE, f);
    fclose(f);
    return img;
}

// 將一維線性資料格式的影像寫入檔案的bmp_write_1D()函式
void bmp_write_1D(uch *img, char *fn)
{
    FILE *f = fopen(fn, "wb");
    if (f == NULL)
    {
        printf("\n\nFILE CREATION ERROR: %s\n\n", fn);
        exit(1);
    }
    // write header
    fwrite(inf.hdinfo, sizeof(uch), 54, f);
    // write data
    fwrite(img, sizeof(uch), IMAGESIZE, f);
    printf("\nOutput File name: %17s (%u x %u) File Size=%u", fn, inf.hpxs, inf.vpxs, IMAGESIZE);
    fclose(f);
}

int main(int argc, char **argv)
{
    char Rev = 'h';
    float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime;
    cudaError_t cudaStatus, cudaStatus2;
    cudaEvent_t time1, time2, time3, time4;
    char InputFileName[255], OutputFileName[255], ProgName[255];
    ui BlkPerRow, ThrPerBlk = 256, NumBlocks, GPUDataTransfer;
    cudaDeviceProp GPUprop;
    ul SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
    char SupportedBlocks[100];
    strcpy(ProgName, "imrevG");
    switch (argc)
    {
    case 5:
        ThrPerBlk = atoi(argv[4]);
    case 4:
        Rev = tolower(argv[3][0]);
    case 3:
        strcpy(InputFileName, argv[1]);
        strcpy(OutputFileName, argv[2]);
        break;

    default:
        printf("\n\nUsage: %s InputFilename OutputFilename [h/v/c/t] [ThrPerBlk]", ProgName);
        printf("\n\nExample: %s pic.bmp out.bmp", ProgName);
        printf("\n\nExample: %s pic.bmp out.bmp h", ProgName);
        printf("\n\nExample: %s pic.bmp out.bmp v 128", ProgName);
        printf("\n\nh=horizontal flip, v=vertical flip, c=copy, t=Transpose image\n\n");
        exit(EXIT_FAILURE);
    }
    if ((Rev != 'h') && (Rev != 'v') && (Rev != 'c') && (Rev != 't'))
    {
        printf("Invalid flip option '%c'. Must be 'v','h', 't', or 'c'... \n", Rev);
        exit(EXIT_FAILURE);
    }
    if ((ThrPerBlk < 32) || (ThrPerBlk > 1024))
    {
        printf("Invalid ThrPerBlk option '%u'. Must be between 32 and 1024. \n", ThrPerBlk);
        exit(EXIT_FAILURE);
    }
    // 配置CPU記憶體以儲存輸入及輸出的影像資料
    myimg = bmp_read_1D(InputFileName); // 讀取輸入影像檔案到CPU記憶體

    if (myimg == NULL)
    {
        printf("Cannot allocate memory for the input image...\n");
        exit(EXIT_FAILURE);
    }
    dupimg = (uch *)malloc(IMAGESIZE); // 配置主記憶體給拷貝影像之用
    if (dupimg == NULL)
    {
        free(myimg);
        printf("Cannot allocate memory for the input image...\n");
        exit(EXIT_FAILURE);
    }
    int NumGPUs = 0;
    cudaGetDeviceCount(&NumGPUs); // 取得系統的GPU數量
    if (NumGPUs == 0)
    {
        printf("\nNo CUDA Device is available\n");
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaSetDevice(0); // 設定使用第0個GPU
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        exit(EXIT_FAILURE);
    }
    cudaGetDeviceProperties(&GPUprop, 0); // 取得第0個GPU的資源細節

    SupportedKBlocks = (ui)GPUprop.maxGridSize[0] * (ui)GPUprop.maxGridSize[1] *
                       (ui)GPUprop.maxGridSize[2] / 1024; // 能支援的最大GPU KBlocks數量
    SupportedMBlocks = SupportedKBlocks / 1024;           // 能支援的最大GPU MBlocks數量
    sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks >= 5) ? 'M' : 'K');
    MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock; // 每一Block能支援的最大Threads數量
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    cudaEventCreate(&time3);
    cudaEventCreate(&time4);
    cudaEventRecord(time1, 0); // GPU傳輸開始的時間戳記
    // 分別配置GPU記憶體給輸入及輸出的影像使用
    cudaStatus = cudaMalloc((void **)&gpuimg, IMAGESIZE);
    cudaStatus2 = cudaMalloc((void **)&gpudupimg, IMAGESIZE);
    if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess))
    {
        fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
        exit(EXIT_FAILURE);
    }

    // 將輸入資料從主記憶體拷貝到GPU的記憶體中.
    cudaStatus = cudaMemcpy(gpuimg, myimg, IMAGESIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy CPU to GPU failed!");
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(time2, 0); // Time stamp after the CPU --> GPU tfr is done
    BlkPerRow = (INFH + ThrPerBlk - 1) / ThrPerBlk;
    NumBlocks = INFV * BlkPerRow;
    switch (Rev)
    {
    // 水平翻轉
    case 'h':
        Hrev<<<NumBlocks, ThrPerBlk>>>(gpudupimg, gpuimg, INFH);
        gpuresult = gpudupimg;
        GPUDataTransfer = 2 * IMAGESIZE;
        break;
    // 垂直翻轉
    case 'v':
        Vrev<<<NumBlocks, ThrPerBlk>>>(gpudupimg, gpuimg, INFH, INFV);
        gpuresult = gpudupimg;
        GPUDataTransfer = 2 * IMAGESIZE;
        break;

    // 水平兼垂直翻轉
    case 't':
        Hrev<<<NumBlocks, ThrPerBlk>>>(gpudupimg, gpuimg, INFH);
        Vrev<<<NumBlocks, ThrPerBlk>>>(gpuimg, gpudupimg, INFH, INFV);
        gpuresult = gpuimg;
        GPUDataTransfer = 4 * IMAGESIZE;
        break;
    // 拷貝影像
    case 'c':
        NumBlocks = (IMAGESIZE + ThrPerBlk - 1) / ThrPerBlk;
        Pixdup<<<NumBlocks, ThrPerBlk>>>(gpudupimg, gpuimg, IMAGESIZE);
        gpuresult = gpudupimg;
        GPUDataTransfer = 2 * IMAGESIZE;
        break;
    }
    cudaStatus = cudaDeviceSynchronize(); // 等待kernel工作結束，並回傳任何可能的錯誤
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(time3, 0);

    // 將輸出資料從GPU的記憶體拷貝到主記憶體.
    cudaStatus = cudaMemcpy(dupimg, gpuresult, IMAGESIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy GPU to CPU failed!");
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(time4, 0);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventSynchronize(time3);
    cudaEventSynchronize(time4);
    cudaEventElapsedTime(&totalTime, time1, time4);
    cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
    cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
    cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

    cudaStatus = cudaDeviceSynchronize();
    // checkError(cudaGetLastError()); // screen for errors in kernel launches
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
        free(myimg);
        free(dupimg);
        exit(EXIT_FAILURE);
    }
    bmp_write_1D(dupimg, OutputFileName); // 將處理後的影像寫回硬碟檔案
    printf("\n\n--------------------------------------------------------------------------\n");
    printf("%s ComputeCapab=%d.%d [max %s blocks; %d thr/blk] \n", GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
    printf("--------------------------------------------------------------------------\n");
    printf("%s %s %s %c %u [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName,
           OutputFileName, Rev, ThrPerBlk, NumBlocks, BlkPerRow);
    printf("--------------------------------------------------------------------------\n");
    printf("CPU->GPU Transfer =%7.2f ms ... %4d MB ... %6.2f GB/s\n", tfrCPUtoGPU,
           DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrCPUtoGPU));
    printf("Kernel Execution =%7.2f ms ... %4d MB ... %6.2f GB/s\n", kernelExecutionTime,
           DATAMB(GPUDataTransfer), DATABW(GPUDataTransfer, kernelExecutionTime));

    printf("GPU->CPU Transfer =%7.2f ms ... %4d MB ... %6.2f GB/s\n", tfrGPUtoCPU,
           DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrGPUtoCPU));
    printf("--------------------------------------------------------------------------\n");
    printf("Total time elapsed =%7.2f ms %4d MB ... %6.2f GB/s\n", totalTime, DATAMB((2 * IMAGESIZE + GPUDataTransfer)), DATABW((2 * IMAGESIZE + GPUDataTransfer), totalTime));
    printf("--------------------------------------------------------------------------\n\n");
    // 釋放所使用的GPU記憶體及CPU記憶體，並刪除所有events
    cudaFree(gpuimg);
    cudaFree(gpudupimg);
    cudaEventDestroy(time1);
    cudaEventDestroy(time2);
    cudaEventDestroy(time3);
    cudaEventDestroy(time4);
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        free(myimg);
        free(dupimg);
        exit(EXIT_FAILURE);
    }
    free(myimg);
    free(dupimg);
    return (EXIT_SUCCESS);
}