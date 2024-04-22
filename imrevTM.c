#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "im.h"
#define RNDS 129
#define MAXTHRDS 128        // 執行緒數量上限
long NTHRDS;                // 執行緒個數
int ThPara[MAXTHRDS];       // 執行緒參數
pthread_t ThHdle[MAXTHRDS]; // 執行緒handler
pthread_attr_t ThAttr;      // 執行緒屬性
// 循序翻轉函數指標
void (*Revfunc)(unsigned char **img);
// 多執行緒翻轉函數指標
void *(*ThRevfunc)(void *arg);
unsigned char **myimg;
struct ImgInfo inf;
void RevImgH(unsigned char **im)
{
    struct pixel px;
    int row, col;
    for (row = 0; row < inf.vpxs; row++)
    {
        col = 0;
        while (col < (inf.hpxs * 3) / 2)
        {
            px.r = im[row][col + 2];
            px.g = im[row][col + 1];
            px.b = im[row][col];
            im[row][col + 2] = im[row][inf.hpxs * 3 - (col + 1)];
            im[row][col + 1] = im[row][inf.hpxs * 3 - (col + 2)];
            im[row][col] = im[row][inf.hpxs * 3 - (col + 3)];
            im[row][inf.hpxs * 3 - (col + 1)] = px.r;
            im[row][inf.hpxs * 3 - (col + 2)] = px.g;
            im[row][inf.hpxs * 3 - (col + 3)] = px.b;
            col += 3;
        }
    }
}
void RevImgV(unsigned char **im)
{
    struct pixel px;
    int row, col;
    for (col = 0; col < inf.hbytes; col += 3)
    {
        row = 0;
        while (row < inf.vpxs / 2)
        {
            px.r = im[row][col + 2];
            px.g = im[row][col + 1];
            px.b = im[row][col];
            im[row][col + 2] = im[inf.vpxs - (row + 1)][col + 2];
            im[row][col + 1] = im[inf.vpxs - (row + 1)][col + 1];
            im[row][col] = im[inf.vpxs - (row + 1)][col];
            im[inf.vpxs - (row + 1)][col + 2] = px.r;
            im[inf.vpxs - (row + 1)][col + 1] = px.g;
            im[inf.vpxs - (row + 1)][col] = px.b;
            row++;
        }
    }
}
// 水平翻轉影像執行緒函式
void *ThRevImgH(void *tid)
{
    struct pixel px;
    int row, col;
    long ts = *((int *)tid);              // 執行緒 ID
    ts = ts * inf.vpxs / NTHRDS;          // start index
    long te = ts + inf.vpxs / NTHRDS - 1; // end index
    for (row = ts; row <= te; row++)
    {
        col = 0;
        while (col < inf.hpxs * 3 / 2)
        {
            px.r = myimg[row][col + 2];
            px.g = myimg[row][col + 1];
            px.b = myimg[row][col];
            myimg[row][col + 2] = myimg[row][inf.hpxs * 3 - (col + 1)];
            myimg[row][col + 1] = myimg[row][inf.hpxs * 3 - (col + 2)];
            myimg[row][col] = myimg[row][inf.hpxs * 3 - (col + 3)];
            myimg[row][inf.hpxs * 3 - (col + 1)] = px.r;
            myimg[row][inf.hpxs * 3 - (col + 2)] = px.g;
            myimg[row][inf.hpxs * 3 - (col + 3)] = px.b;
            col += 3;
        }
    }
    pthread_exit(NULL);
}
// 垂直翻轉影像執行緒函式
void *ThRevImgV(void *tid)
{
    struct pixel px;
    int row, col;
    long ts = *((int *)tid);                // 執行緒 ID
    ts *= inf.hbytes / NTHRDS;              // start index
    long te = ts + inf.hbytes / NTHRDS - 1; // end index
    for (col = ts; col <= te; col += 3)
    {
        row = 0;
        while (row < inf.vpxs / 2)
        {
            px.r = myimg[row][col + 2];
            px.g = myimg[row][col + 1];
            px.b = myimg[row][col];
            myimg[row][col + 2] = myimg[inf.vpxs - (row + 1)][col + 2];
            myimg[row][col + 1] = myimg[inf.vpxs - (row + 1)][col + 1];
            myimg[row][col] = myimg[inf.vpxs - (row + 1)][col];
            myimg[inf.vpxs - (row + 1)][col + 2] = px.r;
            myimg[inf.vpxs - (row + 1)][col + 1] = px.g;
            myimg[inf.vpxs - (row + 1)][col] = px.b;
            row++;
        }
    }
    pthread_exit(0);
}

// 水平翻轉影像執行緒＋記憶體存取改良函式
void *ThRevImgHM(void *tid)
{
    struct pixel px;
    int row, col;
    unsigned char buf[16384]; // 存放一列像素的buffer
    /*由於此程式將會頻繁使用buf這塊資料，並且由於
    buf 的size小(只有16KB)，所以一段時間後buf會被系統
    自動載入到L1 cache中，之後每次存取buf資料時，即
    直接至L1 cache中存取，而不必再經由main memory */
    long ts = *((int *)tid);
    ts *= inf.vpxs / NTHRDS;
    long te = ts + inf.vpxs / NTHRDS - 1;
    for (row = ts; row <= te; row++)
    {
        memcpy((void *)buf, (void *)myimg[row], (size_t)inf.hbytes);
        col = 0;
        while (col < inf.hpxs * 3 / 2)
        {
            px.b = buf[col];
            px.g = buf[col + 1];
            px.r = buf[col + 2];
            buf[col] = buf[inf.hpxs * 3 - (col + 3)];
            buf[col + 1] = buf[inf.hpxs * 3 - (col + 2)];
            buf[col + 2] = buf[inf.hpxs * 3 - (col + 1)];
            buf[inf.hpxs * 3 - (col + 3)] = px.b;
            buf[inf.hpxs * 3 - (col + 2)] = px.g;
            buf[inf.hpxs * 3 - (col + 1)] = px.r;
            col += 3;
        }
        memcpy((void *)myimg[row],
               (void *)buf, (size_t)inf.hbytes);
    }
    pthread_exit(NULL);
}
// 垂直翻轉影像執行緒＋記憶體存取改良函式
void *ThRevImgVM(void *tid)
{
    struct pixel px;
    int row, row2, col;
    unsigned char buf[16384];  // 存放某一列
    unsigned char buf2[16384]; // 存放另一列
    long ts = *((int *)tid);
    ts *= inf.vpxs / NTHRDS / 2;
    long te = ts + (inf.vpxs / NTHRDS / 2) - 1;
    for (row = ts; row <= te; row++)
    {
        memcpy((void *)buf, (void *)myimg[row],
               (size_t)inf.hbytes);
        row2 = inf.vpxs - (row + 1);
        memcpy((void *)buf2, (void *)myimg[row2],
               (size_t)inf.hbytes);
        // 互換row與row2的資料
        memcpy((void *)myimg[row], (void *)buf2,
               (size_t)inf.hbytes);
        memcpy((void *)myimg[row2], (void *)buf,
               (size_t)inf.hbytes);
    }
    pthread_exit(NULL);
}
int main(int argc, char **argv)
{
    char Rev;
    int a, i, ThErr;
    struct timeval t;
    double start, end;
    double elaptime;
    char Revtype[50];
    switch (argc)
    {
    case 3:
        NTHRDS = 0;
        Rev = 'h';
        break;
    case 4:
        NTHRDS = 0;
        Rev = tolower(argv[3][0]);
        break;
    case 5:
        NTHRDS = atoi(argv[4]);
        Rev = tolower(argv[3][0]);
        break;
    default:
        printf("\n\nUsage: imrevTM input output [h|v|i|w] [0,1-128]");
        printf("\n\nUse 'h', 'v' for regular, and 'i', 'w' for the memory-friendly version of the program\n\n");
        printf("\n\nNTHRDS=0 for the serial version, and 1-128 for multi-threaded version\n\n");
        printf("\n\nExample: imrevTM pic.bmp out.bmp w8\n\n");
        printf("\n\nExample: imrevTM pic.bmp out.bmp v0\n\n");
        printf("\n\nNothing ... Exiting ...\n\n");
        exit(EXIT_FAILURE);
    }
    if ((NTHRDS < 0) || (NTHRDS > MAXTHRDS))
    {
        printf("\nNumber of threads must be between 0 and %u... \n", MAXTHRDS);
        printf("\n'1' means threads version with a single thread\n");
        printf("\nYou can specify '0' which means the 'serial' (non-threaded) version... \n\n");
        printf("\n\nNothing ... Exiting ...\n\n");
        exit(EXIT_FAILURE);
    }

    if (NTHRDS == 0)
        printf("\nExecuting the serial (non-threaded) version ...\n");
    else
        printf("\nExecuting the multi-threaded version with %li threads ...\n", NTHRDS);
    switch (Rev)
    {
    case 'h':
        ThRevfunc = ThRevImgH;
        Revfunc = RevImgH;
        strcpy(Revtype, "horizontal (h)");
        break;
    case 'v':
        ThRevfunc = ThRevImgV;
        Revfunc = RevImgV;
        strcpy(Revtype, "vertical (v)");
        break;
    case 'i':
        ThRevfunc = ThRevImgHM;
        Revfunc = RevImgH;
        strcpy(Revtype, "horizontal (i)");
        break;
    case 'w':
        ThRevfunc = ThRevImgVM;
        Revfunc = RevImgV;
        strcpy(Revtype, "vertical (w)");
        break;
    default:
        printf("Rev option '%c' is invalid. Can only be 'h', 'v', 'i', or 'w'\n", Rev);
        printf("\n\nNothing executed ... Exiting ...\n\n");
        exit(EXIT_FAILURE);
    }
    myimg = bmp_read(argv[1]);
    gettimeofday(&t, NULL);
    start = (double)t.tv_sec * 1000000.0 + ((double)t.tv_usec);

    if (NTHRDS > 0)
    {
        pthread_attr_init(&ThAttr);
        pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
        for (a = 0; a < RNDS; a++)
        {
            for (i = 0; i < NTHRDS; i++)
            {
                ThPara[i] = i;
                ThErr = pthread_create(&ThHdle[i], &ThAttr, ThRevfunc, (void *)&ThPara[i]);
                if (ThErr != 0)
                {
                    printf("\nThread Creation Error %d. Exiting abruptly... \n", ThErr);
                    exit(EXIT_FAILURE);
                }
            }
        }
        pthread_attr_destroy(&ThAttr);
        for (i = 0; i < NTHRDS; i++)
            pthread_join(ThHdle[i], NULL);
    }

    else
    {
        for (a = 0; a < RNDS; a++)
            (*Revfunc)(myimg);
    }
    gettimeofday(&t, NULL);
    end = (double)t.tv_sec * 1000000.0 + ((double)t.tv_usec);
    elaptime = (end - start) / 1000.00;
    elaptime /= (double)RNDS;
    bmp_write(myimg, argv[2]);
    for (i = 0; i < inf.vpxs; i++)
        free(myimg[i]);
    free(myimg);
    printf("\n\nTotal execution time: %9.4f ms. ", elaptime);
    if (NTHRDS > 1)
        printf("(%9.4f ms per thread). ", elaptime / (double)NTHRDS);
    printf("\n\nFlip Type = '%s'", Revtype);
    printf("\n (%6.3f ns/pixel)\n", 1000000 * elaptime / (double)(inf.hpxs * inf.vpxs));
    return (EXIT_SUCCESS);
}