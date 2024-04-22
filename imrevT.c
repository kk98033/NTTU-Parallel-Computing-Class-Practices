#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "im.h"
#define RNDS 1
#define MAXTHRDS 512        // 執行緒數量上限
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

int main(int argc, char **argv)
{
    char Rev;
    int a, i, ThErr;
    struct timeval t;
    double start, end, elaptime = 0;
    switch (argc)
    {
    case 3:
        NTHRDS = 1;
        Rev = 'h';
        break;
    case 4:
        NTHRDS = 1;
        Rev = tolower(argv[3][0]);
        break;
    case 5:
        NTHRDS = atoi(argv[4]);
        Rev = tolower(argv[3][0]);
        break;
    default:
        printf("\n\nUsage: imrevT input output [h/v] [thread count]");
        printf("\n\nExample: imrevT pic.bmp out.bmp h 8\n\n");
        return 0;
    }
    if ((Rev != 'h') && (Rev != 'v'))
    {
        printf("Reverse option '%c' is invalid. Can only be 'h' or 'v' ... Exiting...\n", Rev);
        exit(EXIT_FAILURE);
    }
    if ((NTHRDS < 1) || (NTHRDS > MAXTHRDS))
    {
        printf("\nNumber of threads must be between 1 and %u... Exiting\n", MAXTHRDS);
        exit(EXIT_FAILURE);
    }
    else
    {
        if (NTHRDS != 1)
        {
            printf("\nExecuting the multi-threaded version with %li threads ...\n", NTHRDS);
            ThRevfunc = (Rev == 'h') ? ThRevImgH : ThRevImgV;
        }
        else
        {
            printf("\nExecuting the serial version ...\n");
            Revfunc = (Rev == 'h') ? RevImgH : RevImgV;
        }
    }
    myimg = bmp_read(argv[1]);
    gettimeofday(&t, NULL);
    start = (double)t.tv_sec * 1000000.0 + ((double)t.tv_usec);
    if (NTHRDS > 1)
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
                    printf("\nThread Creation Error %d. Exiting... \n", ThErr);
                    exit(EXIT_FAILURE);
                }
            }
            pthread_attr_destroy(&ThAttr);
            for (i = 0; i < NTHRDS; i++)
                pthread_join(ThHdle[i], NULL);
        }
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
    {
        free(myimg[i]);
    }
    free(myimg);
    printf("\n\nTotal execution time: %9.4f ms (%s reverse)", elaptime, Rev == 'h' ? "horizontal" : "vertical");
    printf(" (%6.3f ns/pixel)\n", 1000000 * elaptime / (double)(inf.hpxs * inf.vpxs));
    return (EXIT_SUCCESS);
}