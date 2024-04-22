#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "im.h"
// 實驗回數
#define RNDS 1
struct ImgInfo inf;
// 水平翻轉影像函式
unsigned char **RevImgH(unsigned char **im)
{
    struct pixel px; // 資料交換用暫存空間
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
    return im;
}

// 垂直翻轉影像函式
unsigned char **RevImgV(unsigned char **im)
{
    struct pixel px; // 資料交換用暫存空間
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
    return im;
}

int main(int argc, char **argv)
{
    unsigned char **imdata;
    unsigned int k;
    double elaptime = 0;
    clock_t start, stop;
    if (argc != 4)
    {
        printf("\n\nUsage: imrev [input image file] [output image file] [h|v]");
        printf("\n\nExample: imrev pic.bmp pic_v.bmp v\n\n");
        return 0;
    }
    imdata = bmp_read(argv[1]);

    start = clock();
    switch (argv[3][0])
    {
    case 'v':
        for (k = 0; k < RNDS; k++)
            imdata = RevImgV(imdata);
        break;
    case 'h':
        for (k = 0; k < RNDS; k++)
            imdata = RevImgH(imdata);
        break;
    default:
        printf("\nInvalid Option!\n");
        return 0;
    }
    stop = clock();
    elaptime = 1000 * ((double)(stop - start)) / (double)CLOCKS_PER_SEC / (double)RNDS;
    // 寫入影像檔
    bmp_write(imdata, argv[2]);
    // 釋放記憶體
    for (int i = 0; i < inf.vpxs; i++)
    {
        free(imdata[i]);
    }
    free(imdata);
    printf("\n\nTotal elapsed time: %9.4f ms", elaptime);
    printf(" (%7.3f ns per pixel)\n", 1000000 * elaptime / (double)(inf.hpxs * inf.vpxs));
    return 0;
}