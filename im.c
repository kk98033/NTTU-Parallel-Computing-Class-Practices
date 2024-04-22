#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "im.h"
// 影像檔讀取函式
unsigned char **bmp_read(char *fname)
{
    int i;
    unsigned char hdinfo[54];
    unsigned char tmp;
    unsigned char **myimage;
    FILE *f = fopen(fname, "rb");
    if (f == NULL)
    {
        printf("\n%s NOT FOUND\n", fname);
        exit(1);
    }
    // 讀取影像檔頭資訊
    fread(hdinfo, sizeof(unsigned char), 54, f);
    // 讀取影像寬和高
    int width = *(int *)&hdinfo[18];
    int height = *(int *)&hdinfo[22];
    // 複製檔頭資訊
    for (i = 0; i < 54; i++)
        inf.hdinfo[i] = hdinfo[i];
    inf.hpxs = width;
    inf.vpxs = height;
    int rowbytes = (width * 3 + 3) & (~3);
    inf.hbytes = rowbytes;
    printf("\n Input image file name: %20s (%u x %u)", fname, inf.hpxs, inf.vpxs);
    // 配置記憶體空間以準備讀取影像內容
    myimage = (unsigned char **)malloc(height * sizeof(unsigned char *));
    for (i = 0; i < height; i++)
        myimage[i] = (unsigned char *)malloc(rowbytes * sizeof(unsigned char));
    // 讀取影像檔內容到配置的記憶體空間
    for (i = 0; i < height; i++)
        fread(myimage[i], sizeof(unsigned char), rowbytes, f);
    fclose(f);
    return myimage;
}

// 影像檔寫入函式
void bmp_write(unsigned char **im, char *fname)
{
    unsigned long int x, y;
    char temp;
    FILE *f = fopen(fname, "wb");
    if (f == NULL)
    {
        printf("\nCREATING FILE ERROR: %s\n", fname);
        exit(1);
    }

    // 寫入影像檔頭資訊
    for (x = 0; x < 54; x++)
        fputc(inf.hdinfo[x], f);
    // 將影像內容寫入檔案
    for (x = 0; x < inf.vpxs; x++)
    {
        for (y = 0; y < inf.hbytes; y++)
        {
            temp = im[x][y];
            fputc(temp, f);
        }
    }
    printf("\n Output image file name: %20s (%u x %u)", fname, inf.hpxs, inf.vpxs);
    fclose(f);
}