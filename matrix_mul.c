#include <stdio.h>
#include <time.h>
#define N 1000
unsigned long A[N][N], B[N][N], C[N][N];
void init()
{
    int i, j, k;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            A[i][j] = i + j;
            B[i][j] = j * j;
            C[i][j] = 0;
        }
}
void mmul_ijk()
{
    int i, j, k;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}
void mmul_ikj()
{
    int i, j, k;
    for (i = 0; i < N; i++)
        for (k = 0; k < N; k++)
            for (j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];
}
void mmul_jki()
{
    int i, j, k;
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
            for (i = 0; i < N; i++)
                C[i][j] += A[i][k] * B[k][j];
}

void run(void (*mmul)(), char *fname)
{
    printf("========= %s ============\n", fname);
    time_t start, stop;
    init();
    start = time(NULL);
    mmul();
    stop = time(NULL);
    printf("elapsed time =%ld\n", stop - start);
}
int main()
{
    run(mmul_ijk, "mmul_ijk");
    run(mmul_ikj, "mmul_ikj");
    run(mmul_jki, "mmul_jki");
}