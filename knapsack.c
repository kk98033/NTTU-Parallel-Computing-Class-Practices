#include <stdio.h>
#include <stdlib.h>
unsigned int max(unsigned int a, unsigned int b) { return (a > b) ? a : b; }
unsigned int knapsack(unsigned int W, unsigned int weight[], unsigned int val[], unsigned int N)
{
    unsigned int n, w;
    unsigned int K[N + 1][W + 1];
    for (n = 0; n <= N; n++)
    {
        for (w = 0; w <= W; w++)
        {
            if (n == 0 || w == 0)
                K[n][w] = 0;
            else if (weight[n - 1] <= w)
                K[n][w] = max(val[n - 1] + K[n - 1][w - weight[n - 1]], K[n - 1][w]);
            else
                K[n][w] = K[n - 1][w];
        }
    }
    return K[N][W];
}

int main(int argc, char *argv[])
{
    unsigned int val[] = {60, 100, 120};
    unsigned int weight[] = {10, 20, 30};
    unsigned int W = 50;
    unsigned int N = sizeof(val) / sizeof(val[0]);
    if (argc == 2)
        W = (unsigned int)(atoi(argv[1]));
    printf("W=%d, N=%d\n", W, N);
    printf("Best value = %d\n", knapsack(W, weight, val, N));
    return 0;
}