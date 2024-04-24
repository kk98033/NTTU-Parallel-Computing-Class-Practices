#include <stdio.h>
#include <time.h>
#include <pthread.h>

#define N 1000
#define NUM_THREADS 4

unsigned long A[N][N], B[N][N], C[N][N];

void init() {
    int i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = j * j;
            C[i][j] = 0;
        }
}

typedef struct {
    int start_row;
    int end_row;
} ThreadArg;

void* mmul_ikj_thread(void* arg) {
    int i, j, k;
    ThreadArg* t_arg = (ThreadArg*)arg;
    for (i = t_arg->start_row; i < t_arg->end_row; i++)
        for (k = 0; k < N; k++)
            for (j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];
    return NULL;
}

void mmul_ikj_multithreaded() {
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];
    int rows_per_thread = N / NUM_THREADS;
    int i;

    for (i = 0; i < NUM_THREADS; i++) {
        args[i].start_row = i * rows_per_thread;
        args[i].end_row = (i + 1) * rows_per_thread;
        pthread_create(&threads[i], NULL, mmul_ikj_thread, &args[i]);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

void run(void (*mmul)(), char *fname) {
    printf("========= %s ============\n", fname);
    time_t start, stop;
    init();
    start = time(NULL);
    mmul();
    stop = time(NULL);
    printf("elapsed time =%ld\n", stop - start);
}

int main() {
    run(mmul_ikj_multithreaded, "mmul_ikj_multithreaded");
}
