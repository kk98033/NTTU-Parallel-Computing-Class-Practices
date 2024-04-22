#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000
#define NUM_THREADS 4

unsigned long A[N][N], B[N][N], C[N][N];

struct thread_data {
    int start_row;
    int end_row;
};

void *threaded_mmul(void *threadarg) {
    struct thread_data *my_data;
    int i, j, k, start, end;
    
    my_data = (struct thread_data *) threadarg;
    start = my_data->start_row;
    end = my_data->end_row;

    for (i = start; i < end; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0;
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data_array[NUM_THREADS];
    int rc;
    long t;
    int rows_per_thread = N / NUM_THREADS;

    clock_t start, end;
    double cpu_time_used;

    // 初始化矩陣 A 和 B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i * j;
        }
    }

    start = clock();  // 開始計時

    for(t = 0; t < NUM_THREADS; t++) {
        thread_data_array[t].start_row = t * rows_per_thread;
        thread_data_array[t].end_row = (t + 1) * rows_per_thread;
        rc = pthread_create(&threads[t], NULL, threaded_mmul, (void *)&thread_data_array[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for(t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    end = clock();  // 結束計時
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // 計算耗時

    printf("Matrix multiplication completed.\n");
    printf("Total CPU time used: %.2f seconds\n", cpu_time_used);

    return 0;
}
