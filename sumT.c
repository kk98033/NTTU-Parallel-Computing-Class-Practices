#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

struct para
{
    int s;
    int e;
    int res;
};

unsigned int res1 = 0, res2 = 0, res3 = 0;
struct para th1_para, th2_para, th3_para;
unsigned int result = 0;

void *sum(struct para *p)
{
    int i;
    for (i = p->s; i <= p->e; i++)
        p->res += i;
}
int main()
{
    pthread_t th1, th2, th3;
    struct para th1_para = {1, 3500, 0};
    struct para th2_para = {3501, 7000, 0};
    struct para th3_para = {7001, 10000, 0};

    clock_t start, end;
    double multi_thread_time, single_thread_time;


    // Multi-threaded calculation
    start = clock();
    pthread_create(&th1, NULL, sum, &th1_para);
    pthread_create(&th2, NULL, sum, &th2_para);
    pthread_create(&th3, NULL, sum, &th3_para);

    pthread_join(th1, NULL);
    pthread_join(th2, NULL);
    pthread_join(th3, NULL);

    int multi_result = th1_para.res + th2_para.res + th3_para.res;
    end = clock();
    multi_thread_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Single-threaded calculation
    start = clock();
    int single_result = 0;
    for (int i = 1; i <= 10000; i++) {
        single_result += i;
    }
    end = clock();
    single_thread_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Printing results
    printf("Multi-threaded result = %d\n", multi_result);
    printf("Multi-threaded time: %f seconds\n", multi_thread_time);
    printf("Single-threaded result = %d\n", single_result);
    printf("Single-threaded time: %f seconds\n", single_thread_time);

    return 0;
}