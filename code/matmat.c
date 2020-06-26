#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAT_SIZE 3000

#define GIGA 1000000000.
#define NSEC 1000000000.

double* rnd_flt_matrix(int n, int m);
double* zeros_flt_matrix(int n, int m);
double time_elapsed(struct timespec a, struct timespec b);
void matmat(int lda, int ldb, int ldc, int n, int m, int p, 
            double* A, double* B, double* C);

int main(int argc, char const *argv[])
{
	// Strutture che memorizzano le matrici da moltiplicare
    double *A, *B, *C;
    double cpuGflops;
    float elapsed;

    struct timespec start_cpu, finish_cpu;

    long nop;

    // variabili eventi utilizzate per stimare i tempi 
    // di esecuzione sulla Gpu
    
    double cpuTime; //tempo di esecuzione con Cpu

	//allocazione matrici
    A = rnd_flt_matrix(MAT_SIZE, MAT_SIZE);
    B = rnd_flt_matrix(MAT_SIZE, MAT_SIZE);
    C = zeros_flt_matrix(MAT_SIZE, MAT_SIZE);
    C = zeros_flt_matrix(MAT_SIZE, MAT_SIZE);

    printf("Ordine matrici; Tempo CPU; GFlops GPU\n");

    for(int i = 10; i <= MAT_SIZE; i += 10)
    {
        clock_gettime(CLOCK_MONOTONIC, &start_cpu);
        //esegue il prodotto tra matrici sull'host
        matmat(MAT_SIZE, MAT_SIZE, MAT_SIZE, i, i, i, A, B, C);
        clock_gettime(CLOCK_MONOTONIC, &finish_cpu);
        cpuTime = time_elapsed(start_cpu, finish_cpu); 

        nop = 2 * pow(i, 3);

        cpuGflops = (nop / cpuTime) / GIGA;

        printf("%6d;  %5.2lf; %5.2lf\n",  i, cpuTime, cpuGflops);
    }


    free(A);
    free(B);
    free(C);

    return 0;
}

void matmat(int lda, int ldb, int ldc, int n, int m, int p, 
    double* A, double* B, double* C)
{
    for(int k = 0; k < n; ++k)
        for(int i = 0; i < m; ++i)        
            for(int j = 0; j < p; ++j)
                C[i*ldc+j] += A[i*lda+k] * B[k*ldb+j];
}


double* rnd_flt_matrix(int n, int m)
{   
    int size = n * m;  
    double *A = (double*) malloc(sizeof(double) * size);

    for(int i = 0; i < size; ++i)
        A[i] = (float) rand() / RAND_MAX;
    
    return A;
}

double* zeros_flt_matrix(int n, int m)
{
    return (double*) calloc(n * m, sizeof(double));
}

double time_elapsed(struct timespec start, struct timespec finish)
{
    double elapsed = (finish.tv_sec - start.tv_sec);
    return elapsed + (finish.tv_nsec - start.tv_nsec) / NSEC;
}
