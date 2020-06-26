#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define MAT_SIZE 10000

#define GIGA 1000000000.
#define NSEC 1000000000.

#define THREADS 32

__host__ cudaError_t matmatCuda(int, int, int, int, int, int, 
                                double *, double *, double *);
__global__ void productKernel(double*, double*, double*, int, int);
__global__ void adjustMatrix(double *, int, int);

double* rnd_flt_matrix(int n, int m);
double* zeros_flt_matrix(int n, int m);


int main(int argc, char const *argv[])
{
	// Strutture che memorizzano le matrici da moltiplicare
    double *A_h, *B_h, *C_d;
    double gflops_gpu;
    float elapsed;

    long nop;

    // variabili eventi utilizzate per stimare i tempi 
    // di esecuzione sulla Gpu
    cudaEvent_t start_gpu, stop_gpu;
    
    double gpuTime; //tempo di esecuzione con Gpu
    
    cudaError_t cudaStatus;

	//allocazione matrici
    A_h = rnd_flt_matrix(MAT_SIZE, MAT_SIZE);
    B_h = rnd_flt_matrix(MAT_SIZE, MAT_SIZE);
    C_d = zeros_flt_matrix(MAT_SIZE, MAT_SIZE);

    // Inzializza gli eventi di inizio e fine
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    printf("Ordine matrici; Tempo GPU; GFlops GPU\n");

    for(int i = 100; i <= MAT_SIZE; i += 100)
    {
        // Prende il tempo di inizio
        cudaEventRecord(start_gpu, 0);

        // inizializza ed esegue il prodotto tra matrici 
        // sul device @see matmatCuda
        cudaStatus = matmatCuda(MAT_SIZE, MAT_SIZE, MAT_SIZE, 
                                i, i, i, A_h, B_h, C_d);

        // Prende il tempo di fine
        cudaEventRecord(stop_gpu, 0);
        cudaEventSynchronize(stop_gpu);

        if(cudaStatus != cudaSuccess){
            fprintf(stderr, "matmatCuda failed!");
            return 1;
        }

        // Calcolo il tempo trascorso tra i 2 eventi
        cudaEventElapsedTime(&elapsed, start_gpu, stop_gpu);
        
        // Trasforma il tempo in secondi
        elapsed /= 1000.;

        // Prende il tempo di fine
        cudaEventRecord(stop_gpu, 0);
        cudaEventSynchronize(stop_gpu);
        
        // Calcolo il tempo trascorso tra i 2 eventi
        cudaEventElapsedTime(&elapsed, start_gpu, stop_gpu);

        // Trasforma il tempo in secondi
        gpuTime = elapsed / 1000.;
        
        nop = 2 * pow(i, 3);

        gflops_gpu = (nop / gpuTime) / GIGA;

        printf("%6d; %5.2lf; %5.2lf\n", i, gpuTime, gflops_gpu);
    }

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    free(A_h);
    free(B_h);
    free(C_d);

    return 0;
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

// procedura che inserisce il valore 0 
// per riempire gli spazi di matrice vuoti sulla cornice
__global__ void adjustMatrix(double *M, int n, int m)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x * i + j;

    if(i >= n || j >= m)        
        M[offset] = 0.;
}

cudaError_t matmatCuda(int lda, int ldb, int ldc, 
                       int n, int m, int p, double *A, double *B, double *C)
{
    double *A_d = NULL;
    double *B_d = NULL;
    double *C_d = NULL;
	
	// Calcola il numero di blocchi della griglia
    unsigned int nblock = (n + THREADS - 1) / THREADS;
    unsigned int mblock = (m + THREADS - 1) / THREADS;
    unsigned int pblock = (p + THREADS - 1) / THREADS;

	// Calcola la più piccola matrice quadrata multipla 
	// di THREADS che possa contenere la matrice A e B
    int nwidth = nblock * THREADS;
    int mwidth = mblock * THREADS;
    int pwidth = pblock * THREADS;

    cudaError_t cudaStatus;
    
    // Numero di blocchi per la griglia su A
    dim3 blocksA;
    blocksA.x = mblock;
    blocksA.y = nblock;

    // Numero di blocchi per la griglia su B
    dim3 blocksB;
    blocksB.x = pblock;
    blocksB.y = mblock;

	// Numero di blocchi per la griglia su C
	dim3 blocksC;
	blocksC.x = pblock;
	blocksC.y = nblock;
	
	// Numero di thread per blocco
    dim3 threads;
	threads.x = THREADS;
    threads.y = THREADS;
    
    // Scegli la GPU sul quale eseguire, in caso di un sistema multi-GPU.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "1: cudaSetDevice failed!\n");
        fprintf(stderr, "Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

	// **** Allocazione vettori all'interno della memoria del device ****
	
    cudaStatus = cudaMalloc((void**)&A_d, nwidth * mwidth * sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "2: cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&B_d, mwidth * pwidth * sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "3: cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&C_d, nwidth * pwidth * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "4: cudaMalloc failed!\n");
        goto Error;
    }
	// **** Termine allocazione dei vettori ****

	// **** Copia delle matrici in input nella memoria del device ****	
    cudaStatus = cudaMemcpy2D(A_d, mwidth * sizeof(double), 
                              A, lda * sizeof(double), 
                              m * sizeof(double), n, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "5: cudaMemcpy2D failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy2D(B_d, pwidth * sizeof(double), 
                              B, ldb * sizeof(double), p * sizeof(double),
                              m, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "6: cudaMemcpy2D failed!\n");
        goto Error;
    }
	
	// **** Termina copia dei vettori ****
	
	// Riempie la parte della matrice in eccesso con degli 0	
    if( (n % THREADS) || (m % THREADS) || (p % THREADS) ) {
		adjustMatrix<<<blocksA, threads>>>(A_d, n, m);
	    adjustMatrix<<<blocksB, threads>>>(B_d, m, p);
    }
    
	// Calcolo il prodotto tra le 2 matrici
    productKernel<<<blocksC, threads>>>(A_d, B_d, C_d, mwidth, pwidth);
	
	// Copia del vettore risultante dalla memoria del device a quella RAM
    cudaStatus = cudaMemcpy2D(C, ldc * sizeof(double), 
                              C_d, pwidth * sizeof(double), p * sizeof(double),
                              n, cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "7: cudaMemcpy2D failed!\n");
        goto Error;
    }

	
	// Deallocazione vettori ed eventi dalla memoria del device
    Error:
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);

    return cudaStatus;
}

 /**
 * Esegue il prodotto scalare tra  una riga di A e una colonna di B
 *
 * @param A vettore rapprensentante la matrice A
 * @param B vettore rapprensentante la matrice B
 * @param C vettore risulatante reappresentante la matrice C
 * @param width dimensione della matrice
 */
 __global__ void productKernel(double* A, double* B, double* C, int m, int p)
 {
    // id del blocco sull'ordinata all'interno della griglia
    int ib = blockIdx.y;
    // id del blocco sull'ascissa all'interno della griglia	
    int jb = blockIdx.x;
    // id del thread sull'ordinata all'interno del blocco	
    int it = threadIdx.y;
    // id del thread sull'ascissa allinterno del blocco 
    int jt = threadIdx.x;

    int a, b, c, k;

    // Indice della prima sottomatrice di A elaborata dal blocco
    // m e' un multiplo intero di THREADS
    // aBegin  include un certo numero ib  di gruppi di blocchi 
    // rettangolari THREADSxwidth
    int aBegin = m * THREADS * ib;

    //Indice dell'ultima sottomatrice di A elaborata dal blocco
    int aEnd   = aBegin + m - 1;

    // numero di colonne tra una sottomatrice e la successiva
    int aStep  = THREADS;

    // indice della prima sottomatrice di B elaborata dal blocco
    // bBegin include un certo numero jb di blocchi di colonne, 
    // blocchi larghi THREADS
    int bBegin = THREADS * jb;

    // numero di elementi tra una sottomatrice e la successiva
    int bStep  = THREADS * p;

    // Csub e' usata come variabile in cui memorizzare 
    // il valore dell'elemento di C calcolato dal thread
    // Viene aggiornato ripetutamente nel ciclo for seguente
    double Csub = 0;

    // Le matrici vengono divise in blocchi di dimensione THREADS X THREADS
    // per ridurre il numero di accessi alla memoria pricipale del device 
    // che risultano costosi in termini di tempo
    
    // Dichiarazione della variabile in cui salvare 
    // la sottomatrice di A in esame
    __shared__ double As[THREADS][THREADS];

    // Dichiarazione della variabile in cui salvare 
    // la sottomatrice di B in esame
    __shared__ double Bs[THREADS][THREADS];

    // Iterazione sulle sottomatrici 
    // in cui viene suddiviso il calcolo degli elementi del blocco
    for (a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        // Vengono Caricati gli elementi di ciascuna 
        // sottomatrice in memoria condivisa:
        // ogni thread del blocco carica un elemento!
            
        As[it][jt] = A[a +  m * it + jt];
        Bs[it][jt] = B[b +  p * it + jt];

        // i processi vengono sincronizzati per assicurare 
        // che ogni thread del blocco abbia caricato gli elementi.
        __syncthreads();

        // vengono calcolati i contributi agli elementi di matrice di C 
        // dovute alla sottomatrici in esame
        for( k = 0; k < THREADS ; ++k )
            Csub += As[it][k]*Bs[k][jt];
        // l'elemento C[it][jt] viene aggiornato in un numero di volte
        // pari al numero di iterazioni del ciclo for

        // i processi vengono sincronizzati per assicurare che il calcolo 
        // precedente sia terminato prima di caricare nuove
        // sottomatrici
        __syncthreads();
    }

    // vengono inseriti i risultati in C.
    // Ogni thread elabora un elemento di C.
    c = p * THREADS * ib + THREADS * jb;
    C[c +  p * it + jt] = Csub;
 }
 