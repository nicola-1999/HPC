#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

#define L (GX * GY)
#ifndef THREADGRIDX
#define THREADGRIDX 2
#endif 

#ifndef THREADGRIDY
#define THREADGRIDY 32
#endif 

#ifndef BLOCKGRIDX
#define BLOCKGRIDX 2048
#endif

#ifndef BLOCKGRIDY
#define BLOCKGRIDY 32
#endif
// blocco : 512 + 32 verso x, altezza 4 verso y  ... lancia la versione con stride anche al contrario
// se il blocco invece è 512 verso x e 8 verso y funziona?
void CUDACheckError(const char* action){
  cudaError_t error;
  error = cudaGetLastError();
  if(error != cudaSuccess){
    fprintf(
      stderr, "ERROR: Error while '%s': %s\n",
      action,
      cudaGetErrorString(error)
    );
    exit(-1);
  }
}

#ifdef USE_DEBUG
#define CUDA_CALL( call )       \
  call;                         \
  CUDACheckError( #call );
#else
#define CUDA_CALL( call )       \
  call;
#endif


__global__ void jacobi_update_cuda(double *grid, double *grid_new) {
    int tidx  = threadIdx.x;
    int tidy  = threadIdx.y;

    int bidx  = blockIdx.x;
    int bidy  = blockIdx.y;

    int bdimx = blockDim.x;
    int bdimy = blockDim.y;

    int gdimx = gridDim.x;
    int gdimy = gridDim.y;

    // Coordinate globali iniziali
    int i_start = bidx * bdimx + tidx; // r
    int j_start = bidy * bdimy + tidy; // c

    // Calcolo stride in entrambe le direzioni
    int stride_i = gdimx * bdimx;
    int stride_j = gdimy * bdimy;

    for (int i = i_start; i < GY; i += stride_i) { 
        for (int j = j_start; j < GX; j += stride_j) {

            // Escludiamo ghost borders e la piastra fissa a destra
            if (i >= HY && i < GY - HY && j >= HX && j < GX - HX - K) {
                int idx = i * GX + j;

                grid_new[idx] = (
                    grid[(i - 1) * GX + j] +   // sopra
                    grid[(i + 1) * GX + j] +   // sotto
                    grid[i * GX + (j - 1)] +   // sinistra
                    grid[i * GX + (j + 1)] +   // destra
                    grid[idx]                  // centro
                ) / 5.0;
            }
        }
    }
}



int main() {

    //declaration
    double *grid_h, *grid_new_h;
    double *grid_d, *grid_new_d;
    //double L = GX * GY;
    double start_time, end_time;
    char filename[256];
    int iter;
    // allocation on host
    posix_memalign((void**)&grid_h, 4096, L*sizeof(double));
    posix_memalign((void**)&grid_new_h, 4096, L*sizeof(double));

    //initialization
    init(grid_h);
    // checksum
    double checksum_init = 0.0;
    for (int i = HY; i < GY - HY; i++)
    {
        for(int j = HX; j < GX - HX; j++)
        {
            int idx = i * GX + j;
            checksum_init += grid_h[idx];
        }
    }
    memcpy(grid_new_h, grid_h, GX*GY*sizeof(double)); // copy grid on grid-new

    // define threadblock
    dim3 threadblock(THREADGRIDX,THREADGRIDY);
    //define gridblock 
    dim3 gridblock(BLOCKGRIDX, BLOCKGRIDY);

    // allocation on device 
    cudaError_t err;
    err = cudaMalloc((void**) &grid_d, L*sizeof(double));
    if (err != cudaSuccess){
        printf("Errore cudaMalloc su grid_d: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void**) &grid_new_d, L*sizeof(double));
    if (err != cudaSuccess){
        printf("Errore cudaMalloc su grid_new_d: %s\n", cudaGetErrorString(err));
    }

    // Dump initial condition
    //dump(grid_h, "heatDiffusion_init.txt");
    #if DUMP == 1
        sprintf(filename, "video-cuda-stride/grid-%07d",0);
        dump(grid_h, filename);
    #endif


     // copia dati sulla GPU
    CUDA_CALL((cudaMemcpy(grid_d, grid_h, L * sizeof(double), cudaMemcpyHostToDevice)));
    CUDA_CALL((cudaMemcpy(grid_new_d, grid_new_h, L * sizeof(double), cudaMemcpyHostToDevice)));

    //CUDA_CALL((jacobi_update_cuda <<<gridblock,threadblock>>> (grid_d,grid_new_d)))
    //CUDA_CALL((cudaMemcpy(grid_h, grid_new_d, L*sizeof(double), cudaMemcpyDeviceToHost)));
    //int myindex = 100;
    //double checksum_iter = 0.0;
    start_time = omp_get_wtime();
    for(iter = 1; iter <= MAXITER; iter += 2){
        //jacobi_update(...)
        CUDA_CALL((jacobi_update_cuda <<<gridblock,threadblock>>> (grid_d, grid_new_d)));
        CUDA_CALL(cudaDeviceSynchronize());// host capisce quando il kernel termina

        // #####################
        /*
        // copio la grid 
        CUDA_CALL((cudaMemcpy(grid_new_h, grid_new_d, L*sizeof(double), cudaMemcpyDeviceToHost)));

        if(myindex % 100 == 0){
            // checksum
            checksum_iter = 0.0;
            for (int i = HY; i < GY - HY; i++)
            {
                for(int j = HX; j < GX - HX; j++)
                {
                    int idx = i * GX + j;
                    checksum_iter += grid_new_h[idx];
                }
            }

            printf("Jacobi-cuda-stride, %d, %e, \n", myindex/100, checksum_iter);
            myindex = myindex + 100;
        }
        */
       
        //dumpiter(...)
        #if DUMP == 1
            if ((iter / 2) % (DUMPSTEP / 2) == 0){
                CUDA_CALL((cudaMemcpy(grid_new_h, grid_new_d, L*sizeof(double), cudaMemcpyDeviceToHost)));
                sprintf(filename, "video-cuda-stride/grid-%07d",iter);
                dump(grid_new_h, filename);
            }
        #endif
        //swappointers(...)
        CUDA_CALL(((jacobi_update_cuda <<<gridblock,threadblock>>> (grid_new_d, grid_d))));
        CUDA_CALL(cudaDeviceSynchronize());// host capisce quando il kernel termina
    } // end for iter

    end_time = (omp_get_wtime() - start_time) * 1.e3;
    CUDA_CALL((cudaMemcpy(grid_h, grid_new_d, L*sizeof(double), cudaMemcpyDeviceToHost)));

    
    // dump()
    // copio il risultato finale dal device
    //dump(grid_h, "heatDiffusion_final.txt");

      //////////////////////////////////////////////////////////
  // check data
  /*
    size_t nerr = 0;

    for (int j=HY; j < HY+GLY; j++) {
        for (int i=HX; i < HX+GLX; i++) {
        size_t idx = j*GX+i;
        int ref = ((j-HY)*GLX)+(i-HX);
        if (*(grid_h+idx) != ref) {
            if (nerr < 10) {
            fprintf( stderr, "M[%03d][%03d]: %d != %d\n", (i-HX), (j-HY), *(grid_h+idx), ref);
            nerr++;
            }
        }
        }
    }

    if (nerr == 0) {
        fprintf(stderr, "TEST SUCCESS: no errors found !\n");
    } else {
        fprintf(stderr, "TEST FAILED: %ld errors found !\n", nerr);
    }
    */

    // print config
    printf("Grid configuration: %dx%d blocks, %dx%d threads/block\n",
        BLOCKGRIDX, BLOCKGRIDY, THREADGRIDX, THREADGRIDY);
    printf("Total threads: %dx%d = %d\n",
        BLOCKGRIDX*THREADGRIDX, BLOCKGRIDY*THREADGRIDY,
        BLOCKGRIDX*THREADGRIDX*BLOCKGRIDY*THREADGRIDY);
    printf("Matrix size: %dx%d = %d elements\n", GLX, GLY, GLX*GLY);

    //statistic info
    double checksum = 0.0;
    for (int i = HY; i < GY - HY; i++)
    {
        for(int j = HX; j < GX - HX; j++)
        {
            int idx = i * GX + j;
            checksum += grid_h[idx];
        }
    }

    //statistic info
    printf("[statistics] %dx%d %d iter dT: %.2f msec dT/iter: %.2f usec GFLOPs: %.2f checksum_init: %e checksum_final: %e \n\n",
            GLX, GLY, MAXITER, end_time, (end_time*1000.0)/(double)MAXITER, 
            (5.0*(double)MAXITER*(double)GLX*(double)GLY)/(end_time*1e6),
            checksum_init,checksum);
    /*
    il numeratore, 5.0×MAXITER×GLX×GLY, è una stima semplificata del numero totale di operazioni in virgola mobile eseguite.
    Ciò suggerisce che, per ogni punto della griglia (GLX×GLY) e per ogni iterazione (MAXITER), l'algoritmo esegue circa 5 operazioni di aggiornamento della griglia, in virgola mobile.

    Il denominatore, end_time×10^6, converte il tempo totale di esecuzione da millisecondi a secondi,
    il che è necessario per ottenere il risultato finale in FLOPs al secondo.
    Il risultato finale viene poi scalato in GFLOPs.
    */
    CUDA_CALL(cudaFree(grid_d));
    CUDA_CALL(cudaFree(grid_new_d));
    free(grid_h);
    free(grid_new_h);
    return 0;
}

/*
### lattice 1024 x 1024 ###
    Best trial:
  Execution time: 362.0
  Params: 
    nthreads_x: 64
    nthreads_y: 1
    num_blocks_x: 512
    num_blocks_y: 512

### lattice 16384 x 16834 ###
    Best trial:
  Execution time: 118436.49
  Params: 
    nthreads_x: 64
    nthreads_y: 8
    num_blocks_x: 128
    num_blocks_y: 2048

## lattice 8192 x 8192 ##



*/