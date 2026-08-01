#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include "common.h"
#define L (GX * GY)


#ifndef THREADGRIDX
#define THREADGRIDX 2
#endif 

#ifndef THREADGRIDY
#define THREADGRIDY 288
#endif 

#ifndef BLOCKGRIDX
#define BLOCKGRIDX 128
#endif

#ifndef BLOCKGRIDY
#define BLOCKGRIDY 1
#endif 


// https://rocm.docs.amd.com/projects/HIP/en/docs-develop/what_is_hip.html

//The HIPify tools, based on the clang front-end and Perl language,
// can convert CUDA API calls into the corresponding HIP API calls => hipify-clang file.cu -o file.cpp

// Macro di utilità per controllare gli errori HIP

// 1) optuna per trovare la miglior combinazione di threads e blocchi : fatto
// 2) dim_problema : 1024, 4096, 16.384, mantieni fisso il problema prima di lanciare gli esperimentti
// Non necessariamente devo avere 1 thread per ogni riga del lattice, posso variare la dimensione dei threads e del numero di blocchi con una potenza di 2
// cerca il numero minimo e massimimo di n° di blocchi in amd , in cuda circa 65.000
// 3) calcola la somma double degli elementi della griglia come checksum
// 4) Calcola tutte le coordinate dove vengono allocati i threads cuda sul lattice, si fa con un ciclo for, notare se più threads sono allocati nella stessa posizione del lattice, oppure in una zona sbagliata



#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t err = call;                                                   \
        if (err != hipSuccess) {                                                 \
            fprintf(stderr, "Errore HIP: %s, in %s, linea %d\n",                 \
                    hipGetErrorString(err), __FILE__, __LINE__);                 \
            exit(1);                                                             \
        }                                                                        \
    } while (0)


// grid-stride-loop kernel serve a gestire i casi in cui la dimensione del problema non è un multiplo esatto del numero di thread,
// grid-stride loop kernel, permette di elaborare l'intera griglia anche se i blocchi definiti inizialmente non mappano tutto il lattice
/*
_global__ void jacobi_update_amd(double *grid, double *grid_new) {
    int i_start = blockIdx.x * blockDim.x + threadIdx.x; // coordinate globali thread all'interno della griglia cuda, es : blocco 0 * dim_blocco 16 + thread 1 = 1
    int j_start = blockIdx.y * blockDim.y + threadIdx.y;

    // 1) step togli l'if , esegui il codice sotto con dim blocco 16x16, dim grid 32x32, dovrebbe eseguire solo la prima iterazione 
        int i = i_start;
        int j = j_start;
        // Check bounds
        if (i >= HY && i < GY - HY && j >= HX && j < GX - HX - K) {

            int idx = i * GX + j;
            grid_new[idx] = (
                grid[(i - 1) * GX + j] +   // sopra
                grid[(i + 1) * GX + j] +   // sotto
                grid[i * GX + (j - 1)] +   // sinistra 
                grid[i * GX + (j + 1)] +   // destra
                grid[idx]                // centro
            ) / 5.0;
        */
__global__ void jacobi_update_amd(double *grid, double *grid_new) {
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



// https://rocm.docs.amd.com/en/docs-6.2.1/reference/gpu-arch-specs.html
// GPU AMD AMD Instinct MI210 CDNA2
int main() {

    //int nthreads[] = {16,32,64,128,256,512};
    //for (int ti = 0; ti < 6; ti++)
    //{
        //int THREADS_PER_BLOCK = nthreads[ti];
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
        // var di controllo

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

        // Dump initial condition
        //dump(grid_h, "heatDiffusion_init.txt");
        #if DUMP == 1
            sprintf(filename, "video/grid-%07d",0);
            dump(grid_h, filename);
        #endif

        // define threadblock
        dim3 threadblock(THREADGRIDX,THREADGRIDY);
        //define gridblock 
        //dim3 gridblock(GX / threadblock.x, GY / threadblock.y);
        dim3 gridblock(BLOCKGRIDX, BLOCKGRIDY);

        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/
        // // Kernel invocation
        //dim3 threadsPerBlock(16, 16);
        //dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

        // allocation on device amd
        HIP_CHECK(hipMalloc((void**)&grid_d, L*sizeof(double)));
        HIP_CHECK(hipMalloc((void**)&grid_new_d, L*sizeof(double)));



        // copia dati sulla GPU amd
        HIP_CHECK(hipMemcpy(grid_d, grid_h, L * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(grid_new_d, grid_new_h, L * sizeof(double), hipMemcpyHostToDevice));
        start_time = omp_get_wtime();

        for(iter = 1; iter <= MAXITER; iter += 2){
            //jacobi_update(...)
            jacobi_update_amd <<<gridblock,threadblock>>> (grid_d, grid_new_d);
            HIP_CHECK(hipDeviceSynchronize());// host capisce quando il kernel termina
            //dumpiter(...)
            #if DUMP == 1
                //printf("scrittura del frame %d", iter);
                if ((iter / 2) % (DUMPSTEP / 2) == 0){
                    // Prima copio i dati dalla GPU (grid_new_d) alla CPU (grid_h)
                    HIP_CHECK(hipMemcpy(grid_new_h, grid_new_d, L * sizeof(double), hipMemcpyDeviceToHost));
                    //printf("scrittura del frame %d", iter);
                    sprintf(filename, "video/grid-%07d",iter);
                    //printf("[DEBUG] Scrivo frame %d in %s\n", iter, filename);
                    dump(grid_new_h, filename);
                }
            #endif
            //swappointers(...)
            jacobi_update_amd <<<gridblock,threadblock>>> (grid_new_d, grid_d);
            HIP_CHECK(hipDeviceSynchronize());// host capisce quando il kernel termina
        } // end for iter

        end_time = (omp_get_wtime() - start_time) * 1.e3;
        HIP_CHECK(hipMemcpy(grid_h, grid_new_d, L*sizeof(double), hipMemcpyDeviceToHost));
                            //dest    sorg
        // dump()
        // copio il risultato finale dal device
        //dump(grid_h, "heatDiffusion_final.txt");
        // calculate cheksum
        double checksum = 0.0;
        for (int i = HY; i < GY - HY; i++)
        {
            for(int j = HX; j < GX - HX; j++)
            {
                int idx = i * GX + j;
                checksum += grid_h[idx];
            }
        }


            // print config
        printf("Grid configuration: %dx%d blocks, %dx%d threads/block\n",
            BLOCKGRIDX, BLOCKGRIDY, THREADGRIDX, THREADGRIDY);
        printf("Total threads: %dx%d = %d\n",
            BLOCKGRIDX*THREADGRIDX, BLOCKGRIDY*THREADGRIDY,
            BLOCKGRIDX*THREADGRIDX*BLOCKGRIDY*THREADGRIDY);
        printf("Matrix size: %dx%d = %d elements\n", GLX, GLY, GLX*GLY);


        //statistic info
        printf("[statistics] %dx%d %d iter dT: %.2f msec dT/iter: %.2f usec GFLOPs: %.2f checksum_init: %e checksum_final: %e \n\n",
            GLX, GLY, MAXITER, end_time, (end_time*1000.0)/(double)MAXITER, 
            (5.0*(double)MAXITER*(double)GLX*(double)GLY)/(end_time*1e6),
            checksum_init,checksum);
        
        
        HIP_CHECK(hipFree(grid_d));
        HIP_CHECK(hipFree(grid_new_d));
        free(grid_h);
        free(grid_new_h);

    //}
        return 0;
}

// valuta il   checksum in formato esponenziale fino alla 15esima cifra decimale
// risultati : checksum corretto: 11939245,99 => 1.1939245 * 10^7
            // cheksum versione stride 11.940.394,57 => 1.1940394 * 10^7

/* 
    16x16 ,   4096x4096    10000 iter dT: 6565.47 msec dT/iter: 656.55 usec GFLOPs: 127.77
    32x32 ,   4096x4096    10000 iter dT: 5615.38 msec dT/iter: 561.54 usec GFLOPs: 149.39
    64x64 ,   4096x4096    10000 iter dT: 46.10 msec   dT/iter: 4.61   usec GFLOPs: 18196.12
    128x128 , 4096x4096    10000 iter dT: 34.63 msec   dT/iter: 3.46   usec GFLOPs: 24221.49
    256x256 , 4096x4096    10000 iter dT: 46.09 msec   dT/iter: 4.61   usec GFLOPs: 18202.14
    512x512 , 4096x4096    10000 iter dT: 35.29 msec   dT/iter: 3.53   usec GFLOPs: 23767.92
*/

// risultati con optuna :
/*
    Optimization finished.
    Best trial:     1024x1024
    Value: 219.24
    Params: 
        nthreads: 488
        num_blocks: 91

    Best trial:     4096x4096
    Value: 250.42
    Params: 
        nthreads: 384
        num_blocks: 91
*/

/* 
    jacobi-amd kernel stride :

    grid 16834x16834 => 
    Optimization finished.
        Best trial:
        Value: 723.96
        Params: 
            nthreads_x: 64
            nthreads_y: 168
            num_blocks_x: 403
            num_blocks_y: 377
*/

