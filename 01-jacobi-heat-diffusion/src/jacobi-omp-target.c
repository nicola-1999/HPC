#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "common.h"
#define L (GX * GY)


void jacobi_update(double *grid, double *grid_new) {
    #pragma omp target teams distribute parallel for collapse(2)
    for (int i = HY; i < GY - HY; i++) { // - HY così non tocco mai i ghost borders
        for (int j = HX; j < GX - HX - K; j++) { // non aggiorno la piastra nelle K-colonne
            int index = i * GX + j; // corrisponde alle coordinate [i,j] nella matrix
       
            grid_new[index] = (
                grid[(i - 1) * GX + j] +     // sopra
                grid[(i + 1) * GX + j] +     // sotto
                grid[i * GX + (j - 1)] +     // sinistra 
                grid[i * GX + (j + 1)] +     // destra
                grid[index]                  // centro
            )/5;  
        }
    }
}

void swap_pointers(double **a, double **b) {
    double *tmp = *a;
    *a = *b;
    *b = tmp;
}

int main() {

    //declaration
    double *grid, *grid_new;
    //double L = GX * GY;
    double start_time, end_time;
    char filename[256];
    int iter;
    // allocation
    posix_memalign((void**)&grid, 4096, L*sizeof(double));
    posix_memalign((void**)&grid_new, 4096, L*sizeof(double));
    //initialization
    init(grid);
    memcpy(grid_new, grid, GX*GY*sizeof(double)); // copy grid on grid-new

    //OpenMP-target-version
    // Allocate and copy to device
    #pragma omp target enter data map(alloc: grid[0:L], grid_new[0:L])
    #pragma omp target update to(grid[0:L], grid_new[0:L])

    // Dump initial condition
    #pragma omp target update from(grid[0:L])
    dump(grid, "heatDiffusion_init.txt");
    #if DUMP == 1
        #pragma omp target update from(grid[0:L])
        sprintf(filename, "video-omp-target/grid-%07d",0);
        dump(grid, filename);
    #endif
    start_time = omp_get_wtime();

    for(iter = 1; iter <= MAXITER; iter += 2){
        //jacobi_update(...)
        jacobi_update(grid_new, grid);
        //dumpiter(...)
        #if DUMP == 1
            if((iter / 2) % (DUMPSTEP / 2) == 0){
                #pragma omp target update from(grid_new[0:L])
                sprintf(filename, "video-omp-target/grid-%07d",iter);
                dump(grid_new, filename);
            }
        #endif
        //swappointers(...)
        jacobi_update(grid, grid_new);
    } // end for iter

    // dump()
    // copio il risultato finale dal device
    end_time = (omp_get_wtime() - start_time) * 1.e3;
    #pragma omp target update from(grid[0:L])
    dump(grid, "heatDiffusion_final.txt");

    //statistic info
    printf("[statistics] %dx%d %d iter dT: %.2f msec dT/iter: %.2f usec GFLOPs: %.2f\n",
            GLX, GLY, MAXITER, end_time, (end_time*1000.0)/(double)MAXITER, 
            (5.0*(double)MAXITER*(double)GLX*(double)GLY)/(end_time*1e6));
    
    // Free device memory
    #pragma omp target exit data map(release: grid[0:L])
    #pragma omp target exit data map(release: grid_new[0:L])
 
    free(grid);
    free(grid_new);
    return 0;
}