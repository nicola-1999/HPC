#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "common.h"


void jacobi_update(double *grid, double *grid_new) {
    for (int i = HY; i < GY - HY; i++) { // - HY così non tocco mai i ghost borders
        for (int j = HX; j < GX - HX - K; j++) { // non aggiorna le colonne della piastra
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
    double L = GX * GY;
    double start_time, end_time;
    char filename[256];
    int iter;
    // allocation
    posix_memalign((void**)&grid, 4096, L*sizeof(double));
    posix_memalign((void**)&grid_new, 4096, L*sizeof(double));
    //initialization
    init(grid);
    memcpy(grid_new, grid, GX*GY*sizeof(double)); // copy grid on grid-new
    dump(grid, "heatDiffusion_init.txt");
    #if DUMP == 1
        sprintf(filename, "video/grid-%07d",0);
        dump(grid, filename);
    #endif
    start_time = omp_get_wtime();
    for(iter = 1; iter <= MAXITER; iter++){
        //jacobi_update(...)
        jacobi_update(grid, grid_new);
        //dumpiter(...)
        #if DUMP == 1
        if(iter%DUMPSTEP == 0){
            sprintf(filename, "video/grid-%07d",iter);
            dump(grid_new, filename);
        }
        #endif
        //swappointers(...)
        swap_pointers(&grid, &grid_new); // scambia gli indirizzi di memoria &(*grid) dove sono memorizzati i vettori grid e grid_new
    } // end for iter

    end_time = (omp_get_wtime() - start_time) * 1.e3;
    // dump()
    // il risultato finale dopo lo swap resta in grid
    dump(grid, "heatDiffusion_final.txt");

    //statistic info
    printf("[statistics] %dx%d %d iter dT: %.2f msec dT/iter: %.2f usec GFLOPs: %.2f\n",
            GLX, GLY, MAXITER, end_time, (end_time*1000.0)/(double)MAXITER, 
            (5.0*(double)MAXITER*(double)GLX*(double)GLY)/(end_time*1e6));
    
    free(grid);
    free(grid_new);
    return 0;
}