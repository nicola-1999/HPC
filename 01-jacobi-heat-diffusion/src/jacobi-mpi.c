#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "common.h"
#define L (GX * GY)
#define MPIROOT 0

// NX larghezza griglia locale
// NY altezza con halos griglia locale

void init_top_n_bottom_halos(double *lgrid, int NX, int NY)
{   
    int i;
    for(i = 0; i < NX; i++) // inizializzo il TOP halo
    {
        lgrid[i] = 0.0;
    }
    for(i = NX*(NY-1); i < NX*NY; i++) // inizializzo BOTTOM halo
    {
        lgrid[i] = 0.0;
    }
}

/*
    +----------------------------+  ← Rank 0
    | Top HALO      (riceve da rank1)
    | Dati interni  
    | Bottom HALO   (non aggiornato)
    +----------------------------+

    +----------------------------+  ← Rank 1
    | Top HALO      (riceve da rank2)
    | Dati interni
    | Bottom HALO   (riceve da rank0)
    +----------------------------+

    ...

    +----------------------------+  ← Rank N
    | Top HALO      (non aggiornato)
    | Dati interni
    | Bottom HALO   (riceve da rankN-1)
    +----------------------------+

*/

// commenti prof:
// la Sendrecv viene usata da tutti i processi eccetto per quelli di rank 0 e rank n-1, in cui in un caso devono solo spedire e nell'altro devono solo ricevere.
// prima vanno tutte le comunicazioni in una direzione per aggiornare solo il bottom-halo e poi fai quelle nella direzione opposta per aggiornare il top-halo.
void update_top_n_bottom_halos(double *lgrid, int NX, int NY, int rank, int nproc) {

    MPI_Status status;
    int previous = rank - 1;
    int next = rank + 1;
    int err;
    
    if (rank == 0)
    {
        previous = MPI_PROC_NULL;
    }

    if (rank == nproc-1)
    {
        next = MPI_PROC_NULL;
    }
    // comunicazione in direzione UP
    // ogni processo :
    err = MPI_Sendrecv(&lgrid[HY*NX], NX, MPI_DOUBLE, next, 10,  // spedisce prima riga interna al processo next
            &lgrid[NX*(NY-1)], NX, MPI_DOUBLE, previous, 10, // riceve nel bottom halo dal processo previous
            MPI_COMM_WORLD, &status);
    if(err != MPI_SUCCESS)
        printf("Errore MPI_Sendrecv %d , rank: %d , sorgente: %d con tag: %d\n", err,rank, status.MPI_SOURCE, status.MPI_TAG);
    
    // comunicazione in direzione DOWN
    // ogni processo :
    err = MPI_Sendrecv(&lgrid[NX * (NY-2)], NX, MPI_DOUBLE, previous, 20,  // spedisce peneultima riga al processo previous
            &lgrid[0], NX, MPI_DOUBLE, next, 20,                           // riceve su top halo dal processo next
            MPI_COMM_WORLD, &status);
    if(err != MPI_SUCCESS)
        printf("Errore MPI_Sendrecv %d , rank: %d , sorgente: %d con tag: %d\n", err,rank, status.MPI_SOURCE, status.MPI_TAG);

}



void jacobi_update(double *grid, double *grid_new, int NX, int NY) {
    for (int i = HY; i < NY - HY; i++) { // - HY così non tocco mai i ghost borders, nella grid globale, ma in quella locale non li devo considerare?
        for (int j = HX; j < NX - HX - K; j++) { // non aggiorno la piastra nelle K-colonne
            int index = i * NX + j; // corrisponde alle coordinate [i,j] nelle matrix
       
            grid_new[index] = (
                grid[(i - 1) * NX + j] +     // sopra
                grid[(i + 1) * NX + j] +     // sotto
                grid[i * NX + (j - 1)] +     // sinistra 
                grid[i * NX + (j + 1)] +     // destra
                grid[index]                  // centro
            )/5;  
        }
    }
}

void swap_pointers(double **a, double **b){
    double *tmp = *a;
    *a = *b;
    *b = tmp;
}

int main(int argc, char *argv[]){
    // declare variables 
    double *ggrid, *lgrid, *lgrid_new;
    int NX, NY, LL;
    double start_time, end_time;
    int iter;
    double dTmax_local, dTmax_global;
    int size_Of_Cluster, process_Rank;
    char filename[256];

    //printf("MPI_init \n");
    // MPI settings
    MPI_Init(&argc,&argv);
    // MPI_COMM_WORLD tutti i processi in gioco fanno parte dello stesso Comunicator, NON ci sono sotto-gruppi
    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster); // n° processi
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank); // id processo


      // each MPI processes work on a slice NX x NY of ggrid
    NX = GX ;// 512 + 2
    NY = HY + (GLY/size_Of_Cluster) + HY; // 130, caso nproc = 4
    LL = NX * NY; // dimensione local grid
    // allocation local grid
    posix_memalign((void**)&lgrid, 4096, LL*sizeof(double)); // griglia x ogni processo senza allocare halos
    posix_memalign((void**)&lgrid_new, 4096, LL*sizeof(double)); 

    // global grid = 514 * 514 = 264.196
    // local grid =  514 * 130 = 66.820 
    // initialization of the global grid
    if(process_Rank == MPIROOT){
        posix_memalign((void**)&ggrid, 4096, GX*GY*sizeof(double)); // global grid
        init(ggrid);
        //dump
        #if DUMP == 1
            sprintf(filename, "video-mpi/grid-%07d", 0);
            dump(ggrid, filename);
        #endif
    } 

    MPI_Barrier(MPI_COMM_WORLD); // just to start all togheter

    // rank0 scatters the lattice slice of the global ggrid to the local lattice lgrid of each MPI processes
    /* int MPI_Scatter(void *sendbuf, int sendcnt,
        MPI_Datatype sendtype,
        void *recvbuf, int recvcnt,
        MPI_Datatype recvtype, int root,
        MPI_Comm comm)  */
    // MPI_Scatter(X, chunk_size, MPI_FLOAT, X_chunk, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Scatter della griglia ma senza considerare gli halos, verranno trattati dopo
    // GX * HY => dimensione del bordo
    MPI_Scatter(ggrid + (GX*HY), (GX*GLY/ size_Of_Cluster), MPI_DOUBLE,  // sendbuf, NON invio il bordo superiore della griglia globale
                lgrid + (GX*HY), (GX*GLY/ size_Of_Cluster), MPI_DOUBLE, // recvbuf,  
                 0, MPI_COMM_WORLD);

    //MPI_Scatter(ggrid, LL, MPI_DOUBLE, lgrid_new, LL, MPI_DOUBLE, MPIROOT, MPI_COMM_WORLD);
    printf("myid: %d MPI_Scatter eseguito \n", process_Rank);
    //printf("Scatter eseguito");

    // at each iteration the top and bottom halos of each slice should be updated,
    // the rank0 updates only the top halo
    // the rankn updated only the bottom halo
    // all the other ranks update the top and the bottom halos
    init_top_n_bottom_halos(lgrid, NX, NY);

    //copy(lgrid_new, lgrid);
    memcpy(lgrid_new, lgrid, LL*sizeof(double)); // copy lgrid on lgrid-new


    MPI_Barrier(MPI_COMM_WORLD);  // just to start all togheter

    //start_time = omp_get_wtime();
    start_time = MPI_Wtime();

    // jacobi loop
    printf("myid: %d  Ready for Jacobi loop : \n", process_Rank);
    for (iter = 1; iter < MAXITER; iter++){
        update_top_n_bottom_halos(lgrid,NX,NY,process_Rank,size_Of_Cluster);
        jacobi_update(lgrid, lgrid_new,NX,NY);
        swap_pointers(&lgrid,&lgrid_new);
        #if DUMP == 1
            if(iter % DUMPSTEP == 0){
                MPI_Gather(lgrid + (GX*HY), (GX*GLY/size_Of_Cluster), MPI_DOUBLE,
                        ggrid + (GX*HY), (GX*GLY/size_Of_Cluster), MPI_DOUBLE,
                        MPIROOT, MPI_COMM_WORLD);
                
                if(process_Rank == MPIROOT){
                    sprintf(filename, "video-mpi/grid-%07d", iter);
                    dump(ggrid, filename);
                }
            }
        #endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //end_time = omp_get_wtime();
    end_time = MPI_Wtime();
    dTmax_local = (end_time -start_time) * 1e3; 

    //MPI_Gather(ggrid...);
    // ogni processo invia 514*(130-2) = 65.792 double ===> GX * GLY/Size_Of_Cluster
    //  MPI_Gather(Y_chunk, chunk_size, MPI_FLOAT, Y, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(lgrid+(GX*HY), (GX*GLY/ size_Of_Cluster), MPI_DOUBLE,  // sendbuf dei vari proc saltando gli halos
                 ggrid+(GX*HY), (GX*GLY/ size_Of_Cluster), MPI_DOUBLE, // recvbuf di rank 0
                  0, MPI_COMM_WORLD);
    printf("myid: %d Gather su rank0 completato\n", process_Rank);
    //MPI_Reduce(maxdT...);  Reduce per trovare il tempo massimo tra tutti i processi mpi
    MPI_Reduce(&dTmax_local, &dTmax_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    printf("myid: %d MPI Reduce sul tempo completato \n",process_Rank);

    //print_statistics(...);
    if(process_Rank == MPIROOT)
    {
        printf("[statistics] %dx%d %d iter   dT: %.2f msec   dTmax/iter: %.2f usec   GFLOPs: %.2f\n",
        GLX, GLY, MAXITER, dTmax_global, (dTmax_global * 1000.0) / (double)MAXITER,
        (5.0 * (double)MAXITER * (double)GLX * (double)GLY) / (dTmax_global * 1e6));
    }

     MPI_Barrier(MPI_COMM_WORLD);

    if(process_Rank == MPIROOT){
        free(ggrid);   
    }

    free(lgrid);
    free(lgrid_new);
    
    MPI_Finalize();
    
    return 0;
}