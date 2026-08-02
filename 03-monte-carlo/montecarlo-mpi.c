#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

// Monte Carlo usa una seq. di numeri random, immaginiamoli come colpi che cadono dentro un quadrato
// all'iterno del quadrato c'è un cerchio, dobbiamo calcolare il rapporto tra la superficie (n° colpi) del quadrato e 
// la superficie del cerchio (n° colpi che cadono nella circ.) 
// è importante sapere 1) quanti colpi cadono dentro la circonferenza
//                      2) quanto è randomica la sequenza di colpi che generiamo
//
// Le coordinate dei colpi possono essere calcolate in modo indipendente ==> facilmente parallelizzabile 

int main(int argc, char *argv[]){

  int size_Of_Cluster, process_Rank;
  double x_value, y_value,pi;
  unsigned long long int hit, hit_tot, i;
  unsigned long long int n_darts_tot, n_darts_proc;

  //direttive 
  //MPI_COMM_WORLD tutti i processi in gioco fanno parte dello stesso Comunicator, NON ci sono sotto-gruppi
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);  // n° processi
  MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank); //id processo

  // caso AlltoOne

  n_darts_tot=(unsigned long long int)2048; // quanti? Nel caso MPI 
  n_darts_proc = n_darts_tot/size_Of_Cluster; //ogni processo avrà questo numero di colpi da poter sparare
  
  srand(42+process_Rank);//importante perché ogni processo esegue su una sequenza randomica diversa

  hit = 0;

  for (i=0;i<n_darts_proc;i++) {

    x_value = (double)(rand())/RAND_MAX;
    y_value = (double)(rand())/RAND_MAX;

    if (((x_value*x_value)+(y_value*y_value))<=1) hit++;
    // Reduce serve per calcolare il totale dei colpi che ha lanciato ogni processo
    // 1 elemento x ogni processo da sommare
    // 0 è il rank del processo che ottiene il risultato della reduce
    // send buffer => hit del singolo processo
    // receive buffer => hit totali
  }
  MPI_Reduce(&hit,&hit_tot,1,MPI_UNSIGNED_LONG,MPI_SUM,0,MPI_COMM_WORLD);
  pi = (double)(hit_tot*4)/(double)(n_darts_tot);

  // stampa solo il processo che ha i risultati della Reduce => rank 0
  if(process_Rank == 0){
    printf("There were %d hits in the circle \n", hit_tot);
    printf("The estimated value of pi is: " );
    printf("%.15f \n", pi);
    printf("The actual value is:          ");
    printf("%.15f... \n", M_PI );
  }

  MPI_Finalize();
  return 0;

}

