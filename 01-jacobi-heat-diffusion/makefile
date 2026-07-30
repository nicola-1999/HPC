all: jacobi-serial jacobi-omp jacobi-omp-target jacobi-cuda jacobi-mpi

CC = gcc

FLAGS = -DGLX=$(GLX) -DGLY=$(GLY) -DMAXITER=$(MAXITER) -DDUMP=$(DUMP) -DDUMPSTEP=$(DUMPSTEP)

CFLAGS = -Wall -Ofast -fopenmp -lm

jacobi-serial: jacobi-serial.c 
		$(CC) $(FLAGS) $(CFLAGS) -o jacobi-serial jacobi-serial.c 

jacobi-omp: jacobi-omp.c 
		$(CC) $(FLAGS) $(CFLAGS) -o jacobi-omp jacobi-omp.c 

jacobi-omp-target: jacobi-omp-target.c
		nvc $(FLAGS) -Minfo --ptxas-options=-v -mp=gpu -gpu=cc70 -o jacobi-omp-target jacobi-omp-target.c -lm

jacobi-cuda : jacobi-cuda.cu
		nvcc $(FLAGS) -O3 -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp -o jacobi-cuda jacobi-cuda.cu -lm
jacobi-mpi : jacobi-mpi.c
		mpicc $(FLAGS) $(CFLAGS) -o jacobi-mpi jacobi-mpi.c

clean:
		/bin/rm -f jacobi-serial jacobi-omp jacobi-omp-target jacobi-cuda jacobi-mpi