
all: mpi serial

mpi: 
	mpicc -O3 -I${MPIINC} -L${MPILIB} montecarlo-mpi.c -o montecarlo-mpi-pi

serial: 
	gcc -O3 montecarlo.c -o montecarlo-pi

clean: 
	rm -rf montecarlo-pi montecarlo-mpi-pi
