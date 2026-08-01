all: prova-hip jacobi-hip

GLX = 16384
GLY = 16384
THREADGRIDX = 64
THREADGRIDY = 1
BLOCKGRIDX = 64
BLOCKGRIDY = 256
DUMP = 0

FLAGS = -DGLX=$(GLX) -DGLY=$(GLY) -DDUMP=$(DUMP) -DTHREADGRIDX=$(THREADGRIDX) -DTHREADGRIDY=$(THREADGRIDY) -DBLOCKGRIDX=$(BLOCKGRIDX) -DBLOCKGRIDY=$(BLOCKGRIDY)

prova-hip : prova.cpp
		hipcc -o prova-hip prova.cpp

jacobi-hip : jacobi-amd.cpp
		hipcc $(FLAGS) -O3 -fopenmp -o jacobi-hip jacobi-amd.cpp -lm

clean:
		/bin/rm -f prova-hip jacobi-hip
		cd video; /bin/rm -f *