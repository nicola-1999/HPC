all: jacobi-cuda jacobi-cuda-stride

FLAGS = -DGLX=$(GLX) -DGLY=$(GLY) -DMAXITER=$(MAXITER) -DDUMP=$(DUMP) -DDUMPSTEP=$(DUMPSTEP) -DUSE_DEBUG=$(USE_DEBUG)

CFLAGS = -Wall -Ofast -fopenmp -lm

jacobi-cuda : jacobi-cuda.cu
		nvcc $(FLAGS) -O3 -gencode arch=compute_35,code=sm_35 -Xcompiler -fopenmp -o jacobi-cuda jacobi-cuda.cu -lm

jacobi-cuda-stride : jacobi-cuda-stride.cu
		nvcc $(FLAGS) -O3 -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp -o jacobi-cuda-stride jacobi-cuda-stride.cu -lm

jacobi-cuda-check : jacobi-cuda-check.cu
		nvcc $(FLAGS) -O3 -gencode arch=compute_35,code=sm_35 -Xcompiler -fopenmp -o jacobi-cuda-check jacobi-cuda-check.cu -lm

clean:
		/bin/rm -f video-cuda-stride/* jacobi-cuda-stride