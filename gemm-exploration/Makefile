CC=nvcc
CCFLAGS=-arch=sm_86 -O3

main: drivers.cuh kernels.cuh utils.cuh main.cu 
	$(CC) $(CCFLAGS) main.cu -o main

clean:
	rm *.o main
