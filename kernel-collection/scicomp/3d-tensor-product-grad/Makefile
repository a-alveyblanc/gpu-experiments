CC=nvcc
CCFLAGS=-O3 -arch=sm_86 -maxrregcount=64
HEADERS=constants.cuh kernels.cuh drivers.cuh utils.cuh

tp_contract: 3d-tensor-product-grad.cu $(HEADERS)
	$(CC) $(CCFLAGS) -o tp_contract 3d-tensor-product-grad.cu

clean:
	rm tp_contract
