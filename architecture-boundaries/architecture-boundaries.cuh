#ifndef ARCH_BOUNDARIES_H
#define ARCH_BOUNDARIES_H

#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = call;                                                      \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(e));                                          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

/*----------------------------------------------------------------------------*/
/*----------------------------- SM COUNT -------------------------------------*/
/*----------------------------------------------------------------------------*/

__global__ void test_sm_count(float *__restrict__ out, const int niters) {
  /* Kernel that forces 1 block per SM to reveal jumps in execution time at
   * multiples of the SM count on an NVIDIA GPU. Can be similarly adapted to
   * other GPUs using whatever abstraction of an SM/core is used there.
   *
   * Succinctly, we allocate enough resources so that only 1 block can
   * occupy an SM at a time */

  // useful variables
  const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  float a = 1.0;
  float b = 2.0;

  // touch shared memory, allocated by host as max smem size per SM
  extern __shared__ unsigned char smem[];

  // tell the compiler not to unroll this loop
  #pragma unroll 1
  for (int n = 0; n < niters-1; ++n) {
    // two instructions that are dependent but don't really do much
    a = a + b * 1.0f;
    b = b + 1.0001f * 0.999f;
  }

  // make output dependent on both variables to stop the compiler from
  // optimizing either of them out 
  if (gid == 0)
    out[gid] = a + b;
}

void test_sm_count_driver() {
  const uint tbSize = 128;
  dim3 blockSize(tbSize); // make block size match warp size 
  const uint smemSize = 96 * 1024;

  float *out_h = new float[1024];
  float *out_d;
  CUDA_CHECK(cudaMalloc((void **)&out_d, sizeof(float) * 1024));
  CUDA_CHECK(
      cudaMemcpy(out_d, out_h, sizeof(float) * 1024, cudaMemcpyHostToDevice));

  // file for printing results to
  FILE *fp = fopen("timings.txt", "w");
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms = 0.0f;
  for (int nblocks = 1; nblocks < 512; ++nblocks) {
    dim3 gridSize(nblocks);

    cudaFuncSetAttribute(test_sm_count,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);

    cudaEventRecord(start);
    test_sm_count<<<gridSize, blockSize, smemSize>>>(out_d, 10000);
    cudaEventRecord(stop);

    CUDA_CHECK(cudaEventSynchronize(stop));

    cudaEventElapsedTime(&ms, start, stop);
    fprintf(stderr, "%d blocks took: %f ms\n", nblocks, ms);
    fprintf(fp, "%d,%f\n", nblocks, ms);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  delete[] out_h;
  cudaFree(out_d);
  fclose(fp);
}

/*----------------------------------------------------------------------------*/
/*----------------------------- WARPS PER SM ---------------------------------*/
/*----------------------------------------------------------------------------*/

__global__ void test_warps_per_sm() {
}

void test_warps_per_sm_driver() {
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

#endif
