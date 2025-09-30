#ifndef ARCH_BOUNDARIES_H
#define ARCH_BOUNDARIES_H

#include <stdio.h>
#include <cstdint>

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

__global__ void test_warps_per_sm(uint64_t *__restrict__ out, int niters) {
  /* This test attempts to extract the total number of warps per SM. This is
   * done by executing a grid with a single block and gradually increasing its
   * size while keeping work constant. 
   */ 
  extern __shared__ float s[];
  if (threadIdx.x < 10) s[threadIdx.x] = (float) threadIdx.x;

  uint64_t start = clock64();
  #pragma unroll 1
  for (int i = 0; i < niters; ++i)
    __syncthreads();
  uint64_t stop = clock64();

  if (threadIdx.x == 0)
    out[blockIdx.x] = (stop - start);  // store number of cycles
}

void test_warps_per_sm_driver() {
  const uint smemSize = 96 * 1024;

  uint64_t *out_h = new uint64_t[1024];
  uint64_t *out_d;
  CUDA_CHECK(cudaMalloc((void **)&out_d, sizeof(uint64_t) * 1024));
  CUDA_CHECK(
      cudaMemcpy(out_d, out_h, sizeof(uint64_t) * 1024, cudaMemcpyHostToDevice));

  // file for printing results to
  FILE *fp = fopen("warp-test-timings.txt", "w");
  dim3 gridSize(1);
  float ms = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int tbSize = 1; tbSize < 256; ++tbSize) {
    dim3 blockSize(tbSize);

    cudaFuncSetAttribute(test_warps_per_sm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);

    cudaEventRecord(start);
    test_warps_per_sm<<<gridSize, blockSize, smemSize>>>(out_d, 1000);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(
        cudaMemcpy(out_h, out_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cudaEventElapsedTime(&ms, start, stop);
    unsigned long long cycles = (unsigned long long) out_h[0];
    fprintf(stderr, "%d blocks took: %f ms, %llu cycles\n", tbSize, ms, cycles);
    fprintf(fp, "%d,%llu\n", tbSize, cycles);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  delete[] out_h;
  cudaFree(out_d);
  fclose(fp);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*-------------------------- PARTITIONS PER SM -------------------------------*/
/*----------------------------------------------------------------------------*/

__global__ void test_partitions_per_sm(float *result,
                                       uint64_t *__restrict__ out, int niters) {
  /* Now, we want to reveal the number of partitions (or warp schedulers) per
   * SM. Again, we'll focus on a *single* threadblock since we're poking and
   * prodding within a single SM. We'll likely measure this test case in cycles
   * instead of execution time once again. What are some things we know about
   * the partitions (schedulers)?
   *    1. Schedulers (on sm_86 and later) can *dual-issue* independent
   *    instructions every cycle to different execution pipelines
   *    2. Schedulers dispatch warps (chunks of 32 threads) to do some compute
   *    *as long as the warps are ready to execute*
   *
   * With those two things, we can already tell we're going to need to (1)
   * control for the dual-issue feature (this means either purposefully opting
   * in or forcefully avoiding). (2) we need to ensure that, at any point, our
   * warps are capable of executing. We'll avoid global memory in the "hot loop"
   * and we'll avoid dependent instructions.
   *
   * This has a bit more architectural details than the previous tests to get
   * working properly, so we'll go a little further in our explanation. The goal
   * here is to maximize the instruction level parallelism per warp. Said
   * differently, we need to make sure that as many instructions are "in-flight"
   * as possble. The easy part is that, for most cases, a scheduler can issue a
   * single instruction per cycle. Each of these instructions has a certain
   * latency (the number of cycles it takes to finish executing once issued).
   * Therefore, to keep the warps busy, we'll make sure we have at least L
   * instructions where L is the latency of whatever operation we plan to use.
   * From what I could find from a various Google searchs, FMAs have been 4
   * cycles since Turing (sm75). Meaning, we want at least 4 FMAs per iteration
   * in our work loop. We'll add some cushion and increase that to, say, 8 FMAs
   * per iteration to guarantee there's always something to do per cycle.
   *
   * Once we have the data, we find the point at which there is a distinct jump
   * in execution time and divide that by the warp size. On SM86, that should
   * happen at the first point measured after a threadblock size of 128.
   * Therefore, each SM has 128 / 32 = 4 partitions :). Also, for this
   * particular example we should notice that at a threadblock size of 128, we
   * should have the *lowest* number of cycles spent across the entire test
   * because we're fully saturating the SM.
   */
  float a0 = 2.03f;
  float a1 = 3.12f;
  float a2 = 4.38f;
  float a3 = 5.52f;
  float a4 = 6.41f;
  float a5 = 7.63f;
  float a6 = 8.81f;
  float a7 = 9.83f;

  float b0 = 3.03f;
  float b1 = 4.12f;
  float b2 = 5.38f;
  float b3 = 6.52f;
  float b4 = 7.41f;
  float b5 = 8.63f;
  float b6 = 9.81f;
  float b7 = 10.83f;

  uint64_t start = clock64();
  #pragma unroll 1
  for (int i = 0; i < niters; ++i) {
    a0 = fmaf(a0, b0, 1.0f);
    a1 = fmaf(a1, b1, 1.0f);
    a2 = fmaf(a2, b2, 1.0f);
    a3 = fmaf(a3, b3, 1.0f);
    a4 = fmaf(a4, b4, 1.0f);
    a5 = fmaf(a5, b5, 1.0f);
    a6 = fmaf(a6, b6, 1.0f);
    a7 = fmaf(a7, b7, 1.0f);
  }
  uint64_t stop = clock64();

  // stop optimizations
  float v = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (threadIdx.x == 0) {
    out[0] = stop - start;
    result[0] = v;
  }
}

void test_partitions_per_sm_driver() {
  uint64_t *out_h = new uint64_t[1024];
  uint64_t *out_d;
  CUDA_CHECK(cudaMalloc((void **)&out_d, sizeof(uint64_t) * 1024));

  float *result_h = new float[1024];
  float *result_d;
  CUDA_CHECK(cudaMalloc((void **)&result_d, sizeof(float) * 1024));

  // file for printing results to
  FILE *fp = fopen("warp-test-timings.txt", "w");
  dim3 gridSize(1);
  float ms = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int tbSize = 32; tbSize < 256; tbSize += 32) { // warp stride 
    dim3 blockSize(tbSize);

    cudaEventRecord(start);
    test_partitions_per_sm<<<gridSize, blockSize>>>(result_d, out_d, 1000);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(
        cudaMemcpy(out_h, out_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cudaEventElapsedTime(&ms, start, stop);
    unsigned long long cycles = (unsigned long long) out_h[0];
    fprintf(stderr, "%d blocks took: %f ms, %llu cycles\n", tbSize, ms, cycles);
    fprintf(fp, "%d,%llu\n", tbSize, cycles);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  delete[] out_h;
  cudaFree(out_d);
  fclose(fp);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

#endif
