#ifndef DRIVERS_CUH
#define DRIVERS_CUH

#include <cmath>
#include <chrono>
#include <stdio.h>

#include "kernels.cuh"
#include "utils.cuh"

#define BLOCKSIZE 32

void drive_mod_indexing(int n) {
  Matrix *A = build_matrix(n, n);
  Matrix *B = build_matrix(n, n);
  Matrix *C = build_matrix(n, n);

  float *A_d = A->data_device;
  float *B_d = B->data_device;
  float *C_d = C->data_device;

  dim3 block_dim(BLOCKSIZE * BLOCKSIZE);
  dim3 grid_dim(std::ceil((n + BLOCKSIZE - 1) / BLOCKSIZE), 
                std::ceil((n + BLOCKSIZE - 1) / BLOCKSIZE));

  auto start = std::chrono::high_resolution_clock::now();
  mod_indexing<BLOCKSIZE><<<grid_dim, block_dim>>>(A_d, B_d, C_d, n);
  cuda_error_check(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();

  double time = (stop - start).count() * 1e-9;
  long long nflops = (long long) n * n * n;
  printf("Mod indexing:\n    Time (s): %f\n    GFLOPs/s: %.4f\n", 
      time, nflops / time * 1e-9);

  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
}

void drive_normal_indexing(int n) {
  Matrix *A = build_matrix(n, n);
  Matrix *B = build_matrix(n, n);
  Matrix *C = build_matrix(n, n);

  float *A_d = A->data_device;
  float *B_d = B->data_device;
  float *C_d = C->data_device;

  dim3 block_dim(BLOCKSIZE, BLOCKSIZE);
  dim3 grid_dim(std::ceil((n + BLOCKSIZE - 1) / BLOCKSIZE), 
                std::ceil((n + BLOCKSIZE - 1) / BLOCKSIZE));

  auto start = std::chrono::high_resolution_clock::now();
  normal_indexing<BLOCKSIZE, BLOCKSIZE>
      <<<grid_dim, block_dim>>>(A_d, B_d, C_d, n);
  cuda_error_check(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();

  double time = (stop - start).count() * 1e-9;
  long long nflops = (long long) n * n * n;
  printf("Normal indexing:\n    Time (s): %f\n    GFLOPs/s: %.4f\n", 
      time, nflops / time * 1e-9);

  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
}

void drive_fast_indexing(int n) {
  Matrix *A = build_matrix(n, n);
  Matrix *B = build_matrix(n, n);
  Matrix *C = build_matrix(n, n);

  float *A_d = A->data_device;
  float *B_d = B->data_device;
  float *C_d = C->data_device;

  dim3 block_dim(BLOCKSIZE, BLOCKSIZE);
  dim3 grid_dim(std::ceil((n + BLOCKSIZE - 1) / BLOCKSIZE), 
                std::ceil((n + BLOCKSIZE - 1) / BLOCKSIZE));

  auto start = std::chrono::high_resolution_clock::now();
  fast_indexing<BLOCKSIZE, BLOCKSIZE>
      <<<grid_dim, block_dim>>>(A_d, B_d, C_d, n);
  cuda_error_check(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();

  double time = (stop - start).count() * 1e-9;
  long long nflops = (long long) n * n * n;
  printf("Fast indexing:\n    Time (s): %f\n    GFLOPs/s: %.4f\n", 
      time, nflops / time * 1e-9);

  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
}

void drive_shared_memory_tiling(int n) {
  Matrix *A = build_matrix(n, n);
  Matrix *B = build_matrix(n, n);
  Matrix *C = build_matrix(n, n);

  float *A_d = A->data_device;
  float *B_d = B->data_device;
  float *C_d = C->data_device;

  dim3 block_dim(BLOCKSIZE, BLOCKSIZE);
  dim3 grid_dim(std::ceil((n + BLOCKSIZE - 1) / BLOCKSIZE), 
                std::ceil((n + BLOCKSIZE - 1) / BLOCKSIZE));

  auto start = std::chrono::high_resolution_clock::now();
  shared_memory_tiling<BLOCKSIZE, BLOCKSIZE, BLOCKSIZE>
      <<<grid_dim, block_dim>>>(A_d, B_d, C_d, n);
  cuda_error_check(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();

  double time = (double) (stop - start).count() * 1e-9;
  long long nflops = (long long) n * n * n;
  printf("Shared memory tiling:\n    Time (s): %f\n    GFLOPs/s: %.4f\n", 
      time, nflops / time * 1e-9);

  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
}


void drive_thread_block_1d_tiling(int n) {
  Matrix *A = build_matrix(n, n);
  Matrix *B = build_matrix(n, n);
  Matrix *C = build_matrix(n, n);

  float *A_d = A->data_device;
  float *B_d = B->data_device;
  float *C_d = C->data_device;

  const int BM = 64;
  const int BN = 64;

  const int TM = 8;
  const int BK = BM / TM;

  dim3 block_dim(BM * BN / TM);

  // map fastest array axis -> fastest hardware axis
  dim3 grid_dim(std::ceil((n + BN - 1) / BN),  // blockIdx.x <=> cols of B
                std::ceil((n + BM - 1) / BM)); // blockIdx.y <=> rows of A

  auto start = std::chrono::high_resolution_clock::now();
  thread_block_1d<BM, BN, BK, TM>
      <<<grid_dim, block_dim>>>(A_d, B_d, C_d, n);
  cuda_error_check(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();

  double time = (double) (stop - start).count() * 1e-9;
  long long nflops = (long long) n * n * n;
  printf("1D threadblock tiling:\n    Time (s): %f\n    GFLOPs/s: %.4f\n", 
      time, nflops / time * 1e-9);

  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
}

void drive_thread_block_2d_tiling(int n) {
  Matrix *A = build_matrix(n, n);
  Matrix *B = build_matrix(n, n);
  Matrix *C = build_matrix(n, n);

  float *A_d = A->data_device;
  float *B_d = B->data_device;
  float *C_d = C->data_device;

  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  const int BM = 128;
  const int BN = 128;

  dim3 block_dim((BM / TM) * (BN / TN));

  // map fastest array axis -> fastest hardware axis
  dim3 grid_dim(std::ceil((n + BN - 1) / BN),  // blockIdx.x <=> cols of B
                std::ceil((n + BM - 1) / BM)); // blockIdx.y <=> rows of A

  auto start = std::chrono::high_resolution_clock::now();
  thread_block_2d<BM, BN, BK, TM, TN>
      <<<grid_dim, block_dim>>>(A_d, B_d, C_d, n);
  cuda_error_check(cudaDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();

  double time = (double) (stop - start).count() * 1e-9;
  long long nflops = (long long) n * n * n;
  printf("2D threadblock tiling:\n    Time (s): %f\n    GFLOPs/s: %.4f\n", 
      time, nflops / time * 1e-9);

  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
}
#endif
