#ifndef _3D_GRADIENT_DRIVERS_CUH
#define _3D_GRADIENT_DRIVERS_CUH

#include "utils.cuh"
#include "kernels.cuh"
#include "constants.cuh"

template <class FP_T>
cudaError_t naive_driver(RankedTensor<FP_T> A, RankedTensor<FP_T> u,
                         RankedTensor<FP_T> grad, const int nelts,
                         const int ndofs_1d, const int dim, int niterations) {
  dim3 gridDim(nelts);
  dim3 blockDim(ndofs_1d, ndofs_1d, ndofs_1d);

  naive_kernel<<<gridDim, blockDim>>>(A.device_view, u.device_view,
                                      grad.device_view, nelts, ndofs_1d, dim);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < niterations; ++i)
    naive_kernel<<<gridDim, blockDim>>>(A.device_view, u.device_view,
                                        grad.device_view, nelts, ndofs_1d, dim);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  unsigned long long nflops = nelts*ndofs_1d*ndofs_1d*ndofs_1d*ndofs_1d;
  unsigned long long nbytes = sizeof(FP_T)*(
    nelts*ndofs_1d*ndofs_1d*ndofs_1d + ndofs_1d*ndofs_1d
  );
  Benchmark b(niterations, ms, nflops, nbytes);
  fprintf(stderr, "===== Naive kernel =====\n");
  fprintf(stderr, "Runtime: %.4f ms\n", ms / niterations);
  fprintf(stderr, "GFLOP/s: %.4f\n", b.gflops());
  fprintf(stderr, "Gbyte/s: %.4f\n", b.gb_per_sec());
  fprintf(stderr, "========================\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cudaGetLastError();
}

template <class FP_T>
cudaError_t shared_memory_driver(RankedTensor<FP_T> A, RankedTensor<FP_T> u,
                                 RankedTensor<FP_T> grad, const int nelts,
                                 const int ndofs_1d, const int dim,
                                 int niterations) {
  dim3 gridDim(nelts);
  dim3 blockDim(ndofs_1d, ndofs_1d);

  int smem_bytes = sizeof(FP_T) * 2 * ndofs_1d * ndofs_1d;

  shared_memory_kernel<<<gridDim, blockDim, smem_bytes>>>(
      A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < niterations; ++i)
    shared_memory_kernel<<<gridDim, blockDim, smem_bytes>>>(
        A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  unsigned long long nflops = nelts*ndofs_1d*ndofs_1d*ndofs_1d*ndofs_1d;
  unsigned long long nbytes = sizeof(FP_T)*(
    nelts*ndofs_1d*ndofs_1d*ndofs_1d + ndofs_1d*ndofs_1d
  );
  Benchmark b(niterations, ms, nflops, nbytes);
  fprintf(stderr, "===== Shared memory kernel =====\n");
  fprintf(stderr, "Runtime: %.4f ms\n", ms / niterations);
  fprintf(stderr, "GFLOP/s: %.4f\n", b.gflops());
  fprintf(stderr, "Gbyte/s: %.4f\n", b.gb_per_sec());
  fprintf(stderr, "================================\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cudaGetLastError();
}

template <class FP_T>
cudaError_t shared_memory_no_bank_conflicts_driver(
    RankedTensor<FP_T> A, RankedTensor<FP_T> u, RankedTensor<FP_T> grad,
    const int nelts, const int ndofs_1d, const int dim, int niterations) {
  dim3 gridDim(nelts);
  dim3 blockDim(ndofs_1d, ndofs_1d);

  int smem_bytes =
      sizeof(FP_T) * (ndofs_1d * ndofs_1d + ndofs_1d * (ndofs_1d + 1));

  shared_no_bank_conflicts<<<gridDim, blockDim, smem_bytes>>>(
      A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < niterations; ++i)
    shared_no_bank_conflicts<<<gridDim, blockDim, smem_bytes>>>(
        A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  unsigned long long nflops = nelts*ndofs_1d*ndofs_1d*ndofs_1d*ndofs_1d;
  unsigned long long nbytes = sizeof(FP_T)*(
    nelts*ndofs_1d*ndofs_1d*ndofs_1d + ndofs_1d*ndofs_1d
  );
  Benchmark b(niterations, ms, nflops, nbytes);

  fprintf(stderr, "===== Shared memory kernel, no conflicts =====\n");
  fprintf(stderr, "Runtime: %.4f ms\n", ms / niterations);
  fprintf(stderr, "GFLOP/s: %.4f\n", b.gflops());
  fprintf(stderr, "Gbyte/s: %.4f\n", b.gb_per_sec());
  fprintf(stderr, "==============================================\n");


  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cudaGetLastError();
}

template <class FP_T>
cudaError_t register_tiled_k_driver(RankedTensor<FP_T> A, RankedTensor<FP_T> u,
                             RankedTensor<FP_T> grad, const int nelts,
                             const int ndofs_1d, const int dim,
                             int niterations) {
  dim3 gridDim(nelts);
  dim3 blockDim(ndofs_1d, ndofs_1d);

  int smem_bytes =
      sizeof(FP_T) * ndofs_1d * ndofs_1d + ndofs_1d * (ndofs_1d + 1);

  register_tiled_k<<<gridDim, blockDim, smem_bytes>>>(
      A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < niterations; ++i)
    register_tiled_k<<<gridDim, blockDim, smem_bytes>>>(
        A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  unsigned long long nflops = nelts*ndofs_1d*ndofs_1d*ndofs_1d*ndofs_1d;
  unsigned long long nbytes = sizeof(FP_T)*(
    nelts*ndofs_1d*ndofs_1d*ndofs_1d + ndofs_1d*ndofs_1d
  );
  Benchmark b(niterations, ms, nflops, nbytes);
  fprintf(stderr, "===== Register tiled kernel =====\n");
  fprintf(stderr, "Runtime: %.4f ms\n", ms / niterations);
  fprintf(stderr, "GFLOP/s: %.4f\n", b.gflops());
  fprintf(stderr, "Gbyte/s: %.4f\n", b.gb_per_sec());
  fprintf(stderr, "=================================\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cudaGetLastError();
}

template <class FP_T>
cudaError_t
register_tiled_op_constant(RankedTensor<FP_T> A, RankedTensor<FP_T> u,
                           RankedTensor<FP_T> grad, const int nelts,
                           const int ndofs_1d, const int dim, int niterations) {
  dim3 gridDim(nelts);
  dim3 blockDim(ndofs_1d, ndofs_1d);

  int smem_bytes = sizeof(FP_T) * ndofs_1d * ndofs_1d;
  if (cudaError_t e =
          cudaMemcpyToSymbol(op_const<FP_T>, A.host_view, smem_bytes);
      e != cudaSuccess)
    fprintf(stderr, "cudaMemcpyToSymbolFailed: %s\n", cudaGetErrorString(e));

  register_tiled_const_A<<<gridDim, blockDim, smem_bytes>>>(
      u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < niterations; ++i)
    register_tiled_const_A<<<gridDim, blockDim, smem_bytes>>>(
        u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  unsigned long long nflops = nelts*ndofs_1d*ndofs_1d*ndofs_1d*ndofs_1d;
  unsigned long long nbytes = sizeof(FP_T)*(
    nelts*ndofs_1d*ndofs_1d*ndofs_1d + ndofs_1d*ndofs_1d
  );
  Benchmark b(niterations, ms, nflops, nbytes);
  fprintf(stderr, "===== Register tiled kernel, const A =====\n");
  fprintf(stderr, "Runtime: %.4f ms\n", ms / niterations);
  fprintf(stderr, "GFLOP/s: %.4f\n", b.gflops());
  fprintf(stderr, "Gbyte/s: %.4f\n", b.gb_per_sec());
  fprintf(stderr, "==========================================\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cudaGetLastError();
}

template <class FP_T>
cudaError_t register_tiled_multiple_elements_per_block_driver(
    RankedTensor<FP_T> A, RankedTensor<FP_T> u, RankedTensor<FP_T> grad,
    const int nelts, const int ndofs_1d, const int dim, int niterations) {
  // FIXME: make a configurable, command line arg
  int elts_per_block = 4;
  dim3 gridDim(std::ceil((nelts + elts_per_block - 1) / elts_per_block));
  dim3 blockDim(ndofs_1d, ndofs_1d, elts_per_block);

  int op_smem_bytes = sizeof(FP_T) * ndofs_1d * ndofs_1d;
  int u_slice_smem_bytes = sizeof(FP_T) * ndofs_1d * (ndofs_1d + 1);
  int smem_bytes = op_smem_bytes + elts_per_block * u_slice_smem_bytes;

  register_tiled_k_multiple_elements_per_block<<<gridDim, blockDim,
                                                 smem_bytes>>>(
      A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim,
      elts_per_block);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < niterations; ++i)
    register_tiled_k_multiple_elements_per_block<<<gridDim, blockDim,
                                                   smem_bytes>>>(
        A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim,
        elts_per_block);

  cudaEventRecord(stop);
  if (cudaError_t e = cudaEventSynchronize(stop); e != cudaSuccess) {
    fprintf(stderr, "%s", cudaGetErrorString(e));
  }
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  unsigned long long nflops = nelts*ndofs_1d*ndofs_1d*ndofs_1d*ndofs_1d;
  unsigned long long nbytes = sizeof(FP_T)*(
    nelts*ndofs_1d*ndofs_1d*ndofs_1d + ndofs_1d*ndofs_1d
  );
  Benchmark b(niterations, ms, nflops, nbytes);
  fprintf(stderr, "===== Register tiled kernel, %d elements per block =====\n",
      elts_per_block);
  fprintf(stderr, "Runtime: %.4f ms\n", ms / niterations);
  fprintf(stderr, "GFLOP/s: %.4f\n", b.gflops());
  fprintf(stderr, "Gbyte/s: %.4f\n", b.gb_per_sec());
  fprintf(stderr, "========================================================\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cudaGetLastError();
}

template <class FP_T>
cudaError_t register_tiled_explicit_unroll_driver(
    RankedTensor<FP_T> A, RankedTensor<FP_T> u, RankedTensor<FP_T> grad,
    const int nelts, const int ndofs_1d, const int dim, int niterations) {
  // NOTE: only works for ndofs_1d = 8
  dim3 gridDim(nelts);
  dim3 blockDim(ndofs_1d, ndofs_1d);
  int smem_bytes = sizeof(FP_T) * 2 * ndofs_1d * (ndofs_1d + 4); 

  register_tiled_explicit_unroll<<<gridDim, blockDim, smem_bytes>>>(
      A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < niterations; ++i)
    register_tiled_explicit_unroll<<<gridDim, blockDim, smem_bytes>>>(
        A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  unsigned long long nflops = nelts*ndofs_1d*ndofs_1d*ndofs_1d*ndofs_1d;
  unsigned long long nbytes = sizeof(FP_T)*(
    nelts*ndofs_1d*ndofs_1d*ndofs_1d + ndofs_1d*ndofs_1d
  );
  Benchmark b(niterations, ms, nflops, nbytes);
  fprintf(stderr, "===== Register tiled explicit unroll =====\n");
  fprintf(stderr, "Runtime: %.4f ms\n", ms / niterations);
  fprintf(stderr, "GFLOP/s: %.4f\n", b.gflops());
  fprintf(stderr, "Total GB: %.4f\n", (double)b.nbytes * 1e-9);
  fprintf(stderr, "Gbyte/s: %.4f\n", b.gb_per_sec());
  fprintf(stderr, "==========================================\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cudaGetLastError();
}

#endif
