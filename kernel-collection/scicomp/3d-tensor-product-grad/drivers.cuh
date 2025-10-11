#ifndef _3D_GRADIENT_DRIVERS_CUH
#define _3D_GRADIENT_DRIVERS_CUH

#include "../../../gputils/include/gputils/ranked-tensor.cuh"
#include "../../../gputils/include/gputils/benchmark.cuh"

template <class FP_T>
cudaError_t driver(const char *name, void *kernel, dim3 gridDim, dim3 blockDim,
                   size_t smem_bytes, RankedTensor<FP_T> &A,
                   RankedTensor<FP_T> &u, RankedTensor<FP_T> &grad,
                   const int nelts, const int ndofs_1d, const int dim,
                   int niterations) {
  unsigned long long nflops =
      2 * nelts * ndofs_1d * ndofs_1d * ndofs_1d * ndofs_1d;
  unsigned long long nbytes =
      sizeof(FP_T) *
      (nelts * ndofs_1d * ndofs_1d * ndofs_1d + ndofs_1d * ndofs_1d);

  void *args[] = {(void *)&A.device_view,    (void *)&u.device_view,
                  (void *)&grad.device_view, (void *)&nelts,
                  (void *)&ndofs_1d,         (void *)&dim};

  Benchmark b(name, kernel, args, nbytes, nflops, niterations);

  return b.run(gridDim, blockDim, smem_bytes);
}

// FIXME: remove once implemented in main file
// template <class FP_T>
// cudaError_t persistent_register_tiled_explicit_unroll_driver(
//     RankedTensor<FP_T> A, RankedTensor<FP_T> u, RankedTensor<FP_T> grad,
//     const int nelts, const int ndofs_1d, const int dim, int niterations) {
//   // NOTE: 3090 has 82 SMs
//   dim3 gridDim(82);
//   dim3 blockDim(ndofs_1d, ndofs_1d);
//   int smem_bytes = sizeof(FP_T) * 3 * ndofs_1d * (ndofs_1d + 4); 
//
//   // FIXME: handle "the tail" eventually 
//   int nelts_per_block = nelts / 82;
//
//   persistent_register_tiled_explicit_unroll<<<gridDim, blockDim, smem_bytes>>>(
//       A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim,
//       nelts_per_block);
//
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//
//   cudaEventRecord(start);
//   for (int i = 0; i < niterations; ++i)
//     persistent_register_tiled_explicit_unroll<<<gridDim, blockDim, smem_bytes>>>(
//         A.device_view, u.device_view, grad.device_view, nelts, ndofs_1d, dim,
//         nelts_per_block);
//
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//
//   float ms;
//   cudaEventElapsedTime(&ms, start, stop);
//
//   unsigned long long nflops = nelts*ndofs_1d*ndofs_1d*ndofs_1d*ndofs_1d;
//   unsigned long long nbytes = sizeof(FP_T)*(
//     nelts*ndofs_1d*ndofs_1d*ndofs_1d + ndofs_1d*ndofs_1d
//   );
//   Benchmark b(niterations, ms, nflops, nbytes);
//   fprintf(stderr, "===== Persistent kernel =====\n");
//   fprintf(stderr, "Runtime: %.4f ms\n", ms / niterations);
//   fprintf(stderr, "GFLOP/s: %.4f\n", b.gflops());
//   fprintf(stderr, "Total GB: %.4f\n", (double)b.nbytes * 1e-9);
//   fprintf(stderr, "Gbyte/s: %.4f\n", b.gb_per_sec());
//   fprintf(stderr, "=============================\n");
//
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);
//
//   return cudaGetLastError();
// }

#endif
