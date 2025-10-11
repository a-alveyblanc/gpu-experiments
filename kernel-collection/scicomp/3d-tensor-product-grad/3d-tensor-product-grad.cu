#include <stdio.h>
#include <vector>

#include "../../../gputils/include/gputils/ranked-tensor.cuh"

#include "kernels.cuh"
#include "drivers.cuh"
#include "constants.cuh"
template<> __constant__ float op_const<float>[MAX_N * MAX_N];

int main(int argc, const char **argv) {
  int niterations = 20;
  int dim = 3;
  int nelts = 82000;
  int n = 8;
  using FP_T = float;

  std::vector<int> u_shape = {nelts, n, n, n};
  std::vector<int> u_stride = {n*n*n, 1, n, n*n};
  RankedTensor<FP_T> u("u", u_shape, u_stride);

  std::vector<int> op_shape = {n, n};
  std::vector<int> op_stride = {n, 1};
  RankedTensor<FP_T> op("op", op_shape, op_stride);

  std::vector<int> grad_shape = {dim, nelts, n, n, n};
  std::vector<int> grad_stride = {nelts*n*n*n, n*n*n, 1, n, n*n};
  RankedTensor<FP_T> grad("grad", grad_shape, grad_stride);
  RankedTensor<FP_T> ref_grad("ref_grad", grad_shape, grad_stride);

  u.random();
  op.random();
  grad.fill((FP_T)0.0);

  {
    dim3 gridDim(nelts);
    dim3 blockDim(n, n, n);
    size_t smem_bytes = 0;
    cudaError_t driver_result =
        driver("naive", (void *)naive_kernel<FP_T>, gridDim, blockDim,
               smem_bytes, op, u, ref_grad, nelts, n, dim, niterations);
    if (driver_result != cudaSuccess) {
      fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
      return -1;
    }
  }

  {
    dim3 gridDim(nelts);
    dim3 blockDim(n, n);
    size_t smem_bytes = sizeof(FP_T) * 2 * n * n;
    cudaError_t driver_result =
        driver("shared_memory", (void *)shared_memory_kernel<FP_T>, gridDim,
               blockDim, smem_bytes, op, u, grad, nelts, n, dim, niterations);
    if (driver_result != cudaSuccess) {
      fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
      return -1;
    }
    fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");
  }

  {
    dim3 gridDim(nelts);
    dim3 blockDim(n, n);
    size_t smem_bytes = sizeof(FP_T) * (n * n + n * (n + 1));
    cudaError_t driver_result = driver(
        "shared_no_conflcits", (void *)shared_no_bank_conflicts<FP_T>, gridDim,
        blockDim, smem_bytes, op, u, grad, nelts, n, dim, niterations);
    if (driver_result != cudaSuccess) {
      fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
      return -1;
    }
    fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");
  }

  {
    dim3 gridDim(nelts);
    dim3 blockDim(n, n);
    size_t smem_bytes = sizeof(FP_T) * (n * n + n * (n + 1));
    cudaError_t driver_result =
        driver("register_tiled_k", (void *)register_tiled_k<FP_T>, gridDim,
               blockDim, smem_bytes, op, u, grad, nelts, n, dim, niterations);
    if (driver_result != cudaSuccess) {
      fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
      return -1;
    }
    fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");
  }

  if (n == 8) {
    dim3 gridDim(nelts);
    dim3 blockDim(n, n);
    size_t smem_bytes = sizeof(FP_T) * (n * n + n * (n + 1));
    cudaError_t driver_result =
        driver("register_tiled_k_explicit_unroll",
               (void *)register_tiled_explicit_unroll<FP_T>, gridDim, blockDim,
               smem_bytes, op, u, grad, nelts, n, dim, niterations);
    if (driver_result != cudaSuccess) {
      fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
      return -1;
    }
    fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");
  }

  // TODO:
  // if (n == 8) {
  //   driver_result = persistent_register_tiled_explicit_unroll_driver(
  //       op, u, grad, nelts, n, dim, niterations);
  //   if (driver_result != cudaSuccess) {
  //     fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
  //     return -1;
  //   }
  //   fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");
  // }

  return 0;
}
