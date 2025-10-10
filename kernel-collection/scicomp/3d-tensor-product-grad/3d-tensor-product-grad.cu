#include <stdio.h>
#include <vector>

// FIXME: ugly
#include "constants.cuh"
template<> __constant__ float op_const<float>[MAX_N * MAX_N];

#include "drivers.cuh"
#include "utils.cuh"

int main(int argc, const char **argv) {
  int niterations = 20;
  int dim = 3;
  int nelts = 82000;
  int n = 8;
  using FP_T = float;

  std::vector<int> u_shape = {nelts, n, n, n};
  std::vector<int> u_stride = {n*n*n, 1, n, n*n};
  RankedTensor<FP_T> u(u_shape, u_stride);

  std::vector<int> op_shape = {n, n};
  std::vector<int> op_stride = {n, 1};
  RankedTensor<FP_T> op(op_shape, op_stride);

  std::vector<int> grad_shape = {dim, nelts, n, n, n};
  std::vector<int> grad_stride = {nelts*n*n*n, n*n*n, 1, n, n*n};
  RankedTensor<FP_T> grad(grad_shape, grad_stride);
  RankedTensor<FP_T> ref_grad(grad_shape, grad_stride);

  u.random();
  op.random();
  grad.fill((FP_T)0.0);

  cudaError_t driver_result =
      naive_driver(op, u, ref_grad, nelts, n, dim, niterations);
  if (driver_result != cudaSuccess) {
    fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
    return -1;
  }

  driver_result =
      shared_memory_driver(op, u, grad, nelts, n, dim, niterations);
  if (driver_result != cudaSuccess) {
    fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
    return -1;
  }
  fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");

  driver_result = shared_memory_no_bank_conflicts_driver(op, u, grad, nelts, n,
                                                         dim, niterations);
  if (driver_result != cudaSuccess) {
    fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
    return -1;
  }
  fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");

  driver_result =
      register_tiled_k_driver(op, u, grad, nelts, n, dim, niterations);
  if (driver_result != cudaSuccess) {
    fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
    return -1;
  }
  fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");

  driver_result =
      register_tiled_op_constant(op, u, grad, nelts, n, dim, niterations);
  if (driver_result != cudaSuccess) {
    fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
    return -1;
  }
  fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");

  if (n == 8) {
    driver_result = register_tiled_explicit_unroll_driver(op, u, grad, nelts, n,
                                                          dim, niterations);
    if (driver_result != cudaSuccess) {
      fprintf(stderr, "Driver failed: %s\n", cudaGetErrorString(driver_result));
      return -1;
    }
    fprintf(stderr, "%s", grad == ref_grad ? "Passed\n" : "Failed\n");
  }

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
