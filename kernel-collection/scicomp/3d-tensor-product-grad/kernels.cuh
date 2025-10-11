#ifndef _3D_GRADIENT_KERNELS_CUH
#define _3D_GRADIENT_KERNELS_CUH

#include "constants.cuh"

#define IDX5(d, e, i, j, k)                                                    \
  (d * nelts * ndofs_1d * ndofs_1d * ndofs_1d +                                \
   e * ndofs_1d * ndofs_1d * ndofs_1d + i + j * ndofs_1d +                     \
   k * ndofs_1d * ndofs_1d)
#define IDX4(e, i, j, k)                                                       \
  (e * ndofs_1d * ndofs_1d * ndofs_1d + i + j * ndofs_1d +                     \
   k * ndofs_1d * ndofs_1d)
#define IDX3(e, i, j)                                                          \
  (e * ndofs_1d * ndofs_1d + i * ndofs_1d + j)
#define IDX2(i, j) (i *ndofs_1d + j)
#define IDX2_padded(i, j, stride)                                              \
  (i * stride + j)

template <class FP_T>
__global__ void naive_kernel(const FP_T *__restrict__ A,
                             const FP_T *__restrict__ u,
                             FP_T *__restrict__ grad, const int nelts,
                             const int ndofs_1d, const int dim) {
  /* 1D grid: each block processes a single element
   * 3D blocks: each hardware axis is responsible for each of the spatial axes
   */

  uint e = blockIdx.x;
  uint i = threadIdx.x;
  uint j = threadIdx.y;
  uint k = threadIdx.z;

  FP_T acc[3] = {(FP_T)0.0};
  for (int l = 0; l < ndofs_1d; ++l) {
    acc[0] += A[IDX2(i, l)] * u[IDX4(e, l, j, k)];
    acc[1] += A[IDX2(j, l)] * u[IDX4(e, i, l, k)];
    acc[2] += A[IDX2(k, l)] * u[IDX4(e, i, j, l)];
  }
  
  grad[IDX5(0, e, i, j, k)] = acc[0];
  grad[IDX5(1, e, i, j, k)] = acc[1];
  grad[IDX5(2, e, i, j, k)] = acc[2];
}

template <class FP_T>
__global__ void shared_memory_kernel(const FP_T *__restrict__ A,
                                     const FP_T *__restrict__ u,
                                     FP_T *__restrict__ grad, const int nelts,
                                     const int ndofs_1d, const int dim) {
  extern __shared__ FP_T smem[];
  FP_T *As = smem;
  FP_T *us = smem + ndofs_1d * ndofs_1d;

  uint e = blockIdx.x;
  uint i = threadIdx.x;
  uint j = threadIdx.y;

  As[IDX2(i, j)] = A[IDX2(i, j)];

  for (int k = 0; k < ndofs_1d; ++k) {
    FP_T acc[3] = {(FP_T)0.0};

    us[IDX2(i, j)] = u[IDX4(e, i, j, k)];
    __syncthreads();  // finish loading

    for (int l = 0; l < ndofs_1d; ++l) {
      acc[0] += As[IDX2(i, l)] * us[IDX2(l, j)];
      acc[1] += As[IDX2(j, l)] * us[IDX2(i, l)];
      acc[2] += As[IDX2(k, l)] * u[IDX4(e, i, j, l)];
    }
    __syncthreads();  // finish computing

#pragma unroll
    for (int d = 0; d < dim; ++d)
      grad[IDX5(d, e, i, j, k)] = acc[d];
  }
}

template <class FP_T>
__global__ void
shared_no_bank_conflicts(const FP_T *__restrict__ A, const FP_T *__restrict__ u,
                         FP_T *__restrict__ grad, const int nelts,
                         const int ndofs_1d, const int dim) {
  extern __shared__ FP_T smem[];
  FP_T *As = smem;
  FP_T *us = smem + ndofs_1d * ndofs_1d;
  uint u_stride = ndofs_1d + 1;

  uint e = blockIdx.x;
  uint i = threadIdx.x;
  uint j = threadIdx.y;

  As[IDX2(i, j)] = A[IDX2(i, j)];

  for (int k = 0; k < ndofs_1d; ++k) {
    FP_T acc[3] = {(FP_T)0.0};

    us[IDX2_padded(i, j, u_stride)] = u[IDX4(e, i, j, k)];
    __syncthreads();  // finish loading

    for (int l = 0; l < ndofs_1d; ++l) {
      acc[0] += As[IDX2(i, l)] * us[IDX2_padded(l, j, u_stride)];
      acc[1] += As[IDX2(j, l)] * us[IDX2_padded(i, l, u_stride)];
      acc[2] += As[IDX2(k, l)] * u[IDX4(e, i, j, l)];
    }
    __syncthreads();  // finish computing

#pragma unroll
    for (int d = 0; d < dim; ++d)
      grad[IDX5(d, e, i, j, k)] = acc[d];
  }
}

template <class FP_T>
__global__ __launch_bounds__(64, 1) void register_tiled_k(const FP_T *__restrict__ A,
                                 const FP_T *__restrict__ u,
                                 FP_T *__restrict__ grad, const int nelts,
                                 const int ndofs_1d, const int dim) {
  uint e = blockIdx.x;
  uint i = threadIdx.x;
  uint j = threadIdx.y;

  extern __shared__ FP_T smem[];
  FP_T *As = smem;
  FP_T *us = smem + ndofs_1d*ndofs_1d;
  uint u_stride = ndofs_1d + 1;

  As[IDX2(i, j)] = A[IDX2(i, j)];

  constexpr uint MAX_REG_K = 16;
  FP_T regK[MAX_REG_K] = {(FP_T)0.0};
#pragma unroll
  for (int k = 0; k < ndofs_1d; ++k) {
    FP_T uij_tmp = u[IDX4(e, i, j, k)];
    us[IDX2_padded(i, j, u_stride)] = uij_tmp;
    __syncthreads();

    FP_T acc[2] = {(FP_T)0.0};
#pragma unroll
    for (int l = 0; l < (int) ndofs_1d; ++l) {
      acc[0] += As[IDX2(i, l)] * us[IDX2_padded(l, j, u_stride)];
      acc[1] += As[IDX2(j, l)] * us[IDX2_padded(i, l, u_stride)];
      regK[l] += As[IDX2(l, k)] * uij_tmp; // k constant for entire loop
    }
    __syncthreads();

#pragma unroll
    for (int d = 0; d < dim-1; ++d)
      grad[IDX5(d, e, i, j, k)] = acc[d];
  }

#pragma unroll
  for (int k = 0; k < ndofs_1d; ++k)
    grad[IDX5(2, e, i, j, k)] = regK[k];
}

template <class FP_T>
__global__ void register_tiled_const_A(const FP_T *__restrict__ u,
                                       FP_T *__restrict__ grad, const int nelts,
                                       const int ndofs_1d, const int dim) {
  uint e = blockIdx.x;
  uint i = threadIdx.x;
  uint j = threadIdx.y;

  extern __shared__ FP_T smem[];
  FP_T *us = smem;

  constexpr uint MAX_REG_K = 16;
  FP_T regK[MAX_REG_K] = {(FP_T)0.0};
  for (int k = 0; k < ndofs_1d; ++k) {
    us[IDX2(i, j)] = u[IDX4(e, i, j, k)];
    __syncthreads();

    FP_T acc[2] = {(FP_T)0.0};
#pragma unroll
    for (int l = 0; l < ndofs_1d; ++l) {
      acc[0]  += op_const<FP_T>[IDX2(i, l)] * us[IDX2(l, j)];
      acc[1]  += op_const<FP_T>[IDX2(j, l)] * us[IDX2(i, l)];
      regK[l] += op_const<FP_T>[IDX2(l, k)] * us[IDX2(i, j)];
    }
    __syncthreads();

#pragma unroll
    for (int d = 0; d < dim-1; ++d)
      grad[IDX5(d, e, i, j, k)] = acc[d];
  }

#pragma unroll
  for (int k = 0; k < ndofs_1d; ++k)
    grad[IDX5(2, e, i, j, k)] = regK[k];
}

template <class FP_T>
__global__ void __launch_bounds__(64, 1)
register_tiled_explicit_unroll(const FP_T *__restrict__ A,
                               const FP_T *__restrict__ u,
                               FP_T *__restrict__ grad, const int nelts,
                               const int ndofs_1d, const int dim) {
  uint e = blockIdx.x;
  uint i = threadIdx.x;
  uint j = threadIdx.y;

  extern __shared__ FP_T smem[];
  uint ustride = ndofs_1d + 4;
  uint Astride = ndofs_1d + 4;
  FP_T *As = smem;
  FP_T *us = smem + ndofs_1d*Astride;

  As[IDX2_padded(i, j, Astride)] = A[IDX2(i, j)];
  __syncthreads();

  constexpr uint MAX_REG_K = 16;
  FP_T regK[MAX_REG_K] = {(FP_T)0.0};
  for (int k = 0; k < ndofs_1d; ++k) {
    FP_T uij_tmp = u[IDX4(e, i, j, k)];
    us[IDX2_padded(j, i, ustride)] = uij_tmp;
    __syncthreads();

    FP_T acc_x = (FP_T) 0.0;
    FP_T acc_y = (FP_T) 0.0;

    acc_x += As[IDX2_padded(i, 0, Astride)] * us[IDX2_padded(j, 0, ustride)];
    acc_x += As[IDX2_padded(i, 1, Astride)] * us[IDX2_padded(j, 1, ustride)];
    acc_x += As[IDX2_padded(i, 2, Astride)] * us[IDX2_padded(j, 2, ustride)];
    acc_x += As[IDX2_padded(i, 3, Astride)] * us[IDX2_padded(j, 3, ustride)];

    acc_y += As[IDX2_padded(j, 0, Astride)] * us[IDX2_padded(0, i, ustride)];
    acc_y += As[IDX2_padded(j, 1, Astride)] * us[IDX2_padded(1, i, ustride)];
    acc_y += As[IDX2_padded(j, 2, Astride)] * us[IDX2_padded(2, i, ustride)];
    acc_y += As[IDX2_padded(j, 3, Astride)] * us[IDX2_padded(3, i, ustride)];

    regK[0] += As[IDX2_padded(0, k, Astride)] * uij_tmp;
    regK[1] += As[IDX2_padded(1, k, Astride)] * uij_tmp;
    regK[2] += As[IDX2_padded(2, k, Astride)] * uij_tmp;
    regK[3] += As[IDX2_padded(3, k, Astride)] * uij_tmp;

    acc_x += As[IDX2_padded(i, 4, Astride)] * us[IDX2_padded(j, 4, ustride)];
    acc_x += As[IDX2_padded(i, 5, Astride)] * us[IDX2_padded(j, 5, ustride)];
    acc_x += As[IDX2_padded(i, 6, Astride)] * us[IDX2_padded(j, 6, ustride)];
    acc_x += As[IDX2_padded(i, 7, Astride)] * us[IDX2_padded(j, 7, ustride)];

    acc_y += As[IDX2_padded(j, 4, Astride)] * us[IDX2_padded(4, i, ustride)];
    acc_y += As[IDX2_padded(j, 5, Astride)] * us[IDX2_padded(5, i, ustride)];
    acc_y += As[IDX2_padded(j, 6, Astride)] * us[IDX2_padded(6, i, ustride)];
    acc_y += As[IDX2_padded(j, 7, Astride)] * us[IDX2_padded(7, i, ustride)];

    regK[4] += As[IDX2_padded(4, k, Astride)] * uij_tmp;
    regK[5] += As[IDX2_padded(5, k, Astride)] * uij_tmp;
    regK[6] += As[IDX2_padded(6, k, Astride)] * uij_tmp;
    regK[7] += As[IDX2_padded(7, k, Astride)] * uij_tmp;

    grad[IDX5(0, e, i, j, k)] = acc_x;
    grad[IDX5(1, e, i, j, k)] = acc_y;
    __syncthreads();
  }

#pragma unroll
  for (int k = 0; k < ndofs_1d; ++k)
    grad[IDX5(2, e, i, j, k)] = regK[k];
}

__device__ __forceinline__
void cp_async_16B(void *dst, const void *src) {
  unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16;\n" ::
      "r"(smem_addr), "l"(src)
  );
}

__device__ __forceinline__
void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__
void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

template <class FP_T>
__global__ void
persistent_register_tiled_explicit_unroll(
    const FP_T *__restrict__ A, 
    const FP_T *__restrict__ u,
    FP_T *__restrict__ grad, 
    const int nelts, const int ndofs_1d, const int dim, 
    const int nelts_per_block) {
  /*
   * Implement persistent kernel (1 block / SM) + pipelining to try and push
   * performance much higher
   */
}

#endif
