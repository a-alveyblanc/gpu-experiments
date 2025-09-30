#ifndef KERNELS_CUH
#define KERNELS_CUH

template <int BS>
__global__ void mod_indexing(const float *A, const float *B, float *C,
                             const int n) {
  int i = blockIdx.x * BS + (threadIdx.x / BS);
  int j = blockIdx.y * BS + (threadIdx.x % BS);

  float acc = 0.0f;
  for (int k = 0; k < n; ++k)
    acc += (A[i*n + k] * B[k*n + j]);

  C[i*n + j] = acc;
}

template <int BM, int BN>
__global__ void normal_indexing(const float *A, const float *B, float *C,
                                const int n) {
  int i = blockIdx.x * BM + threadIdx.x;
  int j = blockIdx.y * BN + threadIdx.y;

  float acc = 0.0f;
  for (int k = 0; k < n; ++k)
    acc += (A[i*n + k] * B[k*n + j]);

  C[i*n + j] = acc;
}

template <int BM, int BN>
__global__ void fast_indexing(const float *A, const float *B, float *C,
                              const int n) {
  int i = blockIdx.y * BM + threadIdx.y;
  int j = blockIdx.x * BN + threadIdx.x;

  float acc = 0.0f;
  for (int k = 0; k < n; ++k)
    acc += (A[i*n + k] * B[k*n + j]);

  C[i*n + j] = acc;
}

template <int BN, int BM, int BK>
__global__ void shared_memory_tiling(const float *A, const float *B, float *C,
                                     const int n) {
  // output tile i.e. C[i,j]
  int i = blockIdx.y;
  int j = blockIdx.x;

  A += i * BM * n;          // select particular row
  B += j * BN ;             // select particular column
  C += i * BM * n + j * BN; // select output tile

  __shared__ float As[BK * BK];
  __shared__ float Bs[BK * BK + 1];

  // tile indices
  int ii = threadIdx.y;
  int ij = threadIdx.x;

  float acc = 0.0f;
  for (int _ = 0; _ < n; _ += BK) { // loop over k-tiles
    
    As[ii * BK + ij] = A[ii * n + ij];
    Bs[ii * BK + ij] = B[ii * n + ij];
    __syncthreads(); // ensure all values needed are loaded into smem

    A += BK;     // get next tile in current tile-row 
    B += BK * n; // get next tile in current tile-column 
    
    // dot product
    for (int k = 0; k < BK; ++k) {
      acc += (
        As[ii * BK + k] * Bs[k * BK + ij]    
      );
    }

    __syncthreads(); // ensure all smem values used to compute 
  }

  C[ii * n + ij] = acc;
}

template<int BM, int BN, int BK, int TM>
__global__
void thread_block_1d(const float *A, const float *B, float *C, const int n) {
  int i = blockIdx.y;
  int j = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN + 1];

  A += i * BM * n;
  B += j * BN;
  C += i * BM * n + j * BN;

  float tmp_thread[TM] = {0.0};

  // linearized blocks since we technically need 4 axes
  int iA = threadIdx.x / BK; // {0, 0, ..., 0, 1, ..., 1, ..., BK-1}
  int jA = threadIdx.x % BK; // {0, 1, ..., BK-1, ..., 0, 1, ..., BK-1}

  int iB = threadIdx.x / BN; // {0, 0, ..., 0, 1, ..., 1, ..., BN-1}
  int jB = threadIdx.x % BN; // {0, 1, ..., BN-1, ..., 0, 1, ..., BN-1}

  int ii = threadIdx.x / BN; // {0, 0, ..., 0, 1, ..., 1, ..., BN-1}
  int ji = threadIdx.x % BN; // {0, 1, ..., BN-1, ..., 0, 1, ..., BN-1}

  for (int k = 0; k < n; k += BK) { // sequential over K-blocks 
    // collaborative loading
    As[iA * BK + jA] = A[iA * n + jA];
    Bs[iB * BN + jB] = B[iB * n + jB];
    __syncthreads();

    A += BK;     // fastest block axis, just increment by block size
    B += BK * n; // slow block axis, need to jump with stride n

    // do actual work
    for (int ki = 0; ki < BK; ++ki) {
      float tmp_B = Bs[ki * BN + ji]; // take advantage of locality
      for (int ithread = 0; ithread < TM; ++ithread) {
        tmp_thread[ithread] += As[(ii * TM + ithread) * BK + ki] * tmp_B;
      }
    }
    __syncthreads();
  }

  // store in global array
  for (int ithread = 0; ithread < TM; ++ithread) {
    C[(ii * TM + ithread) * BN + ji] += tmp_thread[ithread];
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    thread_block_2d(const float *A, const float *B, float *C, int n) {
  // shared and register tiles
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float register_tile[TM * TN] = {0.0}; // results
  float register_rows[TM] = {0.0};      // used to perform computation
  float register_cols[TN] = {0.0};      // used to perform computation

  // results in a 2.2x increase in performance because the compiler can infer
  // the strides below and optimize accordingly
  const uint tbp = (BM * BN) / (TM * TN);

  // output block indices
  int i = blockIdx.y;
  int j = blockIdx.x;

  // output thread indices
  int ii = threadIdx.x / (BN / TN);  // each thread computes TN values
  int ij = threadIdx.x % (BN / TN);

  // input thread indices
  int iiA = threadIdx.x / BK;  // reducing over all K col blocks
  int ijA = threadIdx.x % BK;  // meaning block stride is BK

  // stride = # register tiled rows 
  //   > output block is BM * BN
  //   > each block contains (BM * BN) / (TM * TN) threads
  //   > number of rows per step is (BM * BN) / ((TM * TN) / BK)
  int A_stride = tbp / BK;

  int iiB = threadIdx.x / BN;  // reducing over all K row blocks
  int ijB = threadIdx.x % BN;  // meaning block stride is BN
  
  // stride = # register tiled rows
  //   > output block is BM * BN
  //   > each block contains (BM * BN) / (TM * TN) threads
  //   > number of rows per step is (BM * BN) / ((TM * TN) / BN)
  int B_stride = tbp / BN;

  // increment to correct block tiles
  A += i * BM * n;
  B += j * BN; 
  C += i * BM * n + j * BN;

  for (int kouter = 0; kouter < n; kouter += BK) {  // outer loop over K blocks
    // prefetch into shared
    for (uint offset = 0; offset < BM; offset += A_stride)
      As[(iiA + offset)*BK + ijA] = A[(iiA + offset)*n + ijA];
    for (uint offset = 0; offset < BK; offset += B_stride)
      Bs[(iiB + offset)*BN + ijB] = B[(iiB + offset)*n + ijB];
    __syncthreads();

    A += BK;     // move to next K row block
    B += BK * n; // move to next K col block

    // reduction
    for (int kinner = 0; kinner < BK; ++kinner) {
      // prefetch into registers
      for (int ti = 0; ti < TM; ++ti)
        register_rows[ti] = As[(ii * TM + ti) * BK + kinner];
      for (int ti = 0; ti < TN; ++ti)
        register_cols[ti] = Bs[kinner * BN + ij * TN + ti];

      // register tile
      for (int ireg = 0; ireg < TM; ++ireg) {
        for (int jreg = 0; jreg < TN; ++jreg) {
          register_tile[ireg * TM + jreg] += (
            register_rows[ireg] * register_cols[jreg]
          );
        }
      }
    }

    __syncthreads();
  }

  for (int ireg = 0; ireg < TM; ++ireg) {
    for (int jreg = 0; jreg < TN; ++jreg) {
      C[(ii * TM + ireg) * n + ij * TN + jreg] =
          C[(ii * TM + ireg) * n + ij * TN + jreg] +
          register_tile[ireg * TN + jreg];
    }
  }
}

#endif
