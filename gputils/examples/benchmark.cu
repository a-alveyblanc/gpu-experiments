#include "../include/gputils/benchmark.cuh"
#include "../include/gputils/ranked-tensor.cuh"
#include <cmath>

__global__ void matmul(const float *__restrict__ A, const float *__restrict__ B,
                       float *C, int N) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  for (int k = 0; k < N; ++k) {
    acc += A[i*N + k] * B[k*N + j];
  }

  C[i*N + j] = acc;
}

void cpu_matmul(const float *__restrict__ A, const float *__restrict__ B,
                float *C, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < N; ++k) {
        acc += A[i*N + k] * B[k*N + j];
      }
      C[i*N + j] = acc;
    }
  }
}

int main() {
  
  int N = 1024;
  std::vector<int> shape = {N, N};
  std::vector<int> stride = {N, 1};

  RankedTensor<float> A(shape, stride);
  RankedTensor<float> B(shape, stride);
  RankedTensor<float> C(shape, stride);

  A.random();
  B.random();
  C.fill(0.0f);

  size_t nbytes = sizeof(float) * 3 * N * N;
  size_t nflops = N * N * N;
  size_t block_size = 32; 
  dim3 block_dim(block_size, block_size);
  dim3 grid_dim(std::ceil((N + block_size - 1) / block_size),
                std::ceil((N + block_size - 1) / block_size));

  { // run a benchmark
    int niterations = 20;
    void *args[] = {(void *)&A.device_view, (void *)&B.device_view,
                    (void *)&C.device_view, (void *)&N};
    Benchmark b("matmul_example", (void *)matmul, args, nbytes, nflops,
                niterations);

    if (b.run(grid_dim, block_dim, 0 /* warmup = true */)) {
      fprintf(stderr, "Error running benchmark\n");
      return -1;
    }
  }

  { // check correctness
    C.fill(0.0f);
    int niterations = 1;
    bool warmup = false;
    void *args[] = {(void *)&A.device_view, (void *)&B.device_view,
                    (void *)&C.device_view, (void *)&N};

    Benchmark b("matmul_correctness", (void *)matmul, args, nbytes, nflops,
                niterations);
    b.run(grid_dim, block_dim, 0, warmup);
    std::vector<float> C_host(N * N, 0.0f);
    cpu_matmul(A.host_view, B.host_view, C_host.data(), N);

    C.sync_host();
    float tol = 1e-4;
    for (int i = 0; i < N*N; ++i) {
      float error = std::abs(C_host[i] - C.host_view[i]);
      if (error > tol) {
        fprintf(stderr, "Numerical error %f > tolerance\n", error);
        return -1;
      }
    }

    fprintf(stderr, "Success!\n");
  }

  return 0;
}
