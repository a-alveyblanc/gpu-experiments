#ifndef _3D_GRADIENT_UTILS_CUH
#define _3D_GRADIENT_UTILS_CUH

#include <vector>
#include <stdio.h>

template <class FP_T>
struct RankedTensor {
  FP_T *host_view;
  FP_T *device_view;
  
  unsigned int num_entries = 1;
  unsigned int rank;
  unsigned long long num_bytes = 0;

  std::vector<int> shape;
  std::vector<int> stride;

  cudaError_t last_error = cudaSuccess;

  RankedTensor(std::vector<int> shape, std::vector<int> stride)
      : shape(shape), stride(stride) {

    rank = shape.size();

    for (int ax_len : shape) num_entries *= ax_len;
    num_bytes = sizeof(FP_T) * num_entries;

    host_view = new FP_T[num_entries];

    last_error = cudaMalloc(reinterpret_cast<void **>(&device_view), num_bytes);
    if (last_error != cudaSuccess)
      fprintf(stderr, "cudaMalloc failed on RankedTensor creation: %s\n",
              cudaGetErrorString(last_error));
  }

  void sync_device() {
    last_error =
        cudaMemcpy(device_view, host_view, num_bytes, cudaMemcpyHostToDevice);
    if (last_error != cudaSuccess)
      fprintf(stderr, "Failed cudaMemcpy host to device: %s\n",
              cudaGetErrorString(last_error));
  }

  void sync_host() {
    last_error =
        cudaMemcpy(host_view, device_view, num_bytes, cudaMemcpyDeviceToHost);
    if (last_error != cudaSuccess)
      fprintf(stderr, "Failed cudaMemcpy device to host: %s\n",
              cudaGetErrorString(last_error));
  }

  void fill(FP_T val) {
    for (int i = 0; i < num_entries; ++i) host_view[i] = val;
    sync_device();
  }

  void random() {
    for (int i = 0; i < num_entries; ++i)
      host_view[i] = (FP_T)rand() / (FP_T)RAND_MAX;
    sync_device();
  }

  bool operator==(RankedTensor<FP_T> other) {
    other.sync_host();
    sync_host();

    FP_T tolerance = 1e-7;
    if (other.num_entries != num_entries) return false;
    if (other.shape != shape) return false;
    if (other.stride != stride) return false;

    for (int i = 0; i < num_entries; ++i)
      if ((other.host_view[i] - host_view[i]) > tolerance) {
        fprintf(stderr, "Numerical error = %.4f at linearized index = %d\n",
                other.host_view[i] - host_view[i], i);
        return false;
      }

    return true;
  }
};

struct Benchmark {
  int niterations;
  double total_runtime_ms;
  unsigned long long nflops;
  unsigned long long nbytes;

  Benchmark(int niterations, double ms, unsigned long long nflops,
            unsigned long long nbytes)
      : niterations(niterations), total_runtime_ms(ms), nflops(nflops),
        nbytes(nbytes) {}

  double gflops() {
    double ms_per_iter = total_runtime_ms / niterations;
    double s_per_iter = 1e-3 * ms_per_iter;
    double gflop_count = 2.0 * (double)nflops * (double)1e-9;

    return gflop_count / s_per_iter;
  }

  double gb_per_sec() {
    double ms_per_iter = total_runtime_ms / niterations;
    double s_per_iter = 1e-3 * ms_per_iter;
    double gbytes = 1e-9 * (double) nbytes;

    return gbytes / s_per_iter;
  }
};

#endif
