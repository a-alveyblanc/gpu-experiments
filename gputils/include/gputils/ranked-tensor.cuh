/* Rough "dual view" tensor type for quickly allocating and filling matrices for
 * development
*/

#ifndef _GPUTILS_RANKEDTENSOR_H
#define _GPUTILS_RANKEDTENSOR_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdio.h>

// TODO: 
// 1. add reporting for equality checking
// 2. add default stride pattern
// 3. move away from raw pointers for data
// 4. other clean-ups
template <class FP_T>
struct RankedTensor {
  FP_T *host_view;
  FP_T *device_view;
  
  size_t num_entries = 1;
  size_t rank;
  size_t num_bytes = 0;

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

  ~RankedTensor() {
    delete[] host_view;
    cudaFree(device_view);
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

  FP_T operator()(std::vector<int> indices) {
    size_t idx = 0;
    for (int i = 0; i < shape.size(); ++i) {
      idx += (shape[i]*stride[i]);
    }
    return host_view[idx];
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
                std::abs(other.host_view[i] - host_view[i]), i);
        return false;
      }

    return true;
  }
};

#endif
