/* Useful for a quicker feedback loop than what Nsight would provide when
 * implementing new optimizations.
 */

#ifndef _GPUTILS_BENCHMARK_H
#define _GPUTILS_BENCHMARK_H

#include <stdio.h>
#include <string>

struct Benchmark {
  float runtime_ms = 0.0f;
  double nbytes;
  double nflops;
  uint niterations;
  std::string name;

  void *kernel = nullptr;
  dim3 grid{};
  dim3 block{};
  void **args = nullptr;
  size_t smem_bytes = 0;

  cudaStream_t stream = 0;

  cudaError_t last_error;
  cudaEvent_t start;
  cudaEvent_t stop;

  // no specific kernel provided
  Benchmark(std::string name, double nbytes, double nflops, uint niterations,
            cudaStream_t stream = 0)
      : name(name), nbytes(nbytes), nflops(nflops), niterations(niterations),
        stream(stream) {
    cuda_check(cudaEventCreate(&start),
               "creating benchmark event in constructor");
    cuda_check(cudaEventCreate(&stop),
               "creating benchmark event in constructor");
  }

  // optionally provide a kernel to launch with, for example, 
  //     Benchmark.run(grid_dim, block_dim, smem_bytes)
  Benchmark(std::string name, void *kernel, void **args, size_t nbytes,
            double nflops, uint niterations, cudaStream_t stream = 0)
      : Benchmark(name, nbytes, nflops, niterations, stream) {
    this->kernel = kernel;
    this->args = args;
  }

  ~Benchmark() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cudaError_t begin() {
    if (!cuda_check(cudaEventRecord(start, stream), "recording event"))
      return last_error;
    return last_error;
  }

  cudaError_t end() {
    if (cuda_check(cudaEventRecord(stop, stream), "recording event"))
      return last_error;
    if (cuda_check(cudaEventSynchronize(stop), "synchronizing event"))
      return last_error;

    return cudaEventElapsedTime(&runtime_ms, start, stop);
  }

  bool cuda_check(cudaError_t ret, const char *error_string) {
    last_error = ret;
    if (last_error != cudaSuccess)
      fprintf(stderr, "%s: %s\n", error_string, cudaGetErrorString(last_error));
    return last_error;
  }

  double seconds_per_iter() {
    return (runtime_ms / niterations) * 1e-3;
  }

  double gflops() {
    return (nflops * 1e-9) / seconds_per_iter();
  }

  double gbps() {
    return (nbytes * 1e-9) / seconds_per_iter();
  }

  void report() {
    fprintf(stderr, "===== %s =====\n", name.c_str());
    fprintf(stderr, "Total time   : %.4f s\n", runtime_ms * 1e-3);
    fprintf(stderr, "Time per iter: %.4f s\n", seconds_per_iter());
    fprintf(stderr, "GFLOP/s      : %.4f\n", gflops());
    fprintf(stderr, "Gbps         : %.4f\n", gbps());
    fprintf(stderr, "\n");
  }

  cudaError_t run(dim3 grid_dim, dim3 block_dim, size_t smem_bytes,
                  bool warmup = true) {
    if (!kernel || !args) {
      fprintf(stderr, "No kernel or arguments supplied\n");
      return last_error = cudaErrorUnknown;
    }

    if (warmup) {
      if (cuda_check(cudaLaunchKernel(kernel, grid_dim, block_dim, args,
                                       smem_bytes, stream),
                      "Warm-up kernel launch"))
        return last_error;
      if (cuda_check(cudaStreamSynchronize(stream), "Warm-up synchronize"))
        return last_error;
    }

    runtime_ms = 0.0f;

    if (begin()) return last_error;

    for (uint i = 0; i < niterations; ++i) {
      if (cuda_check(cudaLaunchKernel(kernel, grid_dim, block_dim, args,
                                       smem_bytes, stream),
                      "Kernel launch"))
        return last_error;
    }

    if (end()) return last_error;
    report();

    return last_error;
  }
};

#endif
