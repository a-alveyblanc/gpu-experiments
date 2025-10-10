#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>

struct Matrix {
  float *data_host;
  float *data_device;
  int n;
  int m;
};

void cuda_error_check(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void alloc_and_copy_matrix(Matrix *mat) {
  cuda_error_check(
      cudaMalloc((void**) &mat->data_device, sizeof(float) * mat->m * mat->n));
  cuda_error_check(
      cudaMemcpy(mat->data_device, mat->data_host, 
        sizeof(float) * mat->n * mat->m, cudaMemcpyHostToDevice));
}

Matrix *build_matrix(const int n, const int m) {
  Matrix *mat = new Matrix;
  mat->data_host = (float*) malloc(sizeof(float) * n * m);
  mat->n = n;
  mat->m = m;

  for (int i = 0; i < m * n; ++i) mat->data_host[i] = 0.0f;

  alloc_and_copy_matrix(mat);

  return mat;
}

void free_matrix(Matrix *mat) {
  free(mat->data_host);
  cudaFree(mat->data_device);

  delete mat;
}

#endif
