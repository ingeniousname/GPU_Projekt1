#pragma once
#include <cuda_runtime.h>

__device__ void warpReducei(volatile int* shmem_ptr, int t);
//__device__ void warpReducef(volatile float* shmem_ptr, int t, int offset);
__global__ void sum_reduction(int* v, int* v_r);
int reduce(int* dev_data, int size);
void count(int* dev_data, int* out_data, int n, int k);

