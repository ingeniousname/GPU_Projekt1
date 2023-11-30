#pragma once
#include <cuda_runtime.h>

__device__ void warpReduce(volatile int* shmem_ptr, int t);
__global__ void sum_reduction(int* v, int* v_r);
int reduce(int* dev_data, int size);
int count(int* dev_data, int size, int what);

