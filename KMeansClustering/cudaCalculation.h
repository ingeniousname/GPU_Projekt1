#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(int* c, const int* a, const int* b);
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
int performCalculations();
