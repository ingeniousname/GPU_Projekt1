#pragma once
#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>

#define CALL(x) do{ \
	cudaError_t cudaStatus = x; \
	if(cudaStatus != cudaSuccess) \
	{ \
		fprintf(stderr, "CudaError: %s, %s\n", cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus)); \
		exit(1); \
	} \
} while(false)