#pragma once
#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>

// Makro pozwalaj¹ce na sprawdzenie b³êdu podczas wywo³ywania funkcji CUDA
#define CALL(x) do{ \
	cudaError_t cudaStatus = x; \
	if(cudaStatus != cudaSuccess) \
	{ \
		fprintf(stderr, "CudaError: %s, %s\n", cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus)); \
		exit(1); \
	} \
} while(false) // zastosowanie konstrukcji do-while jest celowe - mo¿na postawiæ œrednik na koñcu gdy owijamy funkcjê w makro