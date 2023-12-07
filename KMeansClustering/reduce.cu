#include "reduce.cuh"
#include <device_launch_parameters.h>
#include "cudaError.h"

#define SHMEM_SIZE 256 * 4

// funkcja zastêpuj¹ce ostatnie dodawania na poziomie bloku w algorytmie redukcji
__device__ void warpReducei(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

// algorytm redukcji
__global__ void sum_reduction(int* v, int size) {
	__shared__ int partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;


	// za³aduj wartoœci do pamiêci wspó³dzielonej z dwóch s¹siednich bloków
	if (i < size)
	{
		partial_sum[threadIdx.x] = v[i];
		if (i + blockDim.x < size)
			partial_sum[threadIdx.x] += v[i + blockDim.x];
	}
	else partial_sum[threadIdx.x] = 0;

	__syncthreads();

	// sumowanie
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// ostatnie sumy 
	if (threadIdx.x < 32) {
		warpReducei(partial_sum, threadIdx.x);
	}

	// zapisanie wyniku na pocz¹tek tablicy
	if (threadIdx.x == 0) {
		v[blockIdx.x] = partial_sum[0];
	}
}

__global__ void isValueKernel(int* data, int* boolarr, int val, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
	{
		if (data[i] == val)
			boolarr[i] = 1;
	}

}

int reduce(int* dev_data, int n)
{
	int res = 0;
	int TB_SIZE = 256;
	int GRID_SIZE = (n / TB_SIZE) / 2;


	do {
		int GRID_SIZE = (int)ceil((n + (TB_SIZE * 2) - 1) / (TB_SIZE * 2));
		sum_reduction << <GRID_SIZE, TB_SIZE >> > (dev_data, n);
		n = GRID_SIZE;

	} while (n > TB_SIZE * 2);
	sum_reduction << <1, TB_SIZE >> > (dev_data, n);
	CALL(cudaGetLastError());
	CALL(cudaDeviceSynchronize());
	CALL(cudaMemcpy(&res, dev_data, sizeof(int), cudaMemcpyDeviceToHost));
	return res;
}

void count(int* dev_data, int* out_data, int n, int k)
{
	int res = 0;
	int TB_SIZE = 256;
	int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE;
	int* dev_boolarr;

	CALL(cudaMalloc((void**)&dev_boolarr, n * sizeof(int)));
	for (int i = 0; i < k; i++)
	{
		CALL(cudaMemset(dev_boolarr, 0, n * sizeof(int)));
		isValueKernel << <GRID_SIZE, TB_SIZE >> > (dev_data, dev_boolarr, i, n);
		out_data[i] = reduce(dev_boolarr, n);
	}

	cudaFree(dev_boolarr);
}
