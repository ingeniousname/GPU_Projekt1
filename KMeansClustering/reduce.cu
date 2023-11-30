#include "reduce.cuh"
#include <device_launch_parameters.h>
#include "cudaError.h"

#define SHMEM_SIZE 256 * 4

__device__ void warpReduce(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}



__global__ void sum_reduction(int* v, int size) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	if (i < size)
	{
		partial_sum[threadIdx.x] = v[i];
		if (i + blockDim.x < size)
			partial_sum[threadIdx.x] += v[i + blockDim.x];
	}
	else partial_sum[threadIdx.x] = 0;

	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v[blockIdx.x] = partial_sum[0];
	}
}

__global__ void isValueKernel(int* data, int* boolarr, int* val_and_size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < val_and_size[1])
	{
		if (data[i] == val_and_size[i])
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
	CALL(cudaMemcpy(&res, dev_data, sizeof(int), cudaMemcpyDeviceToHost));
	return res;
}

int count(int* dev_data, int n, int what)
{
	int res = 0;
	int TB_SIZE = 256;
	int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE;
	int *dev_boolarr, *dev_val_and_size;

	CALL(cudaMalloc((void**)&dev_boolarr, n * sizeof(int)));
	CALL(cudaMalloc((void**)&dev_val_and_size, 2 * sizeof(int)));
	CALL(cudaMemset(dev_boolarr, 0, n * sizeof(int)));

	int vns[2];
	vns[0] = what;
	vns[1] = n;
	CALL(cudaMemcpy(dev_val_and_size, vns, 2 * sizeof(int), cudaMemcpyHostToDevice));

	isValueKernel << <GRID_SIZE, TB_SIZE >> > (dev_data, dev_boolarr, dev_val_and_size);
	res = reduce(dev_boolarr, n);

	cudaFree(dev_boolarr);
	cudaFree(dev_val_and_size);
	return res;
}
