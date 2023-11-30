#include "cudaError.h"
#include "update_index_naive.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <iostream>
#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void calcDistKernel(float* data, double* clusterData, int* n, int* d, int* k, float* min_dist)
{
    int res_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int data_idx = res_idx / (*k);
    int c_idx = res_idx % (*k);

    if (res_idx < (*n) * (*k))
    {
        for (int i = 0; i < *d; i++)
        {
            min_dist[res_idx] += (data[data_idx * (*d) + i] - clusterData[c_idx * (*d) + i]) * (data[data_idx * (*d) + i] - clusterData[c_idx * (*d) + i]);
        }
    }

    __syncthreads();
}

__global__ void updateIndexKernel(float* data, float* min_dist, double* new_cluster_data, int* cluster_count, int* n, int* d, int* k, int* indicies, int* delta)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float min = min_dist[idx * (*k)];
    int min_idx = 0;
    int last_idx = indicies[idx];
    if (idx < (*n))
    {
        for (int i = 1; i < (*k); i++)
        {
            if (min > min_dist[idx * (*k) + i])
            {
                min = min_dist[idx * (*k) + i];
                min_idx = i;
            }
        }
        if (min_idx != last_idx)
        {
            indicies[idx] = min_idx;
            atomicAdd(delta, 1);
        }
        atomicAdd(cluster_count + min_idx, 1);

        for (int i = 0; i < (*d); i++)
        {
            atomicAddDouble(new_cluster_data + (min_idx * (*d)) + i, (double)data[idx * (*d) + i]);
        }

    }

    __syncthreads();
   
}

void updateIndexNaive(PointsData& data)
{
    float* dev_data;
    double* dev_clusterData, *dev_newClusterData;
    int* dev_d, *dev_k, *dev_n, *dev_delta;
    float* dev_min_dist;
    int* dev_clusterIdx, *dev_clusterCount;


    int* clusterCount = new int[data.k];
    const int N_THREADS = 256;
    const float threshold = 1e-7;
    int delta = 0;

    // set device
    CALL(cudaSetDevice(0));

    // malloc
    CALL(cudaMalloc((void**)&dev_data, data.n * data.d * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_clusterData, data.d * data.k * sizeof(double)));
    CALL(cudaMalloc((void**)&dev_newClusterData, data.d * data.k * sizeof(double)));
    CALL(cudaMalloc((void**)&dev_d, sizeof(int)));
    CALL(cudaMalloc((void**)&dev_n, sizeof(int)));
    CALL(cudaMalloc((void**)&dev_k, sizeof(int)));
    CALL(cudaMalloc((void**)&dev_delta, sizeof(int)));
    CALL(cudaMalloc((void**)&dev_clusterIdx, data.n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_clusterCount, data.k * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_min_dist, data.n * data.k * sizeof(float)));


    // memset/memcpy
    CALL(cudaMemcpy(dev_data, data.data, data.n * data.d * sizeof(float), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_d, &data.d, sizeof(int), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_k, &data.k, sizeof(int), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_n, &data.n, sizeof(int), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_clusterIdx, data.clusterIndex, data.n * sizeof(int), cudaMemcpyHostToDevice));



    do
    {
        CALL(cudaMemcpy(dev_clusterData, data.clusterData, data.d * data.k * sizeof(double), cudaMemcpyHostToDevice));
        CALL(cudaMemset(dev_min_dist, 0, data.n * data.k * sizeof(float)));
        CALL(cudaMemset(dev_clusterCount, 0, data.k * sizeof(float)));
        CALL(cudaMemset(dev_newClusterData, 0, data.d * data.k * sizeof(double)));
        CALL(cudaMemset(dev_delta, 0, sizeof(int)));

        dim3 threads = dim3(N_THREADS);
        dim3 calcBlocks = dim3((data.n * data.k + (N_THREADS - 1)) / N_THREADS);

        calcDistKernel <<<calcBlocks, threads >>> (dev_data, dev_clusterData, dev_n, dev_d, dev_k, dev_min_dist);
        // Check for any errors launching the kernel
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());

        dim3 updateBlocks = dim3((data.n + (N_THREADS - 1)) / N_THREADS);
            
        updateIndexKernel << <updateBlocks, threads >> > (dev_data, dev_min_dist, dev_newClusterData, dev_clusterCount, dev_n, dev_d, dev_k, dev_clusterIdx, dev_delta);
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());


        // Copy output vector from GPU buffer to host memory.
        CALL(cudaMemcpy(data.clusterIndex, dev_clusterIdx, data.n * sizeof(int), cudaMemcpyDeviceToHost));
        CALL(cudaMemcpy(data.clusterData, dev_newClusterData, data.k * data.d * sizeof(double), cudaMemcpyDeviceToHost));
        CALL(cudaMemcpy(&delta, dev_delta, sizeof(int), cudaMemcpyDeviceToHost));
        CALL(cudaMemcpy(clusterCount, dev_clusterCount , data.k * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < data.k; i++)
            for (int j = 0; j < data.d; j++)
            {
                data.clusterData[i * data.d + j] /= clusterCount[i];
            }
    } while ((float)delta / data.n > threshold);



    cudaFree(dev_clusterData);
    cudaFree(dev_newClusterData);
    cudaFree(dev_clusterIdx);
    cudaFree(dev_clusterCount);
    cudaFree(dev_data);
    cudaFree(dev_delta);
    cudaFree(dev_d);
    cudaFree(dev_k);
    cudaFree(dev_n);
    cudaFree(dev_min_dist);
}