#include "cudaError.h"
#include "update_index_naive.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "reduce.cuh"
#include "clustering_par.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

__device__ void warpReducef(volatile float* shmem_ptr, int t, int offset) {
	shmem_ptr[offset + t] += shmem_ptr[offset + t + 32];
	shmem_ptr[offset + t] += shmem_ptr[offset + t + 16];
	shmem_ptr[offset + t] += shmem_ptr[offset + t + 8];
	shmem_ptr[offset + t] += shmem_ptr[offset + t + 4];
	shmem_ptr[offset + t] += shmem_ptr[offset + t + 2];
	shmem_ptr[offset + t] += shmem_ptr[offset + t + 1];
}

__global__ void calcDistKernel_par(float* data, float* clusterData, int* ndk, float* dist)
{
    extern __shared__ float clusterData_shared[];
    int res_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = ndk[0], d = ndk[1], k = ndk[2];
    int tid = threadIdx.x;
    if (tid < d * k)
        clusterData_shared[tid] = clusterData[tid];

    __syncthreads();

    int data_idx = res_idx / k;
    int c_idx = res_idx % k;
    if (res_idx < n * k)
    {
        float v = 0.f;
        for (int i = 0; i < d; i++)
        {
            float part_v = data[i * n + data_idx] - clusterData_shared[c_idx * d + i];
            v += part_v * part_v;
        }
        dist[res_idx] = v;
    }

    __syncthreads();
}

__global__ void calculateNewClustersKernel_AtomicAdd(float* data, float* min_dist, float* new_cluster_data, int* cluster_count, int* ndk, int* indicies, int* delta)
{
    extern __shared__ float min_dist_shared[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = ndk[0], d = ndk[1], k = ndk[2];




    int tid = threadIdx.x;
    for (int i = 0; i < k; i++)
    {
        int out_idx = blockDim.x * blockIdx.x * k + blockDim.x * i + tid;
        if (out_idx < n * k)
            min_dist_shared[blockDim.x * i + tid] = min_dist[out_idx];
        __syncthreads();

    }




    float min = min_dist_shared[tid * k];
    int min_idx = 0;
    int last_idx = indicies[idx];
    if (idx < n)
    {
        for (int i = 1; i < k; i++)
        {
            if (min > min_dist_shared[tid * k + i])
            {
                min = min_dist_shared[tid * k + i];
                min_idx = i;
            }
        }
        if (min_idx != last_idx)
        {
            indicies[idx] = min_idx;
            delta[idx] = 1;
        }
        atomicAdd(cluster_count + min_idx, 1);

        for (int i = 0; i < d; i++)
        {
            atomicAdd(new_cluster_data + (min_idx * d) + i, data[i * n + idx]);
        }

    }

    __syncthreads();

}

__global__ void updateIndexKernel_par(float* min_dist, int* ndk, int* indicies, int* delta)
{
    extern __shared__ float min_dist_shared[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = ndk[0], d = ndk[1], k = ndk[2];

    int tid = threadIdx.x;
    for (int i = 0; i < k; i++)
    {
        int out_idx = blockDim.x * blockIdx.x * k + blockDim.x * i + tid;
        if(out_idx < n * k)
            min_dist_shared[blockDim.x * i + tid] = min_dist[out_idx];
        __syncthreads();

    }

    if (idx < n)
    {
        float min = min_dist_shared[tid * k];
        int min_idx = 0;
        int last_idx = indicies[idx];
        for (int i = 1; i < k; i++)
        {
            if (min > min_dist_shared[tid * k + i])
            {
                min = min_dist_shared[tid * k + i];
                min_idx = i;
            }
        }
        if (min_idx != last_idx)
        {
            indicies[idx] = min_idx;
            delta[idx] = 1;
        }
    }

    __syncthreads();

}

__global__ void createIndexMapKernel_thrust(int* indicies, int* ndk, int* out_indexmap)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = ndk[0], d = ndk[1];
    if (idx < n)
    {
        for (int i = 0; i < d; i++)
        {
            out_indexmap[i * n + idx] = d * indicies[idx] + i;
        }

    }

    __syncthreads();

}

__global__ void prepReduceForIndexKernel(int* indicies, float* data, float* out_array, int* ndk, int stride, int idx)
{
    extern __shared__ float localSum[];
    int n = ndk[0], d = ndk[1], k = ndk[2];
    int tid = threadIdx.x;
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    //int logic_val = indicies[global_idx] == idx ? 1 : 0;
    for (int i = 0; i < d; i++)
    {
        if (global_idx < n)
        {
            if(indicies[global_idx] == idx)
                localSum[i * blockDim.x + tid] = data[i * n + global_idx];
            else localSum[i * blockDim.x + tid] = 0;
        }
        else localSum[i * blockDim.x + tid] = 0;
        //localSum[i * blockDim.x + tid] = logic_val;
        __syncthreads();

    }



    for (int i = 0; i < d; i++)
    {
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
            // Each thread does work unless it is further than the stride
            if (tid < s) {
                localSum[i * blockDim.x + tid] += localSum[i * blockDim.x + tid + s];
            }
            __syncthreads();
        }

        if (threadIdx.x < 32) {
            warpReducef(localSum, threadIdx.x, blockDim.x * i);
        }
    }



    if (tid < d)
        out_array[stride * tid + blockIdx.x] = localSum[tid * blockDim.x];



    __syncthreads();
}

__global__ void calculateLocalSums(int* indicies, float* data, float* out_array, int* ndk, int stride)
{
    extern __shared__ float localSum[];
    int n = ndk[0], d = ndk[1], k = ndk[2];
    int tid = threadIdx.x;
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < d * k)
        localSum[tid] = 0;
    __syncthreads();


    if (global_idx < n)
    {

        for (int i = 0; i < d; i++)
        {
                //localSum[d * indicies[global_idx] + i] += data[i * n + global_idx];
            //atomicAdd(localSum + d * indicies[global_idx] + i, data[global_idx]);
            //atomicAdd(localSum + d + i, 1);
            atomicAdd(out_array + stride * (d * indicies[global_idx] + i) + blockIdx.x, 1);
            __syncthreads();
        }

    }


    //if (tid < d * k)
    //    out_array[stride * tid + blockIdx.x] = localSum[tid];



    __syncthreads();


}

__global__ void reduceLocalSums(float* v, int stride, int size, int new_stride)
{
    extern __shared__ float partial_sum[];


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int dk_idx = i / stride;
    int b_idx = (i % stride) / blockDim.x;


    if (i < size)
    {
        partial_sum[threadIdx.x] = v[i];
        v[i] = 0;
    }
    else partial_sum[threadIdx.x] = 0;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warpReducef(partial_sum, threadIdx.x, 0);
    }

    if (threadIdx.x == 0) {
        if (new_stride == 0)
        {
            v[blockIdx.x] = partial_sum[0];
        }
        else v[new_stride * dk_idx + b_idx] = partial_sum[0];
    }

}

int calculateStride(int n, int N_THREADS)
{
    int stride = std::max(n / N_THREADS, 1);
    stride = std::ceil((double)stride / (N_THREADS)) * N_THREADS;
    return stride;
}

void Kmeans_par_AtomicAddNewCluster(PointsData_FlatArray& data, int kk)
{

    data.ndk[2] = kk;

    float* dev_data;
    float* dev_clusterData, * dev_newClusterData;
    int* dev_ndk, *dev_delta;
    float* dev_min_dist;
    int* dev_clusterIdx, *dev_clusterCount;


    int n = data.ndk[0], d = data.ndk[1], k = data.ndk[2];
    int* clusterCount = new int[k];
    const int N_THREADS = 256;
    const float threshold = 1e-7;
    int delta = 0;

    // set device
    CALL(cudaSetDevice(0));

    // malloc
    CALL(cudaMalloc((void**)&dev_data, n * d * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_clusterData, d * k * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_newClusterData, d * k * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_ndk, 3 * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_delta, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_clusterIdx, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_clusterCount, k * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_min_dist, n * k * sizeof(float)));


    // memset/memcpy
    CALL(cudaMemcpy(dev_data, data.data, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_ndk, data.ndk, 3 * sizeof(int), cudaMemcpyHostToDevice));



    do
    {
        CALL(cudaMemcpy(dev_clusterData, data.clusterData, d * k * sizeof(float), cudaMemcpyHostToDevice));
        CALL(cudaMemcpy(dev_clusterIdx, data.clusterIndex, n * sizeof(int), cudaMemcpyHostToDevice));
        CALL(cudaMemset(dev_min_dist, 0, n * k * sizeof(float)));
        CALL(cudaMemset(dev_clusterCount, 0, k * sizeof(float)));
        CALL(cudaMemset(dev_newClusterData, 0, d * k * sizeof(float)));
        CALL(cudaMemset(dev_delta, 0, n * sizeof(int)));

        dim3 threads = dim3(N_THREADS);
        dim3 calcBlocks = dim3((n * k + (N_THREADS - 1)) / N_THREADS);

        calcDistKernel_par << <calcBlocks, threads, k*d*sizeof(float) >> > (dev_data, dev_clusterData, dev_ndk, dev_min_dist);
        // Check for any errors launching the kernel
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());

        dim3 updateBlocks = dim3((n + (N_THREADS - 1)) / N_THREADS);

        calculateNewClustersKernel_AtomicAdd << <updateBlocks, threads, k * N_THREADS * sizeof(int) >> > (dev_data, dev_min_dist, dev_newClusterData, dev_clusterCount, dev_ndk, dev_clusterIdx, dev_delta);
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());


        // Copy output vector from GPU buffer to host memory.
        CALL(cudaMemcpy(data.clusterIndex, dev_clusterIdx, n * sizeof(int), cudaMemcpyDeviceToHost));
        CALL(cudaMemcpy(data.clusterData, dev_newClusterData, k * d * sizeof(float), cudaMemcpyDeviceToHost));
        CALL(cudaMemcpy(clusterCount, dev_clusterCount, k * sizeof(int), cudaMemcpyDeviceToHost));

        delta = reduce(dev_delta, n);

        for (int i = 0; i < k; i++)
        {
            //clusterCount[i] = count(dev_clusterIdx, n, i);
            for (int j = 0; j < d; j++)
            {
                data.clusterData[i * d + j] /= clusterCount[i];
                //std::cout << data.clusterData[i * d + j] << " ";
            }
        }
        //std::cout << "\n";
    } while ((float)delta / n > threshold);



    cudaFree(dev_clusterData);
    cudaFree(dev_newClusterData);
    cudaFree(dev_clusterIdx);
    cudaFree(dev_clusterCount);
    cudaFree(dev_data);
    cudaFree(dev_delta);
    cudaFree(dev_ndk);
    cudaFree(dev_min_dist);
}

void Kmeans_par_UpdateClusterOnCPU(PointsData_FlatArray& data, int kk)
{
    data.ndk[2] = kk;

    float* dev_data;
    float* dev_clusterData;
    int* dev_ndk, * dev_delta;
    float* dev_dist;
    int* dev_clusterIdx;


    int n = data.ndk[0], d = data.ndk[1], k = data.ndk[2];
    int* clusterCount = new int[k];
    const int N_THREADS = 256;
    const float threshold = 1e-7;
    int delta = 0;

    // set device
    CALL(cudaSetDevice(0));

    // malloc
    CALL(cudaMalloc((void**)&dev_data, n * d * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_clusterData, d * k * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_ndk, 3 * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_delta, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_clusterIdx, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_dist, n * k * sizeof(float)));


    // memset/memcpy
    CALL(cudaMemcpy(dev_data, data.data, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_ndk, data.ndk, 3 * sizeof(int), cudaMemcpyHostToDevice));



    do
    {
        CALL(cudaMemcpy(dev_clusterData, data.clusterData, d * k * sizeof(float), cudaMemcpyHostToDevice));
        CALL(cudaMemcpy(dev_clusterIdx, data.clusterIndex, n * sizeof(int), cudaMemcpyHostToDevice));
        CALL(cudaMemset(dev_dist, 0, n * k * sizeof(float)));
        CALL(cudaMemset(dev_delta, 0, n * sizeof(int)));

        dim3 threads = dim3(N_THREADS);
        dim3 calcBlocks = dim3((n * k + (N_THREADS - 1)) / N_THREADS);

        calcDistKernel_par << <calcBlocks, threads, k* d * sizeof(float) >> > (dev_data, dev_clusterData, dev_ndk, dev_dist);
        // Check for any errors launching the kernel
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());

        dim3 updateBlocks = dim3((n + (N_THREADS - 1)) / N_THREADS);

        updateIndexKernel_par << <updateBlocks, threads, k * N_THREADS * sizeof(int) >> > (dev_dist, dev_ndk, dev_clusterIdx, dev_delta);
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());


        // Copy output vector from GPU buffer to host memory.
        CALL(cudaMemcpy(data.clusterIndex, dev_clusterIdx, n * sizeof(int), cudaMemcpyDeviceToHost));

        delta = reduce(dev_delta, n);

        memset(data.clusterData, 0, sizeof(float) * k * d);
        memset(clusterCount, 0, sizeof(int) * k);
        for (int i = 0; i < n; i++)
        {
            clusterCount[data.clusterIndex[i]]++;
            for (int j = 0; j < d; j++)
            {
                data.clusterData[data.clusterIndex[i] * d + j] += data.data[j * n + i];
            }
        }

        for (int i = 0; i < k; i++)
        {
            //clusterCount[i] = count(dev_clusterIdx, n, i);
            for (int j = 0; j < d; j++)
            {
                data.clusterData[i * d + j] /= clusterCount[i];
                //std::cout << data.clusterData[i * d + j] << " ";
            }
        }
        //std::cout << "\n";
    } while ((float)delta / n > threshold);



    cudaFree(dev_clusterData);
    cudaFree(dev_clusterIdx);
    cudaFree(dev_data);
    cudaFree(dev_delta);
    cudaFree(dev_ndk);
    cudaFree(dev_dist);

    delete clusterCount;
}

void Kmeans_par_NewClusterSumGPU(PointsData_FlatArray& data, int kk)
{
    data.ndk[2] = kk;

    float* dev_data;
    float* dev_clusterData, *dev_clusterSum;
    int* dev_ndk, *dev_delta;
    float* dev_min_dist;
    int* dev_clusterIdx;


    int n = data.ndk[0], d = data.ndk[1], k = data.ndk[2];
    float* new_clusters = new float[d * k];
    int* clusterCount = new int[k];
    const int N_THREADS = 256;
    int stride = calculateStride(n, N_THREADS);
    const float threshold = 1e-7;
    int delta = 0;

    // set device
    CALL(cudaSetDevice(0));

    // malloc
    CALL(cudaMalloc((void**)&dev_data, n * d * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_clusterData, d * k * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_clusterSum, k * stride * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_ndk, 3 * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_delta, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_clusterIdx, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_min_dist, n * k * sizeof(float)));


    // memset/memcpy
    CALL(cudaMemcpy(dev_data, data.data, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_ndk, data.ndk, 3 * sizeof(int), cudaMemcpyHostToDevice));



    do
    {
        // reset memory
        CALL(cudaMemcpy(dev_clusterData, data.clusterData, d * k * sizeof(float), cudaMemcpyHostToDevice));
        CALL(cudaMemcpy(dev_clusterIdx, data.clusterIndex, n * sizeof(int), cudaMemcpyHostToDevice));
        CALL(cudaMemset(dev_min_dist, 0, n * k * sizeof(float)));
        CALL(cudaMemset(dev_delta, 0, n * sizeof(int)));

        // grid size for calculating dist
        dim3 threads = dim3(N_THREADS);
        dim3 calcBlocks = dim3((n * k + (N_THREADS - 1)) / N_THREADS);

        calcDistKernel_par << <calcBlocks, threads, k* d * sizeof(float) >> > (dev_data, dev_clusterData, dev_ndk, dev_min_dist);
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());

        // grid size for the rest of the operations
        dim3 updateBlocks = dim3((n + (N_THREADS - 1)) / N_THREADS);

        updateIndexKernel_par << <updateBlocks, threads, k* N_THREADS * sizeof(int) >> > (dev_min_dist, dev_ndk, dev_clusterIdx, dev_delta);
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());

        // calculate delta
        delta = reduce(dev_delta, n);

        // Copy output vector from GPU buffer to host memory.
        CALL(cudaMemcpy(data.clusterIndex, dev_clusterIdx, n * sizeof(int), cudaMemcpyDeviceToHost));


        // reset clusterData
        //memset(data.clusterData, 0, sizeof(float) * k * d);
        memset(clusterCount, 0, sizeof(int) * k);

        for (int i = 0; i < k; i++)
        {
            stride = calculateStride(n, N_THREADS);
            float test[3 * 256];
            CALL(cudaMemset(dev_clusterSum, 0, k * stride * sizeof(float)));
            prepReduceForIndexKernel << <updateBlocks, threads, N_THREADS * d * sizeof(float) >> > (dev_clusterIdx, dev_data, dev_clusterSum, dev_ndk, stride, i);
            CALL(cudaGetLastError());
            CALL(cudaDeviceSynchronize());
            
            CALL(cudaMemcpy(test, dev_clusterSum, 3 * 256 * sizeof(float), cudaMemcpyDeviceToHost));
            
            
            //for (int i = 0; i < 3 * 256; i++)
            //    std::cout << test[i] << ", ";
            //
            //
            //std::cout << "\n\n";


            while (stride > N_THREADS)
            {
                int new_stride = calculateStride(stride, N_THREADS);
                dim3 reduceBlocks(k * stride / N_THREADS);
                reduceLocalSums << <reduceBlocks, threads, N_THREADS * sizeof(float) >> > (dev_clusterSum, stride, k * stride, new_stride);
                CALL(cudaGetLastError());
                CALL(cudaDeviceSynchronize());
                stride = new_stride;
                CALL(cudaMemcpy(test, dev_clusterSum, 3 * 256 * sizeof(float), cudaMemcpyDeviceToHost));
                
                
                //for (int i = 0; i < 3 * 256; i++)
                //    std::cout << test[i] << " ";
                //
                //
                //std::cout << "\n\n";
            } 
            reduceLocalSums << <k, threads, N_THREADS * sizeof(float) >> > (dev_clusterSum, stride, k * stride, 0);
            CALL(cudaGetLastError());
            CALL(cudaDeviceSynchronize());





            //reduceLocalSums << <updateBlocks, threads, N_THREADS * sizeof(float) >> > (dev_clusterSum, stride, d * k * stride, new_stride);
            //CALL(cudaGetLastError());
            //CALL(cudaDeviceSynchronize());
            CALL(cudaMemcpy(new_clusters + i * d, dev_clusterSum, d * sizeof(float), cudaMemcpyDeviceToHost));



        
            

        }

        //for (int i = 0; i < 9; i++)
        //    std::cout << data.clusterData[i] << " ";
        //std::cout << "\n\n";

        for (int i = 0; i < n; i++)
        {
            clusterCount[data.clusterIndex[i]]++;
        }

        for (int i = 0; i < k; i++)
        {
            //clusterCount[i] = count(dev_clusterIdx, n, i);
                for (int j = 0; j < d; j++)
                {
                    if (clusterCount[i] > 0)
                        data.clusterData[i * d + j] = new_clusters[i * d + j] / clusterCount[i];
                    
                    //std::cout << data.clusterData[i * d + j] << " ";
                }
        }
        //std::cout << "\n";
    } while ((float)delta / n > threshold);



    cudaFree(dev_clusterData);
    cudaFree(dev_clusterSum);
    cudaFree(dev_clusterIdx);
    cudaFree(dev_data);
    cudaFree(dev_delta);
    cudaFree(dev_ndk);
    cudaFree(dev_min_dist);

    delete clusterCount;
    delete new_clusters;
}

void Kmeans_par_NewClusterSumGPUWithThrust(PointsData_FlatArray& data, int kk)
{
    data.ndk[2] = kk;

    float* dev_data;
    float* dev_clusterData;
    int* dev_ndk, * dev_delta;
    float* dev_min_dist;
    int* dev_clusterIdx, *dev_idx_map;


    int n = data.ndk[0], d = data.ndk[1], k = data.ndk[2];
    int* clusterCount = new int[k];
    const int N_THREADS = 256;
    const float threshold = 1e-7;
    int delta = 0;

    thrust::device_vector<float> centroidSumKeys(d * k);
    thrust::device_vector<float> centroidSums(d * k);
    thrust::device_vector<float> data_thrust(d * n);
    thrust::device_vector<int> idxmap_thrust(d * n);

    // set device
    CALL(cudaSetDevice(0));

    // malloc
    CALL(cudaMalloc((void**)&dev_data, n * d * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_clusterData, d * k * sizeof(float)));
    CALL(cudaMalloc((void**)&dev_ndk, 3 * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_delta, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_clusterIdx, n * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_idx_map, n * d * sizeof(int)));
    CALL(cudaMalloc((void**)&dev_min_dist, n * k * sizeof(float)));


    // memset/memcpy
    CALL(cudaMemcpy(dev_data, data.data, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CALL(cudaMemcpy(dev_ndk, data.ndk, 3 * sizeof(int), cudaMemcpyHostToDevice));



    do
    {
        CALL(cudaMemcpy(dev_clusterData, data.clusterData, d * k * sizeof(float), cudaMemcpyHostToDevice));
        CALL(cudaMemcpy(dev_clusterIdx, data.clusterIndex, n * sizeof(int), cudaMemcpyHostToDevice));
        CALL(cudaMemset(dev_min_dist, 0, n * k * sizeof(float)));
        CALL(cudaMemset(dev_idx_map, 0, n * d * sizeof(int)));
        CALL(cudaMemset(dev_delta, 0, n * sizeof(int)));

        dim3 threads = dim3(N_THREADS);
        dim3 calcBlocks = dim3((n * k + (N_THREADS - 1)) / N_THREADS);

        calcDistKernel_par << <calcBlocks, threads, k* d * sizeof(float) >> > (dev_data, dev_clusterData, dev_ndk, dev_min_dist);
        // Check for any errors launching the kernel
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());

        dim3 updateBlocks = dim3((n + (N_THREADS - 1)) / N_THREADS);

        updateIndexKernel_par << <updateBlocks, threads, k* N_THREADS * sizeof(int) >> > (dev_min_dist, dev_ndk, dev_clusterIdx, dev_delta);
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());




        // Copy output vector from GPU buffer to host memory.
        CALL(cudaMemcpy(data.clusterIndex, dev_clusterIdx, n * sizeof(int), cudaMemcpyDeviceToHost));
        delta = reduce(dev_delta, n);


        createIndexMapKernel_thrust << <updateBlocks, threads >> > (dev_clusterIdx, dev_ndk, dev_idx_map);
        CALL(cudaGetLastError());
        CALL(cudaDeviceSynchronize());


        //thrust::device_vector<float> data_thrust(dev_data, dev_data + d * n);
        //thrust::device_vector<int> idxmap_thrust(dev_idx_map, dev_idx_map + d * n);

        thrust::copy(dev_data, dev_data + (d * n), data_thrust.begin());
        thrust::copy(dev_idx_map, dev_idx_map + (d * n), idxmap_thrust.begin());

        
        try
        {
            thrust::sort_by_key(idxmap_thrust.begin(), idxmap_thrust.end(), data_thrust.begin());
            thrust::reduce_by_key(idxmap_thrust.begin(), idxmap_thrust.end(), data_thrust.begin(), centroidSumKeys.begin(), centroidSums.begin());

        }
        catch (thrust::system_error err)
        {
            std::cout << err.what() << "\n";
        }


        memset(data.clusterData, 0, sizeof(float) * k * d);
        memset(clusterCount, 0, sizeof(int) * k);

        for (int i = 0; i < n; i++)
        {
            clusterCount[data.clusterIndex[i]]++;
        }







        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < d; j++)
            {
                data.clusterData[i * d + j] = centroidSums[i * d + j] / clusterCount[i];
            }
        }
    } while ((float)delta / n > threshold);



    cudaFree(dev_clusterData);
    cudaFree(dev_clusterIdx);
    cudaFree(dev_data);
    cudaFree(dev_delta);
    cudaFree(dev_ndk);
    cudaFree(dev_min_dist);
    cudaFree(dev_idx_map);

    delete clusterCount;
}

