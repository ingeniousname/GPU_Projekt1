#include "cudaError.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "clustering_par_reduce.cuh"
#include "clustering_par_common.cuh"
#include "reduce.cuh"
#include "Timer.h"

// ostatni krok redukcji 
__device__ void warpReducef(volatile float* shmem_ptr, int t, int offset) {
    shmem_ptr[offset + t] += shmem_ptr[offset + t + 32];
    shmem_ptr[offset + t] += shmem_ptr[offset + t + 16];
    shmem_ptr[offset + t] += shmem_ptr[offset + t + 8];
    shmem_ptr[offset + t] += shmem_ptr[offset + t + 4];
    shmem_ptr[offset + t] += shmem_ptr[offset + t + 2];
    shmem_ptr[offset + t] += shmem_ptr[offset + t + 1];
}

__global__ void prepReduceForIndexKernel(int* indicies, float* data, float* out_array, int* ndk, int stride, int idx)
{
    extern __shared__ float localSum[];
    int n = ndk[0], d = ndk[1], k = ndk[2];
    int tid = threadIdx.x;
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

    // pamiêæ wspó³dzielon¹ mo¿emy w tym przypadku interpretowaæ jako tablicê blockDim.x * k
    // w ka¿dej z rzêdów tej tablicy zapisujemy inne wspó³rzêdne i sumujemy je lokalnie
    for (int i = 0; i < d; i++)
    {
        if (global_idx < n)
        {
            // sprawdzamy, czy nale¿y do danego klastra, je¿eli tak przepisujemy jej wartoœæ do pamiêci wspó³dzielonej
            if (indicies[global_idx] == idx)
                localSum[i * blockDim.x + tid] = data[i * n + global_idx];
            else localSum[i * blockDim.x + tid] = 0;
        }
        else localSum[i * blockDim.x + tid] = 0;
        __syncthreads();

    }


    // sumowanie wspó³rzêdna po wspó³rzêdnej w pamiêci wspó³dzielonej
    for (int i = 0; i < d; i++)
    {
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (tid < s) {
                localSum[i * blockDim.x + tid] += localSum[i * blockDim.x + tid + s];
            }
            __syncthreads();
        }

        if (threadIdx.x < 32) {
            warpReducef(localSum, threadIdx.x, blockDim.x * i);
        }
    }


    // wypisanie danych do tablicy w odpowiednim formacie
    if (tid < d)
        out_array[stride * tid + blockIdx.x] = localSum[tid * blockDim.x];



    __syncthreads();
}

__global__ void reduceLocalSums(float* v, int stride, int size, int new_stride)
{
    // funkcja ta ró¿ni siê od funkcji redukcji z reduce.cu
    // zaledwie innym sposobem umiejscowienia lokalnych sum w tablicy
    // tak aby wspó³rzêdne siê nie pomiesza³y
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
    // chcemy aby przestrzeñ dla sum by³a liczb¹ wartoœci podzielona o rozmiar bloku (w tym obszarze liczymy lokaln¹ sumê)
    // i ¿eby by³a wielokrotnoœci¹ liczby 256
    int stride = std::max(n / N_THREADS, 1);
    stride = std::ceil((double)stride / (N_THREADS)) * N_THREADS;
    return stride;
}

void Kmeans_par_modifiedReduce(PointsData_SOA& data, int MAX_ITERS)
{
    // dane punktów
    float* dev_data;
    // dane centrów klastrów
    float* dev_clusterData;
    // parametry n, d, k
    int* dev_ndk;
    // tablica, w której zapisujemy 1, je¿eli zmieniliœmy centrum dla danego punktu
    int* dev_delta;
    // tablica odleg³oœci miêdzy punktami a centrami
    float* dev_dist;
    // tablica indeksów klastrów dla punktów (do którego klastra przynale¿y dany punkt)
    int* dev_clusterIdx;
    // tablica sum wspó³rzêdnych jakiegoœ klastra
    float* dev_clusterSum;


    int n = data.ndk[0], d = data.ndk[1], k = data.ndk[2];
    float* new_clusters = new float[d * k];
    int* clusterCount = new int[k];
    const int N_THREADS = calculateNumThreads(k);
    int stride = calculateStride(n, N_THREADS);
    int delta = 0;
    int iter = 0;

    // obliczanie czasu dzialania poszczegolnych etapow
    float time_dist = 0;
    float time_index = 0;
    float time_update_clusters = 0;

    // set device
    CALL(cudaSetDevice(0));

    {
        Timer_CPU t("poczatkowa inicjalizacja danych CPU -> GPU", true);
        CALL(cudaMalloc((void**)&dev_data, n * d * sizeof(float)));
        CALL(cudaMalloc((void**)&dev_clusterData, d * k * sizeof(float)));
        CALL(cudaMalloc((void**)&dev_ndk, 3 * sizeof(int)));
        CALL(cudaMalloc((void**)&dev_delta, n * sizeof(int)));
        CALL(cudaMalloc((void**)&dev_clusterIdx, n * sizeof(int)));
        CALL(cudaMalloc((void**)&dev_clusterSum, d * stride * sizeof(float)));
        CALL(cudaMalloc((void**)&dev_dist, n * k * sizeof(float)));


        // memset/memcpy
        CALL(cudaMemcpy(dev_data, data.data, n * d * sizeof(float), cudaMemcpyHostToDevice));
        CALL(cudaMemcpy(dev_ndk, data.ndk, 3 * sizeof(int), cudaMemcpyHostToDevice));
    }



    do
    {
        iter++;
        // inicjalizacja danych
        CALL(cudaMemcpy(dev_clusterData, data.clusterData, d * k * sizeof(float), cudaMemcpyHostToDevice));
        CALL(cudaMemcpy(dev_clusterIdx, data.clusterIndex, n * sizeof(int), cudaMemcpyHostToDevice));
        CALL(cudaMemset(dev_dist, 0, n * k * sizeof(float)));
        CALL(cudaMemset(dev_delta, 0, n * sizeof(int)));

        dim3 threads = dim3(N_THREADS);
        dim3 calcBlocks = dim3((n * k + (N_THREADS - 1)) / N_THREADS);

        // obliczanie odleg³oœci
        {
            Timer_CPU t("obliczanie odleglosci");
            calcDistKernel << <calcBlocks, threads, k* d * sizeof(float) >> > (dev_data, dev_clusterData, dev_ndk, dev_dist);
            CALL(cudaGetLastError());
            CALL(cudaDeviceSynchronize());
            time_dist += t.getElapsed();
        }

        dim3 updateBlocks = dim3((n + (N_THREADS - 1)) / N_THREADS);

        // obliczanie nowych indeksów klastrów dla punktów
        {
            Timer_CPU t("obliczanie nowych indeksow klastrow dla punktow");
            updateIndexKernel << <updateBlocks, threads, k* N_THREADS * sizeof(int) >> > (dev_dist, dev_ndk, dev_clusterIdx, dev_delta);
            CALL(cudaGetLastError());
            CALL(cudaDeviceSynchronize());
            time_index = t.getElapsed();
        }

        delta = reduce(dev_delta, n);

        CALL(cudaMemcpy(data.clusterIndex, dev_clusterIdx, n * sizeof(int), cudaMemcpyDeviceToHost));


        memset(clusterCount, 0, sizeof(int) * k);
        {
            Timer_CPU t("aktualizacja centrow klastrow");

            for (int i = 0; i < k; i++)
            {
                stride = calculateStride(n, N_THREADS);
                CALL(cudaMemset(dev_clusterSum, 0, d * stride * sizeof(float)));
                // przygotowanie tablicy do redukcji
                prepReduceForIndexKernel << <updateBlocks, threads, N_THREADS* d * sizeof(float) >> > (dev_clusterIdx, dev_data, dev_clusterSum, dev_ndk, stride, i);


                while (stride > N_THREADS)
                {
                    int new_stride = calculateStride(stride, N_THREADS);
                    dim3 reduceBlocks(k * stride / N_THREADS);
                    // sumowanie po wspó³rzêdnych dla centrum klastra i-tego
                    reduceLocalSums << <reduceBlocks, threads, N_THREADS * sizeof(float) >> > (dev_clusterSum, stride, k * stride, new_stride);
                    stride = new_stride;
                }
                // ostatnia suma, po której na pocz¹tku tablicy dev_clusterSum znajduj¹ siê sumy wspó³rzêdnych i-tego klastra
                reduceLocalSums << <k, threads, N_THREADS * sizeof(float) >> > (dev_clusterSum, stride, k * stride, 0);
                CALL(cudaGetLastError());
                CALL(cudaDeviceSynchronize());

                CALL(cudaMemcpy(new_clusters + i * d, dev_clusterSum, d * sizeof(float), cudaMemcpyDeviceToHost));

            }


            //for (int i = 0; i < k; i++)
            //{
            //    //clusterCount[data.clusterIndex[i]]++;
            //    count(dev_clusterIdx, n, i);
            //}

            // aktualizacja centrów klastrów
            count(dev_clusterIdx, clusterCount, n, k);
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    if (clusterCount[i] > 0)
                        data.clusterData[i * d + j] = new_clusters[i * d + j] / clusterCount[i];
                }
            }
            time_update_clusters += t.getElapsed();
        }

    } while (delta > 0 && iter < MAX_ITERS);


    std::cout << "Ilosc iteracji: " << iter << ".\n";
    std::cout << "delta = " << delta << ".\n";
    std::cout << "Srednie obliczanie odleglosci od punktow do centrow klastrow: " << time_dist / iter << "s.\n";
    std::cout << "Srednie obliczanie nowych indeksow centrow dla punktow: " << time_index / iter << "s.\n";
    std::cout << "Srednie obliczanie nowych centrow klastrow: " << time_update_clusters / iter << "s.\n";


    // zwalnianie pamiêci
    cudaFree(dev_clusterData);
    cudaFree(dev_clusterSum);
    cudaFree(dev_clusterIdx);
    cudaFree(dev_data);
    cudaFree(dev_delta);
    cudaFree(dev_ndk);
    cudaFree(dev_dist);

    delete clusterCount;
    delete new_clusters;
}