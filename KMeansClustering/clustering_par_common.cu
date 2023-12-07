#include "cudaError.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "reduce.cuh"
#include "clustering_par_common.cuh"
#include "Timer.h"

__global__ void calcDistKernel(float* data, float* clusterData, int* ndk, float* dist)
{
    extern __shared__ float clusterData_shared[];
    int res_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = ndk[0], d = ndk[1], k = ndk[2];
    int tid = threadIdx.x;

    // przepisz dane centrów do pamiêci wspó³dzielonej
    if (tid < d * k)
        clusterData_shared[tid] = clusterData[tid];

    __syncthreads();

    int data_idx = res_idx / k;
    int c_idx = res_idx % k;
    if (res_idx < n * k)
    {
        float v = 0.f;
        // w pêtli kolejno obliczamy kwadrat ró¿nicy dla ka¿dej wspó³rzêdnej
        // wynik zostawiamy w kwadracie dla optymalizacji
        for (int i = 0; i < d; i++)
        {
            float part_v = data[i * n + data_idx] - clusterData_shared[c_idx * d + i];
            v += part_v * part_v;
        }
        // zapisujemy wyniki w tablicy dist
        dist[res_idx] = v;
    }

    __syncthreads();
}

__global__ void updateIndexKernel(float* min_dist, int* ndk, int* indicies, int* delta)
{
    extern __shared__ float min_dist_shared[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = ndk[0], d = ndk[1], k = ndk[2];

    int tid = threadIdx.x;

    // przepisujemy do pamiêci wspó³dzielonej odleg³oœci miêdzy centrami a punktami o indeksach: 
    // [blockIdx.x * blockDim.x, blockIdx.x * blockDim.x + blockDim.x - 1]
    for (int i = 0; i < k; i++)
    {
        int out_idx = blockDim.x * blockIdx.x * k + blockDim.x * i + tid;
        if (out_idx < n * k)
            min_dist_shared[blockDim.x * i + tid] = min_dist[out_idx];
        __syncthreads();

    }

    if (idx < n)
    {
        // znajdujemy najbli¿szy klaster
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

        // je¿eli centrum siê zmieni³o, zapisz nowe centrum i dodaj do delty
        if (min_idx != last_idx)
        {
            indicies[idx] = min_idx;
            delta[idx] = 1;
        }
    }

    __syncthreads();

}

int calculateNumThreads(int k)
{
    return std::max(32, 256 / (1 << (k - 1) / 48));
}

void Kmeans_par_UpdateClusterOnCPU(PointsData_SOA& data, int MAX_ITERS)
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


    int n = data.ndk[0], d = data.ndk[1], k = data.ndk[2];
    float* old_clusters = new float[d * k];
    int* clusterCount = new int[k];
    const int N_THREADS = calculateNumThreads(k);
    int delta = 0;
    int iter = 0;

    // obliczanie czasu dzialania poszczegolnych etapow
    float time_dist = 0;
    float time_index = 0;
    float time_update_clusters = 0;


    CALL(cudaSetDevice(0));

    // malloc
    {
        Timer_CPU t("poczatkowa inicjalizacja danych CPU -> GPU", true);
        CALL(cudaMalloc((void**)&dev_data, n * d * sizeof(float)));
        CALL(cudaMalloc((void**)&dev_clusterData, d * k * sizeof(float)));
        CALL(cudaMalloc((void**)&dev_ndk, 3 * sizeof(int)));
        CALL(cudaMalloc((void**)&dev_delta, n * sizeof(int)));
        CALL(cudaMalloc((void**)&dev_clusterIdx, n * sizeof(int)));
        CALL(cudaMalloc((void**)&dev_dist, n * k * sizeof(float)));


        // memset/memcpy
        CALL(cudaMemcpy(dev_data, data.data, n * d * sizeof(float), cudaMemcpyHostToDevice));
        CALL(cudaMemcpy(dev_ndk, data.ndk, 3 * sizeof(int), cudaMemcpyHostToDevice));
    }


    do
    {
        iter++;
        memcpy(old_clusters, data.clusterData, d * k * sizeof(float));

        // inicjalizacja danych
        {
            Timer_CPU t("inicjalizacja danych CPU -> GPU");
            CALL(cudaMemcpy(dev_clusterData, data.clusterData, d * k * sizeof(float), cudaMemcpyHostToDevice));
            CALL(cudaMemcpy(dev_clusterIdx, data.clusterIndex, n * sizeof(int), cudaMemcpyHostToDevice));
            CALL(cudaMemset(dev_dist, 0, n * k * sizeof(float)));
            CALL(cudaMemset(dev_delta, 0, n * sizeof(int)));
        }

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


        CALL(cudaMemcpy(data.clusterIndex, dev_clusterIdx, n * sizeof(int), cudaMemcpyDeviceToHost));

        // obliczanie delty
        {
            Timer_CPU("obliczanie delty");
            delta = reduce(dev_delta, n);
        }

        memset(data.clusterData, 0, sizeof(float) * k * d);
        memset(clusterCount, 0, sizeof(int) * k);

        // aktualizacja centrów klastrów
        {
            Timer_CPU t("aktualizacja centrow klastrow");
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
                for (int j = 0; j < d; j++)
                {
                    if (clusterCount[i] > 0)
                        data.clusterData[i * d + j] /= clusterCount[i];
                    else data.clusterData[i * d + j] = old_clusters[i * d + j];
                }

            }
            time_update_clusters = t.getElapsed();
        }
    } while (delta > 0 && iter < MAX_ITERS);

    std::cout << "Ilosc iteracji: " << iter << ".\n";
    std::cout << "delta = " << delta << ".\n";
    std::cout << "Srednie obliczanie odleglosci od punktow do centrow klastrow: " << time_dist / iter << "s.\n";
    std::cout << "Srednie obliczanie nowych indeksow centrow dla punktow: " << time_index / iter << "s.\n";
    std::cout << "Srednie obliczanie nowych centrow klastrow: " << time_update_clusters / iter << "s.\n";


    // zwalnianie pamiêci
    cudaFree(dev_clusterData);
    cudaFree(dev_clusterIdx);
    cudaFree(dev_data);
    cudaFree(dev_delta);
    cudaFree(dev_ndk);
    cudaFree(dev_dist);

    delete old_clusters;
    delete clusterCount;
}
